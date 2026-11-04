"""De-risk before committing GPU hours: graft the new read-out onto the
already-trained ViT and probe the latent it produces.

The projector is random here, so this measures what the read-out *passes
through* rather than what a trained projector could do.  A random projection
preserves distances, so if position is readable after grafting it will be
readable after training; if it is not, the read-out is not worth a run.
"""
import sys
import numpy as np, torch, torch.nn.functional as F
from sg_jepa.data import open_trajectory_store
from gwm.latents import load_world_model
from gwm.probe import ProbeConfig, build_position_probe
from gwm.spatial_lewm import build_readout_world_model

ckpt, data = sys.argv[1], sys.argv[2]
dev = torch.device('cuda')
trained, mcfg, _s, cfg = load_world_model(ckpt, device=dev)
store = open_trajectory_store(data)
md = store.metadata; ppu = md['pixels_shape'][0] / md['scene']['view_height']
im = torch.tensor((0.485,0.456,0.406)).reshape(1,3,1,1); istd = torch.tensor((0.229,0.224,0.225)).reshape(1,3,1,1)
idx = np.flatnonzero(store.split_id == 0)

def build(readout):
    torch.manual_seed(999)
    m = build_readout_world_model(mcfg, readout).to(dev)
    # graft the trained ViT; everything else stays at its initialisation
    m.encoder.load_state_dict(trained.encoder.state_dict(), strict=True)
    return m.eval()

def enc(model, indices):
    Z, S = [], []
    with torch.no_grad():
        for b in range(0, len(indices), 24):
            px, st = [], []
            for i in indices[b:b+24]:
                e = store.episode(int(i))
                t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(e.pixels,-1,1))).float().div_(255.)
                px.append((t-im)/istd); st.append(torch.from_numpy(e.state.astype(np.float32)))
            Z.append(model.encode({'pixels': torch.stack(px).to(dev)})['emb'].float())
            S.append(torch.stack(st).to(dev))
    return torch.cat(Z).reshape(-1, mcfg.embed_dim), torch.cat(S).reshape(-1, 8)[:, :2]

def probe(model, tag):
    Xf, Yf = enc(model, idx[:1200]); Xt, Yt = enc(model, idx[1200:1500])
    mu, sd = Yf.mean(0), Yf.std(0).clamp_min(1e-6)
    base = float((Yt - mu).pow(2).sum(-1).mean().sqrt())
    torch.manual_seed(0)
    net = build_position_probe(Xf.shape[1], ProbeConfig(input_norm=False)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    n = Xf.shape[0]; best = 1e9
    for ep in range(300):
        net.train(); perm = torch.randperm(n, device=dev)
        for k in range(0, n, 4096):
            j = perm[k:k+4096]
            loss = F.mse_loss(net(Xf[j]), (Yf[j]-mu)/sd)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            best = min(best, float((net(Xt)*sd+mu - Yt).pow(2).sum(-1).mean().sqrt()))
    print(f"  {tag:<44} {best:.4f} world = {best*ppu:6.2f} px   var_expl {1-(best/base)**2:+.4f}")

print(f"trained ViT from a {cfg.get('readout')} run, grafted onto each read-out")
probe(trained, "cls (the trained run itself, trained projector)")
for r in ("cls", "attn_pool", "spatial_map"):
    probe(build(r), f"{r} (grafted ViT, random projector)")
