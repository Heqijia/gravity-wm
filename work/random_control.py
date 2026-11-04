"""Does training destroy the position information, or was it never readable?

A randomly initialised ViT is the control: its patch tokens certainly encode
where the object is, so if its projected [CLS] latent decodes position and the
trained one does not, training removed it.  If neither does, the [CLS]/projector
read-out never carried it and the pipeline is fine.
"""
import sys
import numpy as np, torch, torch.nn.functional as F
from sg_jepa.data import open_trajectory_store
from gwm.latents import load_world_model
from gwm.probe import ProbeConfig, build_position_probe
from gwm.spatial_lewm import build_readout_world_model

ckpt, data = sys.argv[1], sys.argv[2]
n_fit = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
dev = torch.device('cuda')
trained, mcfg, _s, cfg = load_world_model(ckpt, device=dev)
torch.manual_seed(12345)
random_model = build_readout_world_model(mcfg, 'cls').to(dev)
store = open_trajectory_store(data)
md = store.metadata; ppu = md['pixels_shape'][0] / md['scene']['view_height']
im = torch.tensor((0.485,0.456,0.406)).reshape(1,3,1,1); istd = torch.tensor((0.229,0.224,0.225)).reshape(1,3,1,1)

def enc(model, indices, patch=False):
    model.eval(); model.store_patch_tokens = patch
    Z, S = [], []
    with torch.no_grad():
        for b in range(0, len(indices), 24):
            px, st = [], []
            for i in indices[b:b+24]:
                e = store.episode(int(i))
                t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(e.pixels,-1,1))).float().div_(255.)
                px.append((t-im)/istd); st.append(torch.from_numpy(e.state.astype(np.float32)))
            info = model.encode({'pixels': torch.stack(px).to(dev)})
            Z.append((info['patch_tokens'].mean(-1) if patch else info['emb']).float())
            S.append(torch.stack(st).to(dev))
    return torch.cat(Z), torch.cat(S)

idx = np.flatnonzero(store.split_id == 0)
fit_i, test_i = idx[:n_fit], idx[n_fit:n_fit+300]

def probe(model, patch, tag):
    Zf, Sf = enc(model, fit_i, patch); Zt, St = enc(model, test_i, patch)
    Xf = Zf.reshape(-1, Zf.shape[-1]); Yf = Sf.reshape(-1, 8)[:, :2]
    Xt = Zt.reshape(-1, Zt.shape[-1]); Yt = St.reshape(-1, 8)[:, :2]
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
    print(f"  {tag:<40} {best:.4f} world = {best*ppu:6.2f} px   var_expl {1-(best/base)**2:+.4f}")

print(f"readout={cfg.get('readout')}  fit {len(fit_i)} eps  test {len(test_i)} eps")
probe(random_model, False, "random encoder, emb ([CLS] -> projector)")
probe(trained,      False, "trained encoder, emb ([CLS] -> projector)")
probe(random_model, True,  "random encoder, patch map")
probe(trained,      True,  "trained encoder, patch map")
