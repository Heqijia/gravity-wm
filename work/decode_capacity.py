"""Is position *absent* from the latent, or just hard for a small probe?

The rollout loss reaches 0.088 per dimension against a latent whose per-dimension
variance is 0.675, so the predictor explains most of the latent -- the code is a
learnable, deterministic function of the smooth state.  That is compatible with
the code being injective but highly oscillatory, in which case position is
present and a *stronger* decoder recovers it.  Three decoders of increasing
power settle the question; the probe only has to be frozen and accurate, so a
stronger one is entirely legitimate.
"""
import sys
import numpy as np, torch, torch.nn.functional as F
from sg_jepa.data import open_trajectory_store
from gwm.latents import load_world_model
from gwm.probe import ProbeConfig, build_position_probe

ckpt, data = sys.argv[1], sys.argv[2]
n_fit = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
dev = torch.device('cuda')
model, _c, _s, cfg = load_world_model(ckpt, device=dev); model.eval()
store = open_trajectory_store(data)
md = store.metadata; ppu = md['pixels_shape'][0] / md['scene']['view_height']
im = torch.tensor((0.485,0.456,0.406)).reshape(1,3,1,1); istd = torch.tensor((0.229,0.224,0.225)).reshape(1,3,1,1)

def enc(indices):
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
    return torch.cat(Z), torch.cat(S)

dev_idx = np.flatnonzero(store.split_id == 0)
Zf, Sf = enc(dev_idx[:n_fit]); Zt, St = enc(dev_idx[n_fit:n_fit+300])   # episode-disjoint
Xf = Zf.reshape(-1, Zf.shape[-1]); Yf = Sf.reshape(-1, 8)[:, :2]
Xt = Yt = None
Xt = Zt.reshape(-1, Zt.shape[-1]); Yt = St.reshape(-1, 8)[:, :2]
base = float((Yt - Yf.mean(0)).pow(2).sum(-1).mean().sqrt())
print(f"readout={cfg.get('readout')}  fit {Xf.shape[0]} frames  test {Xt.shape[0]} frames")
print(f"  predict-mean baseline            {base:.4f} world = {base*ppu:6.2f} px")

# 1. nearest neighbour in latent space: tests injectivity, no fitting at all
with torch.no_grad():
    pred = torch.empty_like(Yt)
    for b in range(0, Xt.shape[0], 512):
        d = torch.cdist(Xt[b:b+512], Xf)
        pred[b:b+512] = Yf[d.argmin(1)]
    knn1 = float((pred - Yt).pow(2).sum(-1).mean().sqrt())
    for k in (8,):
        pr = torch.empty_like(Yt)
        for b in range(0, Xt.shape[0], 512):
            d = torch.cdist(Xt[b:b+512], Xf)
            pr[b:b+512] = Yf[d.topk(k, largest=False).indices].mean(1)
        knnk = float((pr - Yt).pow(2).sum(-1).mean().sqrt())
print(f"  1-nearest neighbour              {knn1:.4f} world = {knn1*ppu:6.2f} px")
print(f"  8-nearest neighbours             {knnk:.4f} world = {knnk*ppu:6.2f} px")

# 2. the study's probe (512-256 MLP) and 3. a much larger one
mu, sd = Yf.mean(0), Yf.std(0).clamp_min(1e-6)
for tag, hidden, epochs in [("MLP 512-256", (512, 256), 400), ("MLP 4096-2048", (4096, 2048), 400)]:
    torch.manual_seed(0)
    net = build_position_probe(Xf.shape[1], ProbeConfig(input_norm=False, hidden=hidden)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xf.shape[0]; bs = 4096; best = 1e9
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=dev)
        for k in range(0, n, bs):
            j = perm[k:k+bs]
            loss = F.mse_loss(net(Xf[j]), (Yf[j]-mu)/sd)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            best = min(best, float((net(Xt)*sd+mu - Yt).pow(2).sum(-1).mean().sqrt()))
    params = sum(p.numel() for p in net.parameters())
    print(f"  {tag:<32} {best:.4f} world = {best*ppu:6.2f} px   ({params/1e6:.2f}M params)")
