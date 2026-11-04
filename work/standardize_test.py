"""Does per-dimension standardisation of the latent rescue the probe?

The latent's per-dimension variance spans a factor of 28, so without
standardisation the first linear layer is dominated by a few directions.  The
probe's two existing options are no normalisation and LayerNorm, which is
*per-sample* and therefore removes each latent's own scale -- neither is the
standard per-feature standardisation.  This rules it in or out.
"""
import sys
import numpy as np, torch, torch.nn.functional as F
from sg_jepa.data import open_trajectory_store
from gwm.latents import load_world_model
from gwm.probe import ProbeConfig, build_position_probe

ckpt, data = sys.argv[1], sys.argv[2]
dev = torch.device('cuda')
model, _m, _s, cfg = load_world_model(ckpt, device=dev); model.eval()
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
    return torch.cat(Z).reshape(-1, 256), torch.cat(S).reshape(-1, 8)[:, :2]

idx = np.flatnonzero(store.split_id == 0)
Xf, Yf = enc(idx[:1200]); Xt, Yt = enc(idx[1200:1500])
mu, sd = Yf.mean(0), Yf.std(0).clamp_min(1e-6)
base = float((Yt - mu).pow(2).sum(-1).mean().sqrt())
fmu, fsd = Xf.mean(0), Xf.std(0).clamp_min(1e-8)
print(f"latent per-dim std: min {float(fsd.min()):.4f} max {float(fsd.max()):.4f}")
print(f"predict-mean baseline {base:.4f} world = {base*ppu:.2f} px")

for tag, (A, B) in {
    "raw": (Xf, Xt),
    "per-dim standardised": ((Xf-fmu)/fsd, (Xt-fmu)/fsd),
    "whitened (ZCA)": (None, None),
}.items():
    if A is None:
        C = torch.cov(((Xf-fmu)).T.double()) + 1e-3*torch.eye(256, device=dev, dtype=torch.float64)
        L = torch.linalg.cholesky(torch.linalg.inv(C)).float()
        A, B = (Xf-fmu) @ L, (Xt-fmu) @ L
    torch.manual_seed(0)
    net = build_position_probe(256, ProbeConfig(input_norm=False)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    n = A.shape[0]; best = 1e9
    for ep in range(300):
        net.train(); perm = torch.randperm(n, device=dev)
        for k in range(0, n, 4096):
            j = perm[k:k+4096]
            loss = F.mse_loss(net(A[j]), (Yf[j]-mu)/sd)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            best = min(best, float((net(B)*sd+mu - Yt).pow(2).sum(-1).mean().sqrt()))
    print(f"  {tag:<24} {best:.4f} world = {best*ppu:6.2f} px   var_expl {1-(best/base)**2:+.4f}")
