"""Is position linearly decodable from the cached latents at all?"""
import torch, numpy as np
from gwm.latents import load_latent_cache

cache = load_latent_cache('/fs/scratch/PAS3151/qijia387/gwm/runs/smoke_down/latents.pt')
dev = cache.select(cache.split == 0)
n_ep = min(3000, dev.episodes)
Z = dev.latents[:n_ep].reshape(-1, dev.latent_dim).double()
Y = dev.state[:n_ep].reshape(-1, dev.state.shape[-1])[:, :2].double()
print(f"frames={Z.shape[0]}  latent_dim={Z.shape[1]}")
print(f"latent: mean|.|={Z.abs().mean():.4f}  std={Z.std():.4f}  per-sample norm mean={Z.norm(dim=1).mean():.3f} std={Z.norm(dim=1).std():.3f}")
print(f"position std per axis: {Y.std(0).tolist()}   scale(rms)={float(Y.pow(2).sum(-1).mean().sqrt()):.4f}")

split = int(0.8 * Z.shape[0])
perm = torch.randperm(Z.shape[0], generator=torch.Generator().manual_seed(0))
tr, te = perm[:split], perm[split:]

def ridge(X, Yt, Xe, Ye, lam):
    X1 = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], 1)
    Xe1 = torch.cat([Xe, torch.ones(Xe.shape[0], 1, dtype=Xe.dtype)], 1)
    A = X1.T @ X1 + lam * torch.eye(X1.shape[1], dtype=X.dtype)
    W = torch.linalg.solve(A, X1.T @ Yt)
    pred = Xe1 @ W
    rmse = float((pred - Ye).pow(2).sum(-1).mean().sqrt())
    base = float((Ye - Yt.mean(0)).pow(2).sum(-1).mean().sqrt())
    return rmse, base

for name, feat in [("raw latent", Z),
                   ("layer-normed", torch.nn.functional.layer_norm(Z.float(), (Z.shape[1],)).double()),
                   ("standardized/dim", (Z - Z.mean(0)) / Z.std(0).clamp_min(1e-8))]:
    best = None
    for lam in [1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0]:
        rmse, base = ridge(feat[tr], Y[tr], feat[te], Y[te], lam)
        if best is None or rmse < best[0]: best = (rmse, lam, base)
    rmse, lam, base = best
    print(f"  {name:<18} ridge RMSE={rmse:.4f} world = {rmse*cache.pixels_per_unit:6.2f} px "
          f"(predict-mean baseline {base:.4f})  lam={lam}")
