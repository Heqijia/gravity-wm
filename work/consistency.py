"""Reconcile the reported rollout loss with the latent statistics.

The trainer reports a validation rollout loss of 0.088 per dimension against a
latent whose per-dimension variance measures 0.675, which would mean the
predictor explains most of the latent -- yet nearest-neighbour decoding says
the latent carries no position information at all.  One of those readings is
wrong.  This recomputes the loss the trainer reports, from the same latents the
probe sees, alongside the trivial baselines it has to beat.
"""
import sys
import numpy as np, torch
from sg_jepa.data import ActionStatistics, open_trajectory_store
from gwm.constraint import ConstraintSpec, rollout_latents
from gwm.latents import load_world_model

ckpt, data = sys.argv[1], sys.argv[2]
dev = torch.device('cuda')
model, mcfg, stats, cfg = load_world_model(ckpt, device=dev)
model.eval()
store = open_trajectory_store(data)
im = torch.tensor((0.485,0.456,0.406)).reshape(1,3,1,1); istd = torch.tensor((0.229,0.224,0.225)).reshape(1,3,1,1)
amean = torch.tensor(stats.mean); astd = torch.tensor(stats.std)
idx = np.flatnonzero(store.split_id == 0)[:400]
Z, A = [], []
with torch.no_grad():
    for b in range(0, len(idx), 24):
        px, ac = [], []
        for i in idx[b:b+24]:
            e = store.episode(int(i))
            t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(e.pixels,-1,1))).float().div_(255.)
            px.append((t-im)/istd)
            ac.append((torch.from_numpy(e.action.astype(np.float32)) - amean) / astd)
        out = model.encode({'pixels': torch.stack(px).to(dev), 'action': torch.stack(ac).to(dev)})
        Z.append(out['emb'].float()); A.append(torch.stack(ac).to(dev))
Z = torch.cat(Z); A = torch.cat(A)
H, K = mcfg.history_size, 6
print(f"readout={cfg.get('readout')}  episodes={Z.shape[0]} frames={Z.shape[1]} dim={Z.shape[2]}")
var = Z.reshape(-1, Z.shape[-1]).var(dim=0)
print(f"latent per-dim variance: mean {float(var.mean()):.4f}  median {float(var.median()):.4f}  "
      f"max {float(var.max()):.4f}  min {float(var.min()):.4f}")

with torch.no_grad():
    cond = model.action_encoder(A)
    pred = rollout_latents(model, Z, cond, history=H, horizon=K)
    tgt = Z[:, H:H+K]
    mse_dim = (pred - tgt).pow(2).mean()                       # what the trainer reports
    # trivial baselines the predictor must beat
    copy_last = (Z[:, H-1:H].expand_as(tgt) - tgt).pow(2).mean()
    global_mean = (Z.reshape(-1, Z.shape[-1]).mean(0) - tgt).pow(2).mean()
    linear_extrap = ((2*Z[:, H-1:H] - Z[:, H-2:H-1]).expand_as(tgt) - tgt).pow(2).mean()
print(f"  rollout MSE (per dim)        {float(mse_dim):.4f}   <- compare with the reported value")
print(f"  copy z_t                     {float(copy_last):.4f}")
print(f"  predict the global mean      {float(global_mean):.4f}   <- equals the variance")
print(f"  linear extrapolation         {float(linear_extrap):.4f}")
print(f"  variance explained by rollout {1 - float(mse_dim/global_mean):+.4f}")
