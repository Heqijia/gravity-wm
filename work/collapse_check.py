"""Is the latent collapsing to episode identity?

The rollout loss takes its targets from the same trainable encoder with no
stop-gradient, so the encoder can lower it by making the latent easy to
predict -- and the easiest latent is one that barely moves within an episode.
SIGReg does not prevent this: it constrains the marginal over *all* frames, and
a code that is constant within an episode and Gaussian across episodes
satisfies it exactly.  Position, which varies within an episode, would then be
actively discarded.

The test is a variance decomposition: split the latent's total variance into
the part explained by *which episode* a frame belongs to and the part that
varies *within* an episode.  Position lives entirely in the within part.
"""
import sys, json
import numpy as np, torch
from sg_jepa.data import open_trajectory_store
from gwm.latents import load_world_model

ckpt, data = sys.argv[1], sys.argv[2]
n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 400
dev = torch.device('cuda')
model, _c, _s, cfg = load_world_model(ckpt, device=dev); model.eval()
store = open_trajectory_store(data)
im = torch.tensor((0.485,0.456,0.406)).reshape(1,3,1,1); istd = torch.tensor((0.229,0.224,0.225)).reshape(1,3,1,1)
idx = np.flatnonzero(store.split_id == 0)[:n_ep]
Z, S = [], []
with torch.no_grad():
    for b in range(0, len(idx), 24):
        px, st = [], []
        for i in idx[b:b+24]:
            e = store.episode(int(i))
            t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(e.pixels,-1,1))).float().div_(255.)
            px.append((t-im)/istd); st.append(torch.from_numpy(e.state.astype(np.float32)))
        Z.append(model.encode({'pixels': torch.stack(px).to(dev)})['emb'].float().cpu())
        S.append(torch.stack(st))
Z = torch.cat(Z).double(); S = torch.cat(S).double()      # (E, T, D), (E, T, 8)
E, T, D = Z.shape
print(f"readout={cfg.get('readout')}  episodes={E} frames={T} dim={D}")

per_ep_mean = Z.mean(dim=1, keepdim=True)                  # (E, 1, D)
within = (Z - per_ep_mean).pow(2).mean(dim=(0, 1)).sum()   # variance inside episodes
between = (per_ep_mean.squeeze(1) - Z.reshape(-1, D).mean(0)).pow(2).mean(0).sum()
total = (Z.reshape(-1, D) - Z.reshape(-1, D).mean(0)).pow(2).mean(0).sum()
print(f"latent variance   total {float(total):.4f}")
print(f"  between episodes {float(between):.4f}  ({100*float(between/total):.1f}%)  <- episode identity")
print(f"  within  episodes {float(within):.4f}  ({100*float(within/total):.1f}%)  <- where position must live")

# the same decomposition for the thing the latent is supposed to track
P = S[..., :2]
p_ep = P.mean(dim=1, keepdim=True)
w = (P - p_ep).pow(2).mean(dim=(0,1)).sum(); b = (p_ep.squeeze(1) - P.reshape(-1,2).mean(0)).pow(2).mean(0).sum()
print(f"position variance total {float(w+b):.4f}  between {100*float(b/(w+b)):.1f}%  within {100*float(w/(w+b)):.1f}%")

# consecutive-frame latent change, relative to the spread across episodes
step = (Z[:, 1:] - Z[:, :-1]).pow(2).sum(-1).mean().sqrt()
spread = (Z.reshape(-1, D) - Z.reshape(-1, D).mean(0)).pow(2).sum(-1).mean().sqrt()
print(f"||z_(t+1) - z_t|| = {float(step):.4f}   ||z - mean z|| = {float(spread):.4f}   ratio {float(step/spread):.4f}")
json.dump({'readout': cfg.get('readout'), 'between_frac': float(between/total),
           'within_frac': float(within/total), 'step_ratio': float(step/spread)},
          open('/users/PAS3151/qijia387/Phys-WM/work/collapse_%s.json' % cfg.get('readout'), 'w'), indent=2)
