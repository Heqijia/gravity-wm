"""Where is position lost: in the ViT features, or in the CLS/projector readout?

The constraint can only act on ``emb`` -- the projected CLS token the predictor
outputs -- so that is what limits the position probe.  But if the ViT's patch
tokens carry position far better than the CLS token does, the bottleneck is the
readout rather than the backbone, which is a different problem with a different
fix.  Ridge regression gives an upper bound on what any probe of each feature
could achieve.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from sg_jepa.data import open_trajectory_store
from gwm.latents import load_world_model

sys.path.insert(0, '/users/PAS3151/qijia387/Phys-WM/gravity-wm/scripts')
from diagnose import ridge_decodability

ckpt, data, n_ep = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = torch.device('cuda')
model, cfg, stats, _ = load_world_model(ckpt, device=dev)
model.eval()
model.store_cls_raw = True
model.store_patch_tokens = True

store = open_trajectory_store(data)
held = np.flatnonzero(store.split_id == 1)[:n_ep]
im = torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
istd = torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
feats = {k: [] for k in ('emb', 'cls_raw', 'patch_mean', 'patch_flat')}
Y = []
EP = []
with torch.no_grad():
    for b in range(0, len(held), 16):
        px, st = [], []
        for i in held[b:b + 16]:
            e = store.episode(int(i))
            t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(e.pixels, -1, 1))).float().div_(255.)
            px.append((t - im) / istd)
            st.append(torch.from_numpy(e.state.astype(np.float32)))
        info = model.encode({'pixels': torch.stack(px).to(dev)})
        pt = info['patch_tokens']                      # (B, T, P, D)
        feats['emb'].append(info['emb'].reshape(-1, info['emb'].shape[-1]).float().cpu())
        feats['cls_raw'].append(info['cls_raw'].reshape(-1, info['cls_raw'].shape[-1]).float().cpu())
        feats['patch_mean'].append(pt.mean(dim=2).reshape(-1, pt.shape[-1]).float().cpu())
        # A spatial readout: project each patch to a scalar then keep the map,
        # which is the cheapest feature that can express "where".
        feats['patch_flat'].append(pt.mean(dim=-1).reshape(-1, pt.shape[2]).float().cpu())
        Y.append(torch.stack(st).reshape(-1, 8)[:, :2])
        EP.append(np.repeat(np.arange(b, b + len(st)), st[0].shape[0]))

Y = torch.cat(Y).double()
EPI = np.concatenate(EP)
md = store.metadata
ppu = md['pixels_shape'][0] / md['scene']['view_height']
lam = [1e-4, 1e-2, 1., 1e1, 1e2, 1e3, 1e4]
out = {'frames': int(Y.shape[0]), 'pixels_per_unit': ppu,
       'centred_position_scale': float((Y - Y.mean(0)).pow(2).sum(-1).mean().sqrt())}
print(f"frames {out['frames']}   centred position scale "
      f"{out['centred_position_scale']:.3f} world = {out['centred_position_scale']*ppu:.1f} px")
print(f"{'feature':<12} {'dim':>5} {'ridge RMSE':>11} {'px':>8} {'var expl':>10}")
for name in ('emb', 'cls_raw', 'patch_mean', 'patch_flat'):
    Z = torch.cat(feats[name]).double()
    r = ridge_decodability(Z, Y, lambdas=lam, episode_ids=EPI)
    out[name] = {'dim': int(Z.shape[1]), 'rmse_world': r['ridge_rmse_world'],
                 'rmse_px': r['ridge_rmse_world'] * ppu, 'var_explained': r['variance_explained']}
    print(f"{name:<12} {Z.shape[1]:>5} {r['ridge_rmse_world']:>11.4f} "
          f"{r['ridge_rmse_world']*ppu:>8.2f} {r['variance_explained']:>+10.4f}")
json.dump(out, open('/users/PAS3151/qijia387/Phys-WM/work/readout_probe.json', 'w'), indent=2)
