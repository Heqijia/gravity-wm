"""Encode a subset of episodes from a checkpoint and report position decodability."""
import sys, json
from pathlib import Path
import numpy as np, torch
from sg_jepa.data import open_trajectory_store, ActionStatistics
from gwm.latents import load_world_model
sys.path.insert(0, str(Path('/users/PAS3151/qijia387/Phys-WM/gravity-wm/scripts')))
from diagnose import ridge_decodability

ckpt, data, n_ep = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = torch.device('cuda')
model, cfg, stats, _ = load_world_model(ckpt, device=dev)
model.eval()
store = open_trajectory_store(data)
# held-out episodes only, so decodability is not measured on probe-training frames
held = np.flatnonzero(store.split_id == 1)[:n_ep]
if len(held) < n_ep:
    held = np.concatenate([held, np.flatnonzero(store.split_id == 0)[: n_ep - len(held)]])
im = torch.tensor((0.485,0.456,0.406)).reshape(1,3,1,1); istd = torch.tensor((0.229,0.224,0.225)).reshape(1,3,1,1)
Z, Y = [], []
with torch.no_grad():
    for b in range(0, len(held), 32):
        px, st = [], []
        for i in held[b:b+32]:
            e = store.episode(int(i))
            t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(e.pixels, -1, 1))).float().div_(255.)
            px.append((t - im) / istd); st.append(torch.from_numpy(e.state.astype(np.float32)))
        emb = model.encode({'pixels': torch.stack(px).to(dev)})['emb']
        Z.append(emb.reshape(-1, emb.shape[-1]).float().cpu()); Y.append(torch.stack(st).reshape(-1, 8)[:, :2])
Z, Y = torch.cat(Z).double(), torch.cat(Y).double()
md = store.metadata; ppu = md['pixels_shape'][0] / md['scene']['view_height']
out = {'frames': int(Z.shape[0]), 'position_rms': float(Y.pow(2).sum(-1).mean().sqrt())}
for name, f in [('raw', Z), ('ln', torch.nn.functional.layer_norm(Z.float(), (Z.shape[1],)).double())]:
    r = ridge_decodability(f, Y, lambdas=[1e-4,1e-2,1.,1e1,1e2,1e3,1e4])
    out[name] = {'rmse_world': r['ridge_rmse_world'], 'rmse_px': r['ridge_rmse_world']*ppu,
                 'var_explained': r['variance_explained']}
print(json.dumps(out, indent=2, sort_keys=True))
