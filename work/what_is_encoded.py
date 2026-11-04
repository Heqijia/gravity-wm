"""What does the latent actually encode? Decode each state coordinate separately."""
import sys
import numpy as np, torch
from sg_jepa.data import open_trajectory_store
from gwm.latents import load_world_model
sys.path.insert(0, '/users/PAS3151/qijia387/Phys-WM/gravity-wm/scripts')
from diagnose import ridge_decodability

ckpt, data, n_ep = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = torch.device('cuda')
model, _c, _s, _r = load_world_model(ckpt, device=dev)
model.eval(); model.store_cls_raw = True
store = open_trajectory_store(data)
held = np.flatnonzero(store.split_id == 1)[:n_ep]
im = torch.tensor((0.485,0.456,0.406)).reshape(1,3,1,1); istd = torch.tensor((0.229,0.224,0.225)).reshape(1,3,1,1)
Z, S = [], []
with torch.no_grad():
    for b in range(0, len(held), 16):
        px, st = [], []
        for i in held[b:b+16]:
            e = store.episode(int(i))
            t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(e.pixels,-1,1))).float().div_(255.)
            px.append((t-im)/istd); st.append(torch.from_numpy(e.state.astype(np.float32)))
        emb = model.encode({'pixels': torch.stack(px).to(dev)})['emb']
        Z.append(emb.float().cpu()); S.append(torch.stack(st))
frames = Z[0].shape[1]
Z = torch.cat(Z); Z = Z.reshape(-1, Z.shape[-1]).double()
S = torch.cat(S); S = S.reshape(-1, 8).double()
EPI = np.repeat(np.arange(S.shape[0] // frames), frames)
md = store.metadata; ppu = md['pixels_shape'][0]/md['scene']['view_height']
lam = [1e-4,1e-2,1.,1e1,1e2,1e3,1e4]
print(f"{'target':<10} {'scale':>9} {'ridge RMSE':>11} {'var expl':>9}  (linear probe on emb)")
names = ['x','z','vx','vz']
for j, nm in enumerate(names):
    y = S[:, j:j+1]
    r = ridge_decodability(Z, y, lambdas=lam, episode_ids=EPI)
    print(f"{nm:<10} {float((y-y.mean()).std()):>9.3f} {r['ridge_rmse_world']:>11.4f} {r['variance_explained']:>+9.4f}")
# joint position, and position in pixel terms
r = ridge_decodability(Z, S[:, :2], lambdas=lam, episode_ids=EPI)
print(f"{'(x,z)':<10} {'':>9} {r['ridge_rmse_world']:>11.4f} {r['variance_explained']:>+9.4f}  = {r['ridge_rmse_world']*ppu:.1f} px")
r = ridge_decodability(Z, S[:, 2:4], lambdas=lam, episode_ids=EPI)
print(f"{'(vx,vz)':<10} {'':>9} {r['ridge_rmse_world']:>11.4f} {r['variance_explained']:>+9.4f}")
# how much of emb's variance is explained BY the state (the reverse direction)
r = ridge_decodability(S[:, :4], Z, lambdas=lam, episode_ids=EPI)
print(f"\nreverse: state -> emb   var of emb explained by the true state: {r['variance_explained']:+.4f}")
