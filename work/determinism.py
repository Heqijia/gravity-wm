"""Two checks that settle what the latent is.

1. Is the encoder deterministic?  Encode the same frame twice, and encode it
   inside two different batches.  A latent that depends on batch composition
   is not a function of the image at all, and nothing downstream can work.
2. What does the latent track, if not position?  Decode the frame index and
   the episode index from it.
"""
import sys
import numpy as np, torch
from sg_jepa.data import open_trajectory_store
from gwm.latents import load_world_model
sys.path.insert(0, '/users/PAS3151/qijia387/Phys-WM/gravity-wm/scripts')
from diagnose import ridge_decodability

ckpt, data = sys.argv[1], sys.argv[2]
dev = torch.device('cuda')
model, _m, _s, cfg = load_world_model(ckpt, device=dev); model.eval()
store = open_trajectory_store(data)
im = torch.tensor((0.485,0.456,0.406)).reshape(1,3,1,1); istd = torch.tensor((0.229,0.224,0.225)).reshape(1,3,1,1)

def px_of(i):
    e = store.episode(int(i))
    t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(e.pixels,-1,1))).float().div_(255.)
    return (t-im)/istd, e

idx = np.flatnonzero(store.split_id == 0)[:200]
p0, e0 = px_of(idx[0]); p1, e1 = px_of(idx[1])
with torch.no_grad():
    a = model.encode({'pixels': p0.unsqueeze(0).to(dev)})['emb'][0]
    b = model.encode({'pixels': p0.unsqueeze(0).to(dev)})['emb'][0]
    c = model.encode({'pixels': torch.stack([p0, p1]).to(dev)})['emb'][0]
    big = model.encode({'pixels': torch.stack([p0] + [px_of(i)[0] for i in idx[1:8]]).to(dev)})['emb'][0]
print("same frame, same batch twice   max |diff| = %.3e" % float((a-b).abs().max()))
print("same frame, batch of 2         max |diff| = %.3e" % float((a-c).abs().max()))
print("same frame, batch of 8         max |diff| = %.3e" % float((a-big).abs().max()))
print("latent scale                   max |z|    = %.3e" % float(a.abs().max()))

Z, S, FI, EI = [], [], [], []
with torch.no_grad():
    for k in range(0, len(idx), 24):
        ps, sts = [], []
        for i in idx[k:k+24]:
            p, e = px_of(i); ps.append(p); sts.append(torch.from_numpy(e.state.astype(np.float32)))
        emb = model.encode({'pixels': torch.stack(ps).to(dev)})['emb'].float().cpu()
        Z.append(emb); S.append(torch.stack(sts))
        T = emb.shape[1]
        FI.append(np.tile(np.arange(T), emb.shape[0]))
        EI.append(np.repeat(np.arange(k, k+emb.shape[0]), T))
Z = torch.cat(Z); S = torch.cat(S); T = Z.shape[1]
Zf = Z.reshape(-1, Z.shape[-1]).double()
eids = np.repeat(np.arange(Z.shape[0]), T)
lam = [1e-4,1e-2,1.,1e1,1e2,1e3,1e4]
targets = {
    'frame index': torch.from_numpy(np.concatenate(FI)).double().reshape(-1,1),
    'position (x,z)': S.reshape(-1,8)[:, :2].double(),
    'velocity (vx,vz)': S.reshape(-1,8)[:, 2:4].double(),
}
print()
for nm, y in targets.items():
    r = ridge_decodability(Zf, y, lambdas=lam, episode_ids=eids)
    print(f"  decode {nm:<18} var_expl {r['variance_explained']:+.4f}  (RMSE {r['ridge_rmse_world']:.4f}, baseline {r['predict_mean_rmse_world']:.4f})")
