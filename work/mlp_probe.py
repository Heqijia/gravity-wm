"""Measure the real (nonlinear) position probe on a checkpoint.

Ridge regression only recovers the linear part of the code, so it is a lower
bound on what the MLP probe of ``gwm.probe`` achieves, not an upper bound.  For
a scene whose image is a pure function of position, the latent is necessarily a
function of position; the question is only how nonlinearly it is encoded.
"""
import json, sys
import numpy as np, torch
from sg_jepa.data import open_trajectory_store
from gwm.constraint import ConstraintSpec, gravity_blind_violation, gravity_vector
from gwm.latents import LatentCache, load_world_model
from gwm.posttrain import calibrate_tolerance
from gwm.probe import ProbeConfig, probe_report, train_position_probes
sys.path.insert(0, '/users/PAS3151/qijia387/Phys-WM/gravity-wm/scripts')
from diagnose import ridge_decodability

ckpt, data = sys.argv[1], sys.argv[2]
n_fit = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 500
dev = torch.device('cuda')
model, _c, stats, _r = load_world_model(ckpt, device=dev)
model.eval()
store = open_trajectory_store(data)
md = store.metadata
ppu = md['pixels_shape'][0] / md['scene']['view_height']
dt = float(md['dt'])
im = torch.tensor((0.485,0.456,0.406)).reshape(1,3,1,1); istd = torch.tensor((0.229,0.224,0.225)).reshape(1,3,1,1)
amean = torch.tensor(stats.mean); astd = torch.tensor(stats.std)

def encode(indices):
    Z, S, A, G = [], [], [], []
    with torch.no_grad():
        for b in range(0, len(indices), 24):
            px, st, ac, gv = [], [], [], []
            for i in indices[b:b+24]:
                e = store.episode(int(i))
                t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(e.pixels,-1,1))).float().div_(255.)
                px.append((t-im)/istd); st.append(torch.from_numpy(e.state.astype(np.float32)))
                ac.append((torch.from_numpy(e.action.astype(np.float32))-amean)/astd)
                gv.append(float(e.gravity[0]))
            emb = model.encode({'pixels': torch.stack(px).to(dev)})['emb']
            Z.append(emb.float().cpu()); S.append(torch.stack(st)); A.append(torch.stack(ac)); G += gv
    return torch.cat(Z), torch.cat(S), torch.cat(A), torch.tensor(G)

fit_idx = np.flatnonzero(store.split_id == 0)[:n_fit]
test_idx = np.flatnonzero(store.split_id == 1)[:n_test]
Zf, Sf, _Af, _Gf = encode(fit_idx)
Zt, St, At, Gt = encode(test_idx)
frames = Zf.shape[1]
print(f"fit {Zf.shape[0]} episodes x {frames} frames, test {Zt.shape[0]} episodes")

feat = Zf.reshape(-1, Zf.shape[-1]).to(dev)
pos = Sf.reshape(-1, 8)[:, :2].to(dev)
eids = np.repeat(np.arange(Zf.shape[0]), frames)
scale = float((pos - pos.mean(0)).pow(2).sum(-1).mean().sqrt())
print(f"centred position scale {scale:.3f} world = {scale*ppu:.1f} px")

lam = [1e-4, 1e-2, 1., 1e1, 1e2, 1e3, 1e4]
r = ridge_decodability(feat.double().cpu(), pos.double().cpu(), lambdas=lam, episode_ids=eids)
print(f"  linear ridge (ep split) {r['ridge_rmse_world']:.4f} world = {r['ridge_rmse_world']*ppu:6.2f} px   baseline {r['predict_mean_rmse_world']*ppu:.2f} px")

results = {}
for input_norm in (False, True):
    cfg = ProbeConfig(input_norm=input_norm, max_epochs=300, patience=30, batch_size=2048, seed=7)
    pt, pe, _summ = train_position_probes(feat, pos, eids, config=cfg, device=dev)
    cache = LatentCache(latents=Zt.to(dev), state=St.to(dev), action=At.to(dev), gravity=Gt.to(dev),
                        split=torch.ones(Zt.shape[0], dtype=torch.long, device=dev),
                        source_id=torch.arange(Zt.shape[0], device=dev), dt=dt, pixels_per_unit=ppu,
                        action_mean=amean.to(dev), action_std=astd.to(dev), metadata={})
    with torch.no_grad():
        rep = probe_report(pt(Zt.reshape(-1, Zt.shape[-1]).to(dev)),
                           St.reshape(-1, 8)[:, :2].to(dev),
                           dt_effective=dt*3, gravity=4.0, pixels_per_unit=ppu)
    rows = []
    for stride in (1, 2, 3, 4):
        if frames < 8 + 6 or 8 < 2*stride: continue
        spec = ConstraintSpec(horizon=6, stride=stride, history=8)
        cal = calibrate_tolerance(pt, cache, spec=spec, windows=4096, seed=0)
        eps = 1.25 * cal['c_real_mean']
        g = torch.full((1,), 4.0, device=dev)
        blind = float(gravity_blind_violation(spec, gravity_vector(g), gravity_vector(g+5.0)))
        rows.append((stride, cal['c_real_mean'], eps, blind, blind/eps if eps > 0 else float('inf')))
    results[input_norm] = (rep.to_dict(), rows)
    tag = 'LayerNorm' if input_norm else 'no-LayerNorm'
    print(f"  MLP probe ({tag:<12}) {rep.rmse_world:.4f} world = {rep.rmse_pixels:6.2f} px")
    print(f"      {'m':>2} {'c_real':>11} {'eps':>11} {'c_blind':>9} {'margin':>8}")
    for stride, creal, eps, blind, margin in rows:
        print(f"      {stride:>2} {creal:>11.4e} {eps:>11.4e} {blind:>9.2f} {margin:>8.2f}")
json.dump({str(k): v for k, v in results.items()},
          open('/users/PAS3151/qijia387/Phys-WM/work/mlp_probe.json','w'), indent=2, default=str)
