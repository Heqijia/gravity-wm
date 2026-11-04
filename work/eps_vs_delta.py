"""How large is E[c_real] for a probe of accuracy delta?

Lemma 2 bounds E[c_real] by 16 delta^2 / dt_eff^4, which treats the probe's
per-frame errors as unrelated.  A real probe is a deterministic function of the
latent, hence of the object's state, so along a smooth trajectory its error is a
smooth function of time -- a systematic bias, not independent noise -- and a
second difference of a smooth function is small.  This script measures the gap
by giving the true positions to synthetic probes with controlled accuracy and
controlled error-field smoothness, so that the tolerance to expect for a given
probe accuracy can be read off before any encoder is trained.

The two extremes bracket any real probe:
  'iid'    independent per-frame error -- Remark 12's regime, constant 6
  'smooth' error is a smooth random function of (x, z, theta) with a given
           correlation length, which is what a trained network produces
"""
from __future__ import annotations
import json, math, sys
import numpy as np, torch
from sg_jepa.data import open_trajectory_store
from gwm.constraint import ConstraintSpec, free_fall_violation, gravity_vector, gravity_blind_violation

data = sys.argv[1] if len(sys.argv) > 1 else '/fs/scratch/PAS3151/qijia387/gwm/data/freefall_v2'
store = open_trajectory_store(data)
md = store.metadata
dt = float(md['dt']); ppu = md['pixels_shape'][0] / md['scene']['view_height']
idx = np.flatnonzero(store.split_id == 1)[:400]
states = np.stack([store.episode(int(i)).state for i in idx])      # (E, T, 8)
gravs = np.array([store.episode(int(i)).gravity[0] for i in idx])
S = torch.from_numpy(states).double(); G = torch.from_numpy(gravs).double()
print(f"episodes={S.shape[0]} frames={S.shape[1]} dt={dt} px/unit={ppu:.3f}")

def smooth_field(coords, n_modes, length_scale, seed):
    """A random band-limited function of (x, z, theta), RMS-normalised to 1."""
    gen = torch.Generator().manual_seed(seed)
    freq = torch.randn(n_modes, coords.shape[-1], generator=gen, dtype=torch.float64) / length_scale
    phase = torch.rand(n_modes, 2, generator=gen, dtype=torch.float64) * 2 * math.pi
    amp = torch.randn(n_modes, 2, generator=gen, dtype=torch.float64)
    proj = coords @ freq.T
    out = torch.einsum('...m,mk->...k', torch.sin(proj.unsqueeze(-1) + phase[:, 0]).squeeze(-1), amp) \
        if False else (torch.sin(proj).unsqueeze(-1) * amp).sum(-2)
    return out / out.pow(2).sum(-1).mean().sqrt()

coords = torch.stack([S[..., 0], S[..., 1], S[..., 4]], dim=-1)
rows = []
for stride in (1, 2, 3, 4):
    spec = ConstraintSpec(horizon=6, stride=stride, history=8)
    total = spec.history + spec.horizon
    gvec = gravity_vector(G.float()).double()
    blind5 = float(gravity_blind_violation(spec, gvec, gravity_vector((G + 5.0).float()).double()).mean())
    truth = S[:, :total, :2]
    base, _a, _t = free_fall_violation(truth.float(), gravity=gvec.float(),
                                       counterfactual=gvec.float(), spec=spec, dt=dt)
    for delta_px in (0.25, 0.5, 1.0, 2.0, 4.0):
        delta = delta_px / ppu
        entry = {'stride': stride, 'delta_px': delta_px, 'delta_world': delta,
                 'lemma2_bound': 16 * delta**2 / (stride * dt) ** 4,
                 'exact_c_real': float(base.mean()), 'c_blind_shift5': blind5}
        # independent per-frame error
        gen = torch.Generator().manual_seed(11)
        noise = torch.randn(*truth.shape, generator=gen, dtype=torch.float64)
        noise = noise / noise.pow(2).sum(-1).mean().sqrt() * delta
        c_iid, _a, _t = free_fall_violation((truth + noise).float(), gravity=gvec.float(),
                                            counterfactual=gvec.float(), spec=spec, dt=dt)
        entry['c_real_iid'] = float(c_iid.mean())
        # smooth error field, several correlation lengths (world units)
        for length in (0.3, 1.0, 3.0):
            field = smooth_field(coords[:, :total], 64, length, seed=17) * delta
            c_sm, _a, _t = free_fall_violation((truth + field).float(), gravity=gvec.float(),
                                               counterfactual=gvec.float(), spec=spec, dt=dt)
            entry[f'c_real_smooth_L{length}'] = float(c_sm.mean())
        rows.append(entry)

print(f"\n{'m':>2} {'delta(px)':>9} {'lemma2':>10} {'c_iid':>10} {'L=0.3':>10} {'L=1':>10} {'L=3':>10} {'c_blind':>9} {'margin(L=1)':>12}")
for r in rows:
    eps = 1.25 * r['c_real_smooth_L1.0']
    print(f"{r['stride']:>2} {r['delta_px']:>9.2f} {r['lemma2_bound']:>10.2e} {r['c_real_iid']:>10.3e} "
          f"{r['c_real_smooth_L0.3']:>10.3e} {r['c_real_smooth_L1.0']:>10.3e} {r['c_real_smooth_L3.0']:>10.3e} "
          f"{r['c_blind_shift5']:>9.2f} {r['c_blind_shift5']/eps:>12.1f}")
print("\nexact positions (delta=0) give c_real =", rows[0]['exact_c_real'])
json.dump(rows, open('/users/PAS3151/qijia387/Phys-WM/work/eps_vs_delta.json','w'), indent=2)
