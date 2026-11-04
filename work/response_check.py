"""How much does the predictor actually move when the conditioning gravity changes?

The constraint can only help if the predictor's response to g' is wrong.  The
exact response is known: switching from g to g' displaces the object by
(g - g')(s dt)^2 / 2 downward at step s, independent of everything else.  This
measures the model's realised response against that, in probe coordinates and
against the true positions, so "gravity-blind" and "responds but inaccurately"
can be told apart.
"""
import sys
import numpy as np, torch
from gwm.constraint import ConstraintSpec, condition_on_gravity, decode_positions, rollout_latents
from gwm.evaluate import switched_trajectory
from gwm.latents import load_latent_cache, load_world_model, set_posttraining_modes
from gwm.probe import ProbeConfig, load_position_probe

ckpt, cache_p, probe_d = sys.argv[1], sys.argv[2], sys.argv[3]
dev = torch.device('cuda')
model, _m, _s, _c = load_world_model(ckpt, device=dev)
set_posttraining_modes(model, ('predictor',)); model.eval(); model.requires_grad_(False)
cache = load_latent_cache(cache_p).to(dev)
held = cache.select(cache.split == 1)
rng = cache.metadata.get('heldout_train_gravity_source_range')
ctx = held.select((held.source_id >= rng[0]) & (held.source_id <= rng[1])) if rng else held
probe = load_position_probe(probe_d + '/probe_eval.pt', feature_dim=cache.latent_dim,
                            config=ProbeConfig(), device=dev)
spec = ConstraintSpec(horizon=6, stride=1, history=8)
L = spec.history + spec.horizon
ei, st = ctx.windows(L)
sel = torch.randperm(ei.shape[0])[:3072].to(ei.device)
b = ctx.gather_windows(ei[sel], st[sel], L)
ppu = cache.pixels_per_unit; pivot = spec.history - 1

def rollout(gp):
    g = torch.full((b['latents'].shape[0],), float(gp), device=dev)
    a = condition_on_gravity(b['action'], g, action_mean=cache.action_mean, action_std=cache.action_std)
    pred = rollout_latents(model, b['latents'], model.action_encoder(a),
                           history=spec.history, horizon=spec.horizon)
    seq = torch.cat([b['latents'][:, :spec.history], pred], dim=1)
    return decode_positions(probe, seq)[:, spec.history:], g

print('response at the last rollout step (s=6, dt=0.16), in pixels')
print('%8s %12s %12s %12s %10s' % ("g'", 'model dz', 'exact dz', 'ratio', 'abs err'))
ref, _ = rollout(4.0)
truth_ref = switched_trajectory(b['state'][:, pivot, :2], b['state'][:, pivot, 2:4],
                                torch.full((b['latents'].shape[0],), 4.0, device=dev),
                                horizon=spec.horizon, dt=cache.dt)
for gp in (1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0):
    out, g = rollout(gp)
    truth = switched_trajectory(b['state'][:, pivot, :2], b['state'][:, pivot, 2:4], g,
                                horizon=spec.horizon, dt=cache.dt)
    model_dz = float((out[:, -1, 1] - ref[:, -1, 1]).mean()) * ppu
    exact_dz = float((truth[:, -1, 1] - truth_ref[:, -1, 1]).mean()) * ppu
    err = float((out[:, -1] - truth[:, -1]).pow(2).sum(-1).mean().sqrt()) * ppu
    print('%8.1f %12.3f %12.3f %12.3f %10.3f' % (gp, model_dz, exact_dz, model_dz/exact_dz if exact_dz else float('nan'), err))
print()
print('ratio = 1.0 would be a perfect gravity response; 0.0 would be gravity-blind')
