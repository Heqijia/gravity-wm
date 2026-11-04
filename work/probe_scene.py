"""Verify the chosen scene: in-frame across the whole g grid, exact physics, SNR."""
import numpy as np
from gwm.freefall_data import (SceneSpec, render_episode, sample_episode_parameters,
                               shared_rise_range, verify_free_fall_against_integrator)

IMAGE = 128
for dt, frames, view in [(0.16, 16, 13.0), (0.16, 16, 12.0), (0.18, 14, 12.0), (0.14, 18, 13.0)]:
    spec = SceneSpec(dt=dt, frames=frames, view_height=view, z_center=0.5*view)
    grid = sorted(set(list(np.arange(1.0, 9.01, 0.5)) + [4.0]))
    try:
        rr = shared_rise_range(spec, grid)
    except ValueError as exc:
        print(f"dt={dt} T={frames} V={view}: INFEASIBLE -> {str(exc)[:90]}"); continue
    px1 = spec.gravity_step_pixels(4.0, IMAGE)
    resid, cons, gap = [], 0, []
    for g in grid:
        p = sample_episode_parameters(g, np.random.default_rng([7, int(g*10)]), spec, rise_range=rr)
        _, _, _, d = render_episode(p, spec, image_size=IMAGE)
        resid.append(d["free_fall_residual_max"])
        chk = verify_free_fall_against_integrator(p, spec); cons += int(chk["contacts"]); gap.append(chk["max_second_difference_gap"])
    print(f"dt={dt} T={frames} V={view}: rise={rr[0]:.3f}..{rr[1]:.3f}  "
          f"px/step m=1:{px1:.2f} m=2:{px1*4:.2f} m=3:{px1*9:.2f}  "
          f"analytic_resid={max(resid):.1e}  mjstep_gap={max(gap):.1e}  ncon={cons}")
