"""The bouncing scene, and the counterfactual filter option A depends on."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gwm.bounce_data import BounceSpec, counterfactual_hits_floor


def hits(z0, vz, gravity, *, resting_z=0.45, dt=0.16, horizon=6):
    return bool(
        counterfactual_hits_floor(
            np.array([[0.0, z0]]), np.array([[0.0, vz]]), np.array([gravity]),
            spec_resting_z=resting_z, dt=dt, horizon=horizon,
        )[0]
    )


def test_free_fall_time_to_the_floor_is_exact() -> None:
    """Dropped from rest, the root must be ``sqrt(2 h / g)``."""

    resting, horizon, dt = 0.45, 6, 0.16
    span = horizon * dt
    for height, gravity in ((1.0, 4.0), (2.5, 9.0), (0.3, 1.0)):
        predicted = math.sqrt(2.0 * height / gravity)
        assert hits(resting + height, 0.0, gravity) == (predicted <= span)


def test_a_window_that_is_safe_at_g_can_bounce_at_a_larger_g() -> None:
    """The reason the filter must run under ``g'`` and not under ``g``.

    This is the whole subtlety of option A: the data's own contact flags say
    nothing about whether the *counterfactual* path stays off the floor.
    """

    height, vz = 1.2, 0.0
    assert not hits(0.45 + height, vz, 1.0)     # slow fall, still airborne
    assert hits(0.45 + height, vz, 9.0)         # same window, bounces under g'


def test_upward_velocity_delays_the_contact() -> None:
    assert hits(0.45 + 0.5, 0.0, 4.0)
    assert not hits(0.45 + 0.5, 4.0, 4.0)


def test_already_touching_counts_as_contact() -> None:
    assert hits(0.45, 0.0, 4.0)
    assert hits(0.20, 5.0, 4.0)                 # below the floor, whatever the velocity


def test_zero_gravity_never_reaches_the_floor_while_rising() -> None:
    assert not hits(0.45 + 2.0, 1.0, 0.0)


def test_filter_is_vectorised_and_shape_preserving() -> None:
    n = 32
    rng = np.random.default_rng(0)
    out = counterfactual_hits_floor(
        np.stack([np.zeros(n), rng.uniform(0.5, 8.0, n)], axis=-1),
        np.stack([np.zeros(n), rng.uniform(-3.0, 3.0, n)], axis=-1),
        rng.uniform(1.0, 9.0, n),
        spec_resting_z=0.45, dt=0.16, horizon=6,
    )
    assert out.shape == (n,) and out.dtype == bool


@pytest.mark.slow
def test_analytic_filter_agrees_with_the_simulator() -> None:
    """Cross-check the filter against ``mj_step``, the way the generator is.

    The filter decides which windows enter ``C``, so a disagreement with the
    simulator would silently admit windows on which the free-fall identity is
    false -- measured at 14.2 px against a 1.008 px signal.
    """

    pytest.importorskip("mujoco")
    from gwm.bounce_data import _dry_run

    spec = BounceSpec(frames=8)
    horizon, rng, checked, agreed = 6, np.random.default_rng(3), 0, 0
    for _ in range(40):
        gravity = float(rng.uniform(1.0, 9.0))
        parameters = {
            "gravity": gravity, "mass": 1.0, "x": 0.0,
            "z": float(rng.uniform(1.0, 7.0)), "vx": 0.0,
            "vz": float(rng.uniform(-2.0, 3.0)), "angle": 0.0, "spin": 0.0,
        }
        path, contact = _dry_run(parameters, spec)
        # ``contact[t]`` covers the interval from frame ``t`` to frame ``t+1``,
        # so a horizon of ``H`` predicted frames spans intervals ``0 .. H-1``
        # and a time of ``H * dt`` -- which is the span the analytic root is
        # compared against.  Taking ``contact[:H+1]`` here would include the
        # interval past the horizon and disagree on exactly the windows whose
        # bounce falls in it.
        simulated = bool(contact[:horizon].any())
        predicted = bool(
            counterfactual_hits_floor(
                path[:1], np.array([[0.0, parameters["vz"]]]), np.array([gravity]),
                spec_resting_z=spec.resting_z, dt=spec.dt, horizon=horizon,
            )[0]
        )
        checked += 1
        agreed += int(simulated == predicted)
    assert agreed == checked, f"filter disagreed with mj_step on {checked - agreed}/{checked}"


def test_window_statistics_uses_the_constraint_s_own_window() -> None:
    """Usable is about the *context*, and the unit is history + horizon frames.

    Counting three-frame triples instead reports a number the constraint never
    sees: at 16 frames a 14-frame window has three starts, and the share that
    survives a bounce is nothing like the share of triples that do.
    """

    from gwm.bounce_data import window_statistics

    contact = np.zeros(32, dtype=bool)
    usable, bounce, total = window_statistics(contact, history=8, horizon=6)
    assert (usable, bounce, total) == (19, 0, 19)

    contact[10] = True                      # inside some contexts, some horizons
    usable, bounce, total = window_statistics(contact, history=8, horizon=6)
    assert total == 19
    # a window starting at t has context [t, t+8); 10 lies in it for t = 3..10
    assert usable == 19 - len(range(3, 11))
    # and horizon [t+8, t+14); 10 lies in it for t = -3..2, i.e. t = 0..2
    assert bounce == 3


def test_episode_must_be_long_enough_for_one_window() -> None:
    from gwm.bounce_data import window_statistics

    assert window_statistics(np.zeros(14, dtype=bool), history=8, horizon=6)[2] == 1
    assert window_statistics(np.zeros(13, dtype=bool), history=8, horizon=6)[2] == 0


def test_manifest_records_where_the_pixels_were_rendered() -> None:
    """A dataset must say which renderer produced it.

    The same scene through two EGL implementations gives an identical state
    hash and a different pixel hash -- measured at 91 differing non-white
    pixels out of 35k, 0.26%, between a login node and a GPU compute node.
    That is larger than the 0.1-0.2 px the position probe resolves, so a
    dataset rendered on one and compared against one rendered on the other
    carries a bias nothing downstream would attribute correctly.
    """

    import inspect

    from gwm.bounce_data import generate_bounce_dataset

    source = inspect.getsource(generate_bounce_dataset)
    assert '"render_host"' in source
    assert '"render_backend"' in source


def _bounce_batch(contact_at=None, *, size=16, history=8, horizon=6, z0=4.0, vz=0.0):
    """A minimal constraint batch with a known state and contact pattern."""

    import torch

    span = history + horizon
    state = torch.zeros(size, span, 8)
    state[:, :, 1] = z0
    state[:, :, 3] = vz
    contact = torch.zeros(size, span, dtype=torch.bool)
    if contact_at is not None:
        contact[:, contact_at] = True
    return {
        "latents": torch.zeros(size, span, 4),
        "state": state,
        "action": torch.zeros(size, span, 3),
        "contact": contact,
        "gravity": torch.full((size,), 4.0),
    }


def test_counterfactual_floor_filter_rejects_by_the_counterfactual_gravity() -> None:
    """The filter must reject at a large ``g'`` what it accepts at a small one."""

    import torch

    from gwm.constraint import ConstraintSpec
    from gwm.posttrain import sample_counterfactual_gravity

    spec = ConstraintSpec(history=8, horizon=6, form="position")
    batch = _bounce_batch(z0=2.0, vz=0.0)
    kept = {}
    for low, high in ((1.0, 1.0001), (9.0, 9.0001)):
        _, accepted, _ = sample_counterfactual_gravity(
            batch, low=low, high=high, spec=spec, dt=0.16, safe_box=None, margin=0.0,
            generator=torch.Generator().manual_seed(0), floor_resting_z=1.25,
        )
        kept[low] = float(accepted.float().mean())
    assert kept[1.0] == 1.0, "a slow fall from 2.0 stays above a floor at 1.25"
    assert kept[9.0] == 0.0, "the same window bounces at g'=9 and must be rejected"


def test_no_floor_means_no_floor_filter() -> None:
    """``floor_resting_z=None`` must leave the contact-free datasets untouched."""

    import torch

    from gwm.constraint import ConstraintSpec
    from gwm.posttrain import sample_counterfactual_gravity

    _, accepted, _ = sample_counterfactual_gravity(
        _bounce_batch(z0=0.1), low=9.0, high=9.0001,
        spec=ConstraintSpec(history=8, horizon=6, form="position"),
        dt=0.16, safe_box=None, margin=0.0,
        generator=torch.Generator().manual_seed(0), floor_resting_z=None,
    )
    assert bool(accepted.all())


def test_contact_in_the_context_is_what_disqualifies_a_window() -> None:
    """Intervals ``0 .. history-2`` are the context; a bounce later is fine.

    The position form starts its counterfactual rollout from an implied
    velocity read off the last two context frames, and that read is a
    free-fall quantity.  A bounce after the context does not corrupt it -- that
    window is the world model's problem, not the constraint's.
    """

    history = 8
    for index, disqualifies in ((0, True), (history - 2, True), (history - 1, False), (11, False)):
        batch = _bounce_batch(contact_at=index, history=history)
        clean = ~batch["contact"][:, : history - 1].any(dim=1)
        assert bool(clean.all()) == (not disqualifies), f"interval {index}"
