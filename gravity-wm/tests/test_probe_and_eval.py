"""Probe bookkeeping, and the evaluation targets that define the comparison."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from gwm.constraint import ConstraintSpec, free_fall_violation, gravity_vector
from gwm.evaluate import corollary_two_bound, switched_trajectory
from gwm.freefall_data import SceneSpec, trajectory
from gwm.latents import LatentCache, set_posttraining_modes
from gwm.probe import ProbeConfig, probe_report, train_position_probes


def test_probe_report_ratios():
    predicted = torch.tensor([[0.0, 0.0], [0.3, 0.4]])
    target = torch.zeros(2, 2)
    report = probe_report(
        predicted, target, dt_effective=0.32, gravity=4.0, pixels_per_unit=10.0
    )
    assert math.isclose(report.rmse_world, math.sqrt(0.25 / 2), rel_tol=1e-6)
    assert math.isclose(report.gravity_step_world, 4.0 * 0.32**2, rel_tol=1e-9)
    assert math.isclose(report.rmse_pixels, report.rmse_world * 10.0, rel_tol=1e-9)
    assert math.isclose(
        report.signal_to_noise, report.gravity_step_world / report.rmse_world, rel_tol=1e-9
    )
    assert math.isclose(
        report.lemma2_bound_on_c_real, 16.0 * report.rmse_world**2 / 0.32**4, rel_tol=1e-9
    )


def test_probes_are_trained_on_disjoint_episodes():
    torch.manual_seed(0)
    episodes, frames, dim = 40, 6, 24
    direction = torch.randn(dim, 2)
    features = torch.randn(episodes * frames, dim)
    positions = features @ direction
    episode_ids = np.repeat(np.arange(episodes), frames)
    config = ProbeConfig(max_epochs=12, patience=12, batch_size=64, seed=7)
    train, evaluation, summary = train_position_probes(
        features, positions, episode_ids, config=config, device=torch.device("cpu")
    )
    assert summary["episode_overlap_between_probes"] == 0
    assert summary["probes"][0]["role"] == "train"
    assert summary["probes"][1]["role"] == "eval"
    # Different data and different initialisation, so the two probes differ.
    with torch.no_grad():
        gap = float((train(features[:64]) - evaluation(features[:64])).abs().mean())
    assert gap > 0.0
    assert not any(p.requires_grad for p in train.parameters())


def test_probe_recovers_a_linear_position_code():
    """A probe on a latent that linearly encodes position must become accurate.

    This is the sanity floor for the probe stage: if it cannot fit a code that
    is linear, a later failure is the probe's, not the predictor's.  Note that
    :func:`train_position_probes` splits the labelled episodes in half, so each
    probe is fitted on half of what is passed in; the episode count here is
    chosen so that each half is a realistic amount of data rather than a
    few hundred frames.
    """

    torch.manual_seed(1)
    episodes, frames, dim = 500, 16, 32
    direction = torch.randn(dim, 2) * 0.5
    # The code must be linear in the probe's *input space*, which begins with a
    # LayerNorm, so the synthetic features are layer-normalised before the
    # position is defined from them.  Defining the position from unnormalised
    # features would make it a function of each sample's scale, which the
    # LayerNorm discards -- the very effect ``input_norm`` exists to ablate.
    features = torch.nn.functional.layer_norm(torch.randn(episodes * frames, dim), (dim,))
    positions = features @ direction
    episode_ids = np.repeat(np.arange(episodes), frames)
    config = ProbeConfig(max_epochs=250, patience=40, batch_size=256, seed=3)
    train, evaluation, _summary = train_position_probes(
        features, positions, episode_ids, config=config, device=torch.device("cpu")
    )
    baseline = float(positions.pow(2).sum(-1).mean().sqrt())
    with torch.no_grad():
        errors = [
            float((probe(features) - positions).pow(2).sum(-1).mean().sqrt())
            for probe in (train, evaluation)
        ]
    for error in errors:
        assert error < 0.1 * baseline, f"probe error {error:.4f} vs scale {baseline:.4f}"


def test_input_layernorm_can_be_disabled():
    """Without the leading LayerNorm the probe can also use the latent's scale."""

    torch.manual_seed(2)
    episodes, frames, dim = 400, 16, 32
    direction = torch.randn(dim, 2) * 0.5
    features = torch.randn(episodes * frames, dim) * torch.rand(episodes * frames, 1).add(0.5)
    positions = features @ direction
    episode_ids = np.repeat(np.arange(episodes), frames)
    baseline = float(positions.pow(2).sum(-1).mean().sqrt())
    results = {}
    for input_norm in (True, False):
        config = ProbeConfig(
            input_norm=input_norm, max_epochs=150, patience=30, batch_size=256, seed=5
        )
        probe, _other, _summary = train_position_probes(
            features, positions, episode_ids, config=config, device=torch.device("cpu")
        )
        with torch.no_grad():
            results[input_norm] = float(
                (probe(features) - positions).pow(2).sum(-1).mean().sqrt()
            )
    assert results[False] < results[True]
    assert results[False] < 0.2 * baseline


def test_switched_trajectory_continues_the_generated_path():
    """With ``g' = g`` the evaluation target is the episode's own future."""

    spec = SceneSpec()
    path = trajectory(x0=0.4, z0=5.0, vx=0.9, vz=3.2, gravity=4.0, dt=spec.dt, frames=16)
    pivot = 7
    velocity = torch.tensor([[0.9, 3.2 - 4.0 * pivot * spec.dt]])
    predicted = switched_trajectory(
        torch.tensor(path[pivot : pivot + 1]).float(),
        velocity.float(),
        torch.tensor([4.0]),
        horizon=6,
        dt=spec.dt,
    )
    expected = torch.tensor(path[pivot + 1 : pivot + 7]).float().unsqueeze(0)
    torch.testing.assert_close(predicted, expected, atol=1e-4, rtol=1e-4)


def test_switched_trajectory_has_zero_violation():
    """The evaluation target and the constraint target agree, at any stride."""

    dt = 0.16
    for stride in (1, 2, 3):
        spec = ConstraintSpec(horizon=6, stride=stride, history=2 * stride + 2)
        g = torch.tensor([4.0])
        gp = torch.tensor([8.5])
        pivot = spec.history - 1
        # Context under g, future under g'.
        context_offsets = torch.arange(-pivot, 1, dtype=torch.float32) * dt
        position = torch.tensor([[1.0, 6.0]])
        velocity = torch.tensor([[0.5, 2.0]])
        accel_g = gravity_vector(g)
        context = (
            position.unsqueeze(1)
            + velocity.unsqueeze(1) * context_offsets.reshape(1, -1, 1)
            + 0.5 * accel_g.unsqueeze(1) * context_offsets.reshape(1, -1, 1) ** 2
        )
        future = switched_trajectory(position, velocity, gp, horizon=spec.horizon, dt=dt)
        violation, _a, _t = free_fall_violation(
            torch.cat([context, future], dim=1),
            gravity=accel_g,
            counterfactual=gravity_vector(gp),
            spec=spec,
            dt=dt,
        )
        assert float(violation) < 1e-6, f"stride {stride}"


def test_corollary_two_bound_shape():
    args = dict(delta=0.01, constraint=0.5, dt=0.16, horizon=6)
    values = [corollary_two_bound(step=step, **args) for step in range(1, 7)]
    assert all(b > a for a, b in zip(values, values[1:], strict=False))
    # With a perfect probe and an exactly satisfied constraint the bound is zero.
    assert corollary_two_bound(delta=0.0, constraint=0.0, dt=0.16, horizon=6, step=3) == 0.0
    # The probe term is the stated (2s+1) delta.
    assert math.isclose(
        corollary_two_bound(delta=0.01, constraint=0.0, dt=0.16, horizon=6, step=3),
        7 * 0.01,
        rel_tol=1e-9,
    )
    with pytest.raises(ValueError):
        corollary_two_bound(step=7, **args)


def _cache(episodes=5, frames=10, dim=4):
    torch.manual_seed(0)
    return LatentCache(
        latents=torch.randn(episodes, frames, dim),
        state=torch.randn(episodes, frames, 8),
        action=torch.randn(episodes, frames, 3),
        gravity=torch.linspace(1.0, 9.0, episodes),
        split=torch.zeros(episodes, dtype=torch.long),
        source_id=torch.arange(episodes),
        # Contact-free by construction.
        contact=torch.zeros(episodes, frames, dtype=torch.bool),
        dt=0.16,
        pixels_per_unit=9.85,
        action_mean=torch.zeros(3),
        action_std=torch.ones(3),
        metadata={},
    )


def test_cache_windows_are_contiguous_and_aligned():
    cache = _cache()
    episode_index, start = cache.windows(4)
    assert episode_index.shape[0] == 5 * (10 - 4 + 1)
    batch = cache.gather_windows(episode_index[:3], start[:3], 4)
    assert batch["latents"].shape == (3, 4, 4)
    for row in range(3):
        e, s = int(episode_index[row]), int(start[row])
        torch.testing.assert_close(batch["latents"][row], cache.latents[e, s : s + 4])
        torch.testing.assert_close(batch["state"][row], cache.state[e, s : s + 4])
        torch.testing.assert_close(batch["gravity"][row], cache.gravity[e])
    with pytest.raises(ValueError, match="exceeds episode length"):
        cache.windows(11)


def test_posttraining_modes_freeze_the_latent_pathway():
    from sg_jepa.config import WorldModelConfig
    from sg_jepa.models import build_world_model

    model = build_world_model(WorldModelConfig(image_size=64, patch_size=16, history_size=8))
    report = set_posttraining_modes(model, ("predictor", "pred_proj", "action_encoder"))
    assert report["frozen_modules_in_eval_mode"]
    assert report["normalisation_and_dropout_in_eval"] > 0
    # cuDNN refuses an RNN backward pass on a module in eval mode, so the
    # trainable predictor must stay in train mode.
    assert model.predictor.training
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert not any(p.requires_grad for p in model.projector.parameters())
    assert all(p.requires_grad for p in model.predictor.parameters())
    # Every BatchNorm must be in eval mode, so the same latent decodes the same
    # way in the JEPA branch and the constraint branch.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert not module.training
        if isinstance(module, torch.nn.Dropout):
            assert not module.training
    with pytest.raises(ValueError, match="must stay frozen"):
        set_posttraining_modes(model, ("encoder",))
    with pytest.raises(ValueError, match="must stay frozen"):
        set_posttraining_modes(model, ("projector",))


def test_batchnorm_running_stats_do_not_drift_in_posttraining():
    """A frozen latent pathway must stay byte-identical across forward passes."""

    from sg_jepa.config import WorldModelConfig
    from sg_jepa.models import build_world_model

    model = build_world_model(WorldModelConfig(image_size=64, patch_size=16, history_size=8))
    set_posttraining_modes(model, ("predictor", "pred_proj", "action_encoder"))
    latents = torch.randn(4, 8, 256)
    condition = torch.randn(4, 8, 256)
    first = model.predict(latents, condition)
    second = model.predict(latents * 10.0, condition)
    third = model.predict(latents, condition)
    torch.testing.assert_close(first, third)
    assert not torch.allclose(first, second)


def test_context_selection_by_source_range_excludes_the_wide_grid():
    """The counterfactual protocol's contexts must come from the training law.

    The generator sets aside held-out episodes drawn from the training gravity
    distribution and records their source range.  Selecting by that range is
    exact; selecting by ``|g - mean| <= k sigma`` would also admit the nearby
    points of the wide held-out grid, whose contexts are not
    training-distribution windows even when their gravity happens to be close.
    """

    episodes = 10
    cache = LatentCache(
        latents=torch.randn(episodes, 6, 4),
        state=torch.randn(episodes, 6, 8),
        action=torch.randn(episodes, 6, 3),
        # Two episodes of the wide grid sit at exactly the training mean, so a
        # gravity filter cannot tell them from the reserved contexts.
        gravity=torch.tensor([4.0] * 5 + [4.0, 4.0, 1.0, 9.0, 7.0]),
        split=torch.ones(episodes, dtype=torch.long),
        source_id=torch.arange(100, 100 + episodes),
        # Contact-free by construction.
        contact=torch.zeros(episodes, 6, dtype=torch.bool),
        dt=0.16,
        pixels_per_unit=9.85,
        action_mean=torch.zeros(3),
        action_std=torch.ones(3),
        metadata={"heldout_train_gravity_source_range": [100, 104]},
    )
    low, high = cache.metadata["heldout_train_gravity_source_range"]
    is_context = (cache.source_id >= low) & (cache.source_id <= high)
    contexts = cache.select(is_context)
    grid = cache.select(~is_context)
    assert contexts.episodes == 5
    assert grid.episodes == 5
    assert set(int(v) for v in contexts.source_id.tolist()) == set(range(100, 105))
    # The gravity filter would have pulled in the two grid episodes at g = 4.
    by_gravity = cache.select((cache.gravity - 4.0).abs() <= 2.0)
    assert by_gravity.episodes > contexts.episodes


def test_in_frame_filter_rejects_rather_than_resampling():
    """``P`` must be ``U(G)`` restricted to in-frame pairs, not reweighted.

    Resampling ``g'`` for a window that would leave the view replaces it with
    an easier value *for that window*, which changes the conditional law of
    ``g'`` given the window and under-draws the large shifts exactly where
    they are hardest.  The evaluation grid rejects, so training must reject
    too or the two measures differ and Corollary 2's density bound is lost.
    """

    from gwm.constraint import ConstraintSpec
    from gwm.posttrain import sample_counterfactual_gravity

    torch.manual_seed(0)
    spec = ConstraintSpec(horizon=6, stride=1, history=4)
    size = 512
    state = torch.zeros(size, spec.history + spec.horizon, 8)
    # Every window sits at rest at z = 4 in a box whose floor is z = 1.  Over
    # six steps of 0.16 s the drop is 0.46 g', so gravities above about 6.5
    # leave the view and smaller ones do not: a spread, not all-or-nothing.
    state[:, :, 1] = 4.0
    batch = {"latents": torch.zeros(size, spec.history + spec.horizon, 4), "state": state}
    box = (-5.0, 5.0, 1.0, 11.0)
    generator = torch.Generator().manual_seed(1)
    gravity, accepted, _ax = sample_counterfactual_gravity(
        batch, low=1.0, high=9.0, spec=spec, dt=0.16,
        safe_box=box, margin=0.0, generator=generator,
    )
    rate = float(accepted.float().mean())
    assert 0.1 < rate < 0.95, f"the fixture must reject some but not all draws; got {rate:.2f}"
    # The accepted draws must still look like U(1, 9) truncated, not like a
    # distribution that has had its tail refilled with easy values.
    rejected = gravity[~accepted]
    assert float(rejected.min()) > float(gravity[accepted].min())

    biased, biased_ok, _bax = sample_counterfactual_gravity(
        batch, low=1.0, high=9.0, spec=spec, dt=0.16,
        safe_box=box, margin=0.0,
        generator=torch.Generator().manual_seed(1), max_resamples=24,
    )
    # Resampling inflates the acceptance rate and lowers the mean shift, which
    # is the bias the default avoids.
    assert float(biased_ok.float().mean()) > float(accepted.float().mean())
    assert float(biased.mean()) < float(gravity.mean())


def test_tolerance_calibration_set_excludes_the_wide_grid():
    """``eps`` must come from the context distribution the constraint uses.

    The held-out split holds two different things: episodes drawn from the
    training gravity, which are the real windows of the counterfactual
    protocol, and a wide gravity grid whose *contexts* themselves evolve
    outside the training range.  Calibrating on both measures the predictor on
    inputs post-training never sees and inflates the tolerance -- measured
    here by a factor 1.7, enough to leave the pretrained model feasible and
    the constraint inactive from the first step.
    """

    episodes = 12
    gravity = torch.tensor([4.0] * 6 + [1.0, 2.0, 4.0, 6.0, 8.0, 9.0])
    cache = LatentCache(
        latents=torch.randn(episodes, 6, 4),
        state=torch.randn(episodes, 6, 8),
        action=torch.randn(episodes, 6, 3),
        gravity=gravity,
        split=torch.ones(episodes, dtype=torch.long),
        source_id=torch.arange(200, 200 + episodes),
        # Contact-free by construction.
        contact=torch.zeros(episodes, 6, dtype=torch.bool),
        dt=0.16,
        pixels_per_unit=9.85,
        action_mean=torch.zeros(3),
        action_std=torch.ones(3),
        metadata={"heldout_train_gravity_source_range": [200, 205]},
    )
    low, high = cache.metadata["heldout_train_gravity_source_range"]
    chosen = cache.select((cache.source_id >= low) & (cache.source_id <= high))
    assert chosen.episodes == 6
    assert torch.allclose(chosen.gravity, torch.full((6,), 4.0))
    # The grid episode that happens to sit at g = 4 must still be excluded:
    # its context is a different distribution even at the same gravity.
    by_gravity = cache.select((cache.gravity - 4.0).abs() < 1e-6)
    assert by_gravity.episodes == 7


def test_response_ratio_separates_blind_from_correct():
    """The response ratio must read 0 for a blind predictor and 1 for a right one.

    An aggregate position error cannot tell "does not respond to g'" from
    "responds but inaccurately"; measured on the pretrained model the response
    is near-correct for small shifts and saturates at the far end, which is
    the failure the constraint exists to repair.
    """

    from gwm.evaluate import switched_trajectory

    dt, horizon = 0.16, 6
    position = torch.tensor([[1.0, 6.0], [-2.0, 5.0]])
    velocity = torch.tensor([[0.5, 2.0], [0.0, 1.0]])
    gravity = torch.tensor([4.0, 4.0])
    counterfactual = torch.tensor([9.0, 9.0])
    own = switched_trajectory(position, velocity, gravity, horizon=horizon, dt=dt)
    truth = switched_trajectory(position, velocity, counterfactual, horizon=horizon, dt=dt)
    exact = truth[:, -1, 1] - own[:, -1, 1]

    # a predictor that ignores g' lands on its own-gravity trajectory
    blind = own[:, -1, 1] - own[:, -1, 1]
    assert float((blind / exact).mean()) == pytest.approx(0.0, abs=1e-9)
    # one that gets it right lands on the switched trajectory
    correct = truth[:, -1, 1] - own[:, -1, 1]
    assert float((correct / exact).mean()) == pytest.approx(1.0, rel=1e-6)
    # the exact displacement is the closed form (g - g')(s dt)^2 / 2
    expected = 0.5 * (gravity - counterfactual) * (horizon * dt) ** 2
    torch.testing.assert_close(exact, expected, rtol=1e-5, atol=1e-5)


def test_acceleration_diagnostics_do_not_depend_on_the_constraint_form() -> None:
    """The acceleration columns must mean the same thing in both forms.

    ``free_fall_violation`` returns ``(c, accelerations, targets)`` in the
    acceleration form of Definition 1 and ``(c, positions, targets)`` in the
    position form of Proposition 1.  Reading the residual off the configured
    form reported a position residual in the acceleration columns -- 0.278
    where the acceleration residual was 5.32 -- so the sweep reads it from the
    stride-one acceleration call it already makes for Corollary 2.
    """

    torch.manual_seed(0)
    spec_accel = ConstraintSpec(horizon=4, stride=1, history=4, form="acceleration")
    spec_position = ConstraintSpec(horizon=4, stride=1, history=4, form="position")
    positions = torch.randn(16, spec_accel.history + spec_accel.horizon, 2)
    gravity = torch.full((16, 2), 0.0)
    gravity[:, 1] = -4.0
    counterfactual = gravity.clone()
    counterfactual[:, 1] = -8.0

    residuals = []
    for spec in (spec_accel, spec_position):
        _c, predicted, target = free_fall_violation(
            positions, gravity=gravity, counterfactual=counterfactual, spec=spec, dt=0.16
        )
        residuals.append((predicted - target).pow(2).sum(-1).mean().item())
    # The two forms genuinely differ -- which is why the column has to come
    # from a fixed one rather than from whichever the run used.
    assert residuals[0] != pytest.approx(residuals[1], rel=1e-3)


def test_response_projection_equals_the_old_vertical_ratio() -> None:
    """The vector response must reproduce the scalar one, term for term.

    ``response_ratio`` used to be the ratio of *vertical* shifts, which is
    meaningless once the counterfactual has a horizontal component.  It is now
    the model displacement projected onto the displacement the switch causes.
    For a purely vertical switch the unit vector is -z and the two agree
    exactly, which is what keeps every previously reported number valid.
    """

    torch.manual_seed(0)
    batch, horizon, dt = 64, 6, 0.16
    position = torch.randn(batch, 2)
    velocity = torch.randn(batch, 2)
    gravity = torch.full((batch,), 4.0)
    counterfactual = torch.full((batch,), 8.0)       # g' > g, so the shift is -z
    own = switched_trajectory(position, velocity, gravity, horizon=horizon, dt=dt)
    truth = switched_trajectory(position, velocity, counterfactual, horizon=horizon, dt=dt)
    decoded = truth + 0.03 * torch.randn(batch, horizon, 2)

    old_exact = (truth[:, -1, 1] - own[:, -1, 1]).double()
    old_model = (decoded[:, -1, 1] - own[:, -1, 1]).double()
    old_ratio = float(old_model.sum() / old_exact.sum())

    exact_displacement = truth[:, -1] - own[:, -1]
    model_displacement = decoded[:, -1] - own[:, -1]
    exact_shift = exact_displacement.norm(dim=-1)
    direction = exact_displacement / exact_shift.clamp_min(1e-12).unsqueeze(-1)
    model_shift = (model_displacement * direction).sum(dim=-1)
    new_ratio = float(model_shift.double().sum() / exact_shift.double().sum())

    assert new_ratio == pytest.approx(old_ratio, rel=1e-9, abs=1e-12)


def test_response_projection_sees_a_purely_horizontal_switch() -> None:
    """A horizontal-only counterfactual must give a finite, correct response.

    The old vertical ratio divides by a zero vertical shift here, so the
    two-dimensional experiment is unmeasurable without the projection.
    """

    torch.manual_seed(1)
    batch, horizon, dt = 32, 6, 0.16
    position = torch.randn(batch, 2)
    velocity = torch.randn(batch, 2)
    gravity = torch.full((batch,), 4.0)
    accel_x = torch.full((batch,), 3.0)
    own = switched_trajectory(position, velocity, gravity, horizon=horizon, dt=dt)
    truth = switched_trajectory(
        position, velocity, gravity, horizon=horizon, dt=dt, accel_x=accel_x
    )
    # the switch is purely horizontal, so the vertical shift is exactly zero
    assert torch.allclose(truth[:, -1, 1], own[:, -1, 1], atol=1e-6)
    assert (truth[:, -1, 0] - own[:, -1, 0]).abs().min() > 0.1

    exact_displacement = truth[:, -1] - own[:, -1]
    exact_shift = exact_displacement.norm(dim=-1)
    direction = exact_displacement / exact_shift.unsqueeze(-1)

    for fraction in (0.0, 0.5, 1.0):
        decoded = own + fraction * (truth - own)
        model_shift = ((decoded[:, -1] - own[:, -1]) * direction).sum(dim=-1)
        ratio = float(model_shift.double().sum() / exact_shift.double().sum())
        assert ratio == pytest.approx(fraction, abs=1e-6)
