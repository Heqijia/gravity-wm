"""Evaluation: the four measurements of the method note, and the theory's bounds.

The method note asks for the acceleration error of rollouts as a function of
``g`` inside and outside the training range, the latent prediction error on
real windows, and the ablations.  The theory note adds two checks that can be
made numerically: Corollary 2's test-time position bound, and the KKT
residuals that Theorem 1 claims for the returned iterate.

Three points of protocol.

*Everything is measured with the held-out evaluation probe.*  The constraint is
optimised through ``h``, so a predictor could in principle satisfy it by
exploiting ``h`` rather than by moving correctly.  Scoring with ``h_eval``,
which shares no training episode with ``h``, is what makes that failure
visible; Remark 8(i)'s correction term -- the disagreement between the two
probes on predicted latents -- is also reported, and needs no labels.

*The counterfactual target is the exact switched trajectory.*  For a window
whose true state at step ``t`` is ``(p_t, v_t)``, the true positions after a
switch to ``g'`` are ``p_t + v_t s dt + g' (s dt)^2 / 2``.  No video at ``g'``
is needed, which is the whole point of the construction.

*The bound is checked in the form it is stated.*  Corollary 2 is derived from
the stride-one identity of Lemma 7, so the ``C(theta)`` that enters it is
measured at stride one even when training used a longer stride for signal.
Both are reported.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from gwm.constraint import (
    ConstraintSpec,
    acceleration_vector,
    condition_on_acceleration,
    decode_positions,
    free_fall_violation,
    gravity_vector,
    rollout_latents,
)
from gwm.latents import LatentCache
from gwm.posttrain import jepa_loss, sample_counterfactual_gravity

__all__ = [
    "GravitySweepRow",
    "corollary_two_bound",
    "evaluate_counterfactual_sweep",
    "evaluate_kkt_residuals",
    "evaluate_latent_error",
    "evaluate_true_ood",
    "measure_probe_delta",
    "switched_trajectory",
]


def switched_trajectory(
    position: torch.Tensor,
    velocity: torch.Tensor,
    counterfactual: torch.Tensor,
    *,
    horizon: int,
    dt: float,
    accel_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """``p*_{t+s}`` for ``s = 1..horizon`` after a switch to ``counterfactual``.

    Shapes: ``position`` and ``velocity`` are ``(B, 2)``, ``counterfactual`` is
    a ``(B,)`` scalar gravity and ``accel_x`` an optional ``(B,)`` horizontal
    acceleration.  Returns ``(B, horizon, 2)``.

    This is the ground truth the sweep measures against, so the horizontal
    component has to be carried here too: conditioning the model on
    ``(a_x, -g)`` and scoring it against the ``(0, -g)`` path would report the
    horizontal motion the model correctly produces as error.
    """

    acceleration = acceleration_vector(counterfactual, accel_x).to(position.dtype)
    steps = torch.arange(1, int(horizon) + 1, device=position.device, dtype=position.dtype) * float(dt)
    return (
        position.unsqueeze(1)
        + velocity.unsqueeze(1) * steps.reshape(1, -1, 1)
        + 0.5 * acceleration.unsqueeze(1) * steps.reshape(1, -1, 1) ** 2
    )


@torch.no_grad()
def measure_probe_delta(
    probe: nn.Module, cache: LatentCache, *, max_frames: int = 200_000
) -> dict[str, float]:
    """``delta``: the probe's RMS position error on latents from real frames.

    This is the quantity assumption (2) of the theory bounds, and the only way
    the probe's accuracy enters the convergence analysis (through Lemma 2).
    """

    latents = cache.latents.reshape(-1, cache.latent_dim)
    target = cache.state.reshape(-1, cache.state.shape[-1])[:, :2]
    if latents.shape[0] > max_frames:
        index = torch.randperm(latents.shape[0], device=latents.device)[:max_frames]
        latents, target = latents[index], target[index]
    predicted = probe(latents)
    error = (predicted - target).double()
    delta = float(error.pow(2).sum(dim=-1).mean().sqrt())
    return {
        "delta_world": delta,
        "delta_pixels": delta * cache.pixels_per_unit,
        "delta_x": float(error[:, 0].pow(2).mean().sqrt()),
        "delta_z": float(error[:, 1].pow(2).mean().sqrt()),
        "frames": int(latents.shape[0]),
    }


@dataclass
class GravitySweepRow:
    """Counterfactual accuracy at one conditioning gravity."""

    gravity: float
    windows: int
    acceptance: float
    acceleration_rmse: float
    acceleration_bias: float
    acceleration_bias_fraction: float
    violation_stride: float
    violation_unit_stride: float
    position_rmse_by_step: list[float]
    position_rmse_final: float
    response_ratio: float
    probe_disagreement_by_step: list[float]
    latent_rmse_by_step: list[float]
    # The horizontal component of the conditioning acceleration.  Zero for every
    # scalar-gravity experiment, which is why it carries a default and sits last.
    accel_x: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def _sweep_one_gravity(  # noqa: PLR0913
    model: nn.Module,
    probe_train: nn.Module,
    probe_eval: nn.Module,
    cache: LatentCache,
    gravity_value: float,
    *,
    spec: ConstraintSpec,
    safe_box: tuple[float, float, float, float] | None,
    margin: float,
    batch_size: int,
    max_windows: int,
    seed: int,
    accel_x_value: float = 0.0,
    accel_x_index: int | None = None,
) -> GravitySweepRow:
    length = spec.history + spec.horizon
    episode_index, start = cache.windows(length)
    total = episode_index.shape[0]
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed))[
        : min(max_windows, total)
    ].to(episode_index.device)
    # Stride-one, acceleration form: the form Lemma 7 and Corollary 2 (13) are
    # derived in, and the form the acceleration diagnostics below are read
    # from, whichever form the run itself was constrained with.
    unit_spec = ConstraintSpec(
        horizon=spec.horizon, stride=1, history=spec.history, form="acceleration"
    )

    accumulate = {
        "windows": 0,
        "drawn": 0,
        "accel_sq": 0.0,
        "accel_signed": torch.zeros(2, dtype=torch.float64),
        "violation": 0.0,
        "violation_unit": 0.0,
        "position_sq": torch.zeros(spec.horizon, dtype=torch.float64),
        "disagree_sq": torch.zeros(spec.horizon, dtype=torch.float64),
        "latent_sq": torch.zeros(spec.horizon, dtype=torch.float64),
        "response": 0.0,
        "response_exact": 0.0,
    }
    for begin in range(0, len(order), batch_size):
        chunk = order[begin : begin + batch_size]
        batch = cache.gather_windows(episode_index[chunk], start[chunk], length)
        size = batch["latents"].shape[0]
        device = batch["latents"].device
        target_gravity = torch.full((size,), float(gravity_value), device=device)
        target_accel_x = (
            None
            if accel_x_index is None
            else torch.full((size,), float(accel_x_value), device=device)
        )
        accepted = torch.ones(size, dtype=torch.bool, device=device)
        if safe_box is not None:
            # Same in-frame restriction as training, so the test law is
            # dominated by P with density one on the kept set.  A degenerate
            # range pins the draw to this grid point in both components.
            _unused, accepted, _ax = sample_counterfactual_gravity(
                {**batch, "state": batch["state"]},
                low=float(gravity_value),
                high=float(gravity_value) + 1e-9,
                spec=spec,
                dt=cache.dt,
                safe_box=safe_box,
                margin=margin,
                generator=torch.Generator(device=device).manual_seed(seed),
                max_resamples=0,
                accel_x_low=float(accel_x_value),
                accel_x_high=float(accel_x_value) + (1e-9 if accel_x_index is not None else 0.0),
            )
        accumulate["drawn"] += size
        if not bool(accepted.any()):
            continue
        action = condition_on_acceleration(
            batch["action"],
            gravity=target_gravity,
            accel_x=target_accel_x,
            action_mean=cache.action_mean,
            action_std=cache.action_std,
            accel_x_index=accel_x_index,
        )
        condition = model.action_encoder(action)
        predicted = rollout_latents(
            model, batch["latents"], condition, history=spec.history, horizon=spec.horizon
        )
        sequence = torch.cat([batch["latents"][:, : spec.history], predicted], dim=1)
        decoded_eval = decode_positions(probe_eval, sequence)
        decoded_train = decode_positions(probe_train, sequence)

        per_sample, _residual_pair, _residual_target = free_fall_violation(
            decoded_eval,
            gravity=acceleration_vector(batch["gravity"], None),
            counterfactual=acceleration_vector(target_gravity, target_accel_x),
            spec=spec,
            dt=cache.dt,
        )
        # The acceleration diagnostics come from this call, never from the one
        # above: under Proposition 1's position form ``free_fall_violation``
        # returns decoded positions and their targets, not accelerations, so
        # reading the residual off the configured form silently reported a
        # position residual in the acceleration columns.  The stride-one
        # acceleration form is also the one Lemma 7 and Corollary 2 are stated
        # in, so one call serves both.
        unit_sample, accelerations, targets = free_fall_violation(
            decoded_eval,
            gravity=acceleration_vector(batch["gravity"], None),
            counterfactual=acceleration_vector(target_gravity, target_accel_x),
            spec=unit_spec,
            dt=cache.dt,
        )
        pivot = spec.history - 1
        truth = switched_trajectory(
            batch["state"][:, pivot, :2],
            batch["state"][:, pivot, 2:4],
            target_gravity,
            horizon=spec.horizon,
            dt=cache.dt,
            accel_x=target_accel_x,
        )
        position_error = decoded_eval[:, spec.history :] - truth
        # How far the model moves when the conditioning gravity changes,
        # against how far it should.  Switching g -> g' displaces the object
        # by (g - g')(s dt)^2 / 2 at step s, and the decoded prediction is
        # compared with the trajectory the window's *own* gravity would give,
        # which is available in closed form -- so no second rollout is needed.
        # The ratio is 1 for a correct response and 0 for a gravity-blind one.
        # On the pretrained model it is near 1 for small shifts and falls to
        # 0.47 at the far end: the response *saturates* rather than being
        # absent, which is the failure the constraint has to repair and which
        # an aggregate position error hides.
        own = switched_trajectory(
            batch["state"][:, pivot, :2],
            batch["state"][:, pivot, 2:4],
            batch["gravity"],
            horizon=spec.horizon,
            dt=cache.dt,
        )
        # The response is measured *along the displacement the switch causes*,
        # which is the only form that survives a counterfactual with a
        # horizontal component.  For a purely vertical switch the unit vector is
        # -z and this reduces, term for term, to the ratio of z-shifts the
        # scalar experiments reported.
        exact_displacement = truth[:, -1] - own[:, -1]
        model_displacement = decoded_eval[:, -1] - own[:, -1]
        exact_shift = exact_displacement.norm(dim=-1)
        direction = exact_displacement / exact_shift.clamp_min(1e-12).unsqueeze(-1)
        model_shift = (model_displacement * direction).sum(dim=-1)
        disagreement = decoded_eval[:, spec.history :] - decoded_train[:, spec.history :]
        real_latents = batch["latents"][:, spec.history : spec.history + spec.horizon]

        keep = accepted
        count = int(keep.sum())
        accumulate["windows"] += count
        accumulate["violation"] += float(per_sample[keep].double().sum())
        accumulate["violation_unit"] += float(unit_sample[keep].double().sum())
        residual = (accelerations - targets)[keep].double()
        accumulate["accel_sq"] += float(residual.pow(2).sum(-1).mean(-1).sum())
        # The *signed* mean of the residual, as well as its RMS.  A small
        # systematic residual integrates into position error quadratically
        # (Remark 7) while a larger unsystematic one does not, and the squared
        # violation ``c`` cannot tell them apart -- measured here, a penalty
        # run reached a lower acceleration RMS than the switching method yet a
        # 1.6x larger position error at the end of the rollout.
        accumulate["accel_signed"] += residual.mean(dim=1).sum(0).cpu()
        accumulate["position_sq"] += position_error[keep].double().pow(2).sum(-1).sum(0).cpu()
        accumulate["disagree_sq"] += disagreement[keep].double().pow(2).sum(-1).sum(0).cpu()
        accumulate["latent_sq"] += (
            (predicted - real_latents)[keep].double().pow(2).sum(-1).sum(0).cpu()
        )
        accumulate["response"] += float(model_shift[keep].double().sum())
        accumulate["response_exact"] += float(exact_shift[keep].double().sum())

    windows = max(1, accumulate["windows"])
    return GravitySweepRow(
        gravity=float(gravity_value),
        windows=int(accumulate["windows"]),
        acceptance=accumulate["windows"] / max(1, accumulate["drawn"]),
        accel_x=float(accel_x_value) if accel_x_index is not None else 0.0,
        acceleration_rmse=math.sqrt(accumulate["accel_sq"] / windows),
        acceleration_bias=float((accumulate["accel_signed"] / windows).norm()),
        acceleration_bias_fraction=(
            float((accumulate["accel_signed"] / windows).norm())
            / math.sqrt(accumulate["accel_sq"] / windows)
            if accumulate["accel_sq"] > 0
            else 0.0
        ),
        violation_stride=accumulate["violation"] / windows,
        violation_unit_stride=accumulate["violation_unit"] / windows,
        position_rmse_by_step=[
            math.sqrt(value / windows) for value in accumulate["position_sq"].tolist()
        ],
        position_rmse_final=math.sqrt(float(accumulate["position_sq"][-1]) / windows),
        response_ratio=(
            accumulate["response"] / accumulate["response_exact"]
            if abs(accumulate["response_exact"]) > 1e-9
            else float("nan")
        ),
        probe_disagreement_by_step=[
            math.sqrt(value / windows) for value in accumulate["disagree_sq"].tolist()
        ],
        latent_rmse_by_step=[
            math.sqrt(value / windows) for value in accumulate["latent_sq"].tolist()
        ],
    )


def evaluate_counterfactual_sweep(
    model: nn.Module,
    probe_train: nn.Module,
    probe_eval: nn.Module,
    cache: LatentCache,
    gravities: list[float],
    *,
    spec: ConstraintSpec,
    safe_box: tuple[float, float, float, float] | None,
    margin: float = 0.0,
    batch_size: int = 256,
    max_windows: int = 8192,
    seed: int = 0,
    accel_x_values: list[float] | None = None,
    accel_x_index: int | None = None,
) -> list[GravitySweepRow]:
    """Counterfactual accuracy across conditioning accelerations.

    The context is always a real window from the training gravity
    distribution; only the conditioning acceleration moves.  This is the
    protocol of Theorem 2, in which the full method's error is bounded
    independently of the shift while a gravity-blind predictor's grows with it.

    ``accel_x_values`` sweeps the horizontal component instead of the vertical
    one, at the gravity given in ``gravities`` -- the two-dimensional
    experiment, reported as a second one-dimensional slice rather than a full
    grid because the grid costs the product and says little more.
    """

    if accel_x_values is None:
        pairs = [(value, 0.0) for value in gravities]
    else:
        if len(gravities) != 1:
            raise ValueError(
                "sweeping accel_x expects exactly one gravity to hold fixed; "
                f"got {len(gravities)}"
            )
        pairs = [(gravities[0], value) for value in accel_x_values]
    return [
        _sweep_one_gravity(
            model,
            probe_train,
            probe_eval,
            cache,
            gravity,
            spec=spec,
            safe_box=safe_box,
            margin=margin,
            batch_size=batch_size,
            max_windows=max_windows,
            seed=seed,
            accel_x_value=horizontal,
            accel_x_index=accel_x_index,
        )
        for gravity, horizontal in pairs
    ]


def corollary_two_bound(
    *,
    delta: float,
    constraint: float,
    dt: float,
    horizon: int,
    step: int,
    kappa: float = 1.0,
) -> float:
    """RHS of Corollary 2 (13) at horizon index ``step``.

    ``sqrt(kappa) [ (2s+1) delta + dt^2 sqrt(H s(s+1)(2s+1)/6) sqrt(C) ]``.
    ``kappa`` is one here because the evaluation law is the training law
    restricted the same way, so its density with respect to ``P`` is one on the
    kept set.  ``constraint`` must be the stride-one ``C(theta)``, which is the
    form Lemma 7 and hence (13) are derived for.
    """

    if step < 1 or step > horizon:
        raise ValueError("step must lie in 1..horizon")
    amplification = math.sqrt(horizon * step * (step + 1) * (2 * step + 1) / 6.0)
    return math.sqrt(kappa) * (
        (2 * step + 1) * float(delta) + float(dt) ** 2 * amplification * math.sqrt(max(constraint, 0.0))
    )


@torch.no_grad()
def evaluate_latent_error(
    model: nn.Module,
    cache: LatentCache,
    *,
    spec: ConstraintSpec,
    batch_size: int = 256,
    max_windows: int = 8192,
    seed: int = 0,
) -> dict[str, Any]:
    """``L(theta)`` on real windows: does post-training degrade the world model?

    Reported both as the theory's ``E[sum_s ||zhat - z||^2]`` and as the
    dimension-averaged value the pretraining objective uses, so the number can
    be compared with the pretraining logs.
    """

    length = spec.history + spec.horizon
    episode_index, start = cache.windows(length)
    total = episode_index.shape[0]
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed))[
        : min(max_windows, total)
    ].to(episode_index.device)
    summed, averaged, count = 0.0, 0.0, 0
    per_step = torch.zeros(spec.horizon, dtype=torch.float64)
    for begin in range(0, len(order), batch_size):
        chunk = order[begin : begin + batch_size]
        batch = cache.gather_windows(episode_index[chunk], start[chunk], length)
        loss, mean_squared = jepa_loss(model, batch, spec=spec)
        size = batch["latents"].shape[0]
        summed += float(loss) * size
        averaged += float(mean_squared) * size
        condition = model.action_encoder(batch["action"])
        predicted = rollout_latents(
            model, batch["latents"], condition, history=spec.history, horizon=spec.horizon
        )
        target = batch["latents"][:, spec.history : spec.history + spec.horizon]
        per_step += (predicted - target).double().pow(2).sum(-1).sum(0).cpu()
        count += size
    return {
        "jepa_loss_sum_form": summed / max(1, count),
        "jepa_loss_mean_form": averaged / max(1, count),
        "jepa_loss_by_step": [value / max(1, count) for value in per_step.tolist()],
        "windows": count,
    }


@torch.no_grad()
def evaluate_true_ood(
    model: nn.Module,
    probe_eval: nn.Module,
    cache: LatentCache,
    *,
    spec: ConstraintSpec,
    batch_size: int = 256,
    max_windows: int = 4096,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Episodes whose context already evolves under an out-of-range gravity.

    This is the setting of Remark 8(iii): the context gravity lies outside the
    support of the training episodes, so no finite density bound ``kappa``
    exists and Corollary 2 does not apply.  What remains is the pathwise bound
    of Lemma 7, whose leading term needs no future frames and is computed here
    alongside the measured error, so the two can be compared.
    """

    length = spec.history + spec.horizon
    rows: list[dict[str, Any]] = []
    for value in sorted({float(g) for g in cache.gravity.tolist()}):
        subset = cache.select((cache.gravity - value).abs() < 1e-6)
        if subset.episodes == 0 or subset.frames < length:
            continue
        episode_index, start = subset.windows(length)
        total = episode_index.shape[0]
        order = torch.randperm(total, generator=torch.Generator().manual_seed(seed))[
            : min(max_windows, total)
        ].to(episode_index.device)
        position_sq = torch.zeros(spec.horizon, dtype=torch.float64)
        pathwise = torch.zeros(spec.horizon, dtype=torch.float64)
        violation, count = 0.0, 0
        for begin in range(0, len(order), batch_size):
            chunk = order[begin : begin + batch_size]
            batch = subset.gather_windows(episode_index[chunk], start[chunk], length)
            condition = model.action_encoder(batch["action"])
            predicted = rollout_latents(
                model, batch["latents"], condition, history=spec.history, horizon=spec.horizon
            )
            sequence = torch.cat([batch["latents"][:, : spec.history], predicted], dim=1)
            decoded = decode_positions(probe_eval, sequence)
            per_sample, _acc, _tgt = free_fall_violation(
                decoded,
                gravity=gravity_vector(batch["gravity"]),
                counterfactual=gravity_vector(batch["gravity"]),
                spec=spec,
                dt=subset.dt,
            )
            truth = batch["state"][:, spec.history :, :2]
            error = decoded[:, spec.history :] - truth
            size = batch["latents"].shape[0]
            position_sq += error.double().pow(2).sum(-1).sum(0).cpu()
            # Lemma 7's label-free leading term: s H dt^2 sqrt(c).
            steps = torch.arange(1, spec.horizon + 1, dtype=torch.float64)
            pathwise += (
                steps.reshape(1, -1)
                * spec.horizon
                * subset.dt**2
                * per_sample.double().sqrt().reshape(-1, 1).cpu()
            ).sum(0)
            violation += float(per_sample.double().sum())
            count += size
        rows.append(
            {
                "gravity": value,
                "episodes": int(subset.episodes),
                "windows": count,
                "violation": violation / max(1, count),
                "position_rmse_by_step": [
                    math.sqrt(v / max(1, count)) for v in position_sq.tolist()
                ],
                "lemma7_leading_term_by_step": [
                    v / max(1, count) for v in pathwise.tolist()
                ],
            }
        )
    return rows


def evaluate_kkt_residuals(
    history: list[dict[str, Any]],
    *,
    epsilon: float,
    gate: float,
    constraint_at_output: float,
    zeta_c: float,
    grad_l_bound: float,
    mu: float,
) -> dict[str, Any]:
    """The three residuals of Definition 1, from the logged trajectory.

    ``r_S`` uses ``||d_{n*}||`` plus the noise floor ``5 M zeta / mu`` of
    Theorem 1(c); with the certificate taken directly from the run, the term
    reported here is the measurable part ``||d_{n*}||``.  ``r_F`` equals the
    relative accuracy ``zeta_C`` of the constraint estimate, and ``r_C`` is
    ``4M(1 - gamma + zeta_C)/mu``, which Remark 2 notes does not decrease with
    the number of iterations.
    """

    optimise = [row for row in history if row.get("mode") == "optimise"]
    if not optimise:
        return {"status": "no optimise iterations recorded"}
    best = min(optimise, key=lambda row: row["direction_norm"])
    feasibility = max(0.0, constraint_at_output / epsilon - 1.0)
    return {
        "stationarity_certificate": best["direction_norm"],
        "stationarity_step": best["step"],
        "feasibility_residual_measured": feasibility,
        "feasibility_residual_predicted": zeta_c,
        "complementarity_residual_bound": 4.0 * grad_l_bound * (1.0 - gate + zeta_c) / max(mu, 1e-12),
        "epsilon": epsilon,
        "constraint_at_output": constraint_at_output,
        "gate": gate,
        "zeta_C": zeta_c,
        "grad_l_bound_M": grad_l_bound,
        "mu": mu,
        "feasible": constraint_at_output <= (1.0 + zeta_c) * epsilon,
    }
