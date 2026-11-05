"""Gravity-constrained post-training: the driver and its ablations.

Implements ``min L(theta)`` s.t. ``C(theta) <= eps`` over the predictor of a
pretrained Semigroup-JEPA world model, together with the ablations the method
note asks for:

``pretrained``
    the pretrained predictor, untouched (``theta_0``).
``jepa``
    continued JEPA training of the predictor for the same number of steps --
    gradient descent on ``L`` alone, the ablation of Theorem 2(b).
``factual``
    the constraint with ``g'`` fixed to the real gravity of each window, the
    ablation of Theorem 2(c).  Theorem 2(c) predicts this leaves continued
    JEPA training unchanged whenever the pretrained predictor is already
    physically consistent on real windows.
``counterfactual``
    the full method: ``g' ~ U(G)`` with ``G`` wider than the training range.
``penalty``
    ``L + lambda C`` with counterfactual ``g'`` and a fixed ``lambda``, included
    because Remark 10 claims a fixed penalty cannot certify feasibility.

The distribution ``P``.  Definition 1 pairs a real window with a
counterfactual gravity, and the tolerance argument requires the counterfactual
rollout to stay inside the range of positions the probe was trained on -- "one
reason to keep H short".  ``P`` is therefore defined here as a real window
paired with ``g' ~ U(G)``, *restricted to pairs whose exact counterfactual
trajectory stays inside the rendered field of view*.  The restriction is
computed from the simulator state, is applied identically at evaluation time
so that the test law is absolutely continuous with respect to ``P``, and its
acceptance rate is reported.  Nothing in Theorem 1 requires ``g'`` to be
independent of the window; only Corollary 2 needs the test law to be dominated
by ``P``, which the matched restriction gives.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gwm.bounce_data import counterfactual_hits_floor
from gwm.constraint import (
    ConstraintSpec,
    acceleration_vector,
    condition_on_acceleration,
    decode_positions,
    free_fall_violation,
    gravity_vector,
    rollout_latents,
)
from gwm.latents import LatentCache, trainable_parameters
from gwm.switching import OPTIMISE, SwitchingConfig, SwitchingOptimizer, flat_gradient

__all__ = [
    "EPSILON_SOURCES",
    "PostTrainConfig",
    "PostTrainResult",
    "VARIANTS",
    "calibrate_tolerance",
    "jepa_loss",
    "run_post_training",
    "sample_counterfactual_gravity",
]

VARIANTS = ("pretrained", "jepa", "factual", "counterfactual", "penalty")


@dataclass(frozen=True)
class PostTrainConfig:
    variant: str = "counterfactual"
    steps: int = 3000
    batch_size: int = 192
    constraint_batch_size: int | None = 384
    gravity_low: float = 1.0
    gravity_high: float = 9.0
    # The two-dimensional experiment.  ``accel_x_index`` is the action
    # coordinate the dataset reserves for the horizontal acceleration; with it
    # unset the counterfactual is the scalar gravity of every other experiment
    # and nothing below changes.
    accel_x_index: int | None = None
    accel_x_low: float = 0.0
    accel_x_high: float = 0.0
    rho: float = 0.25
    epsilon_source: str = "real_latents"
    epsilon_scale: float = 1.0
    # An absolute tolerance, overriding the calibration.  Needed for a staged
    # schedule: the calibration measures the *current* checkpoint's factual
    # rollout, so a stage that damages the model relaxes its own successor's
    # tolerance.  Measured on ball_ax, stage one took the factual violation
    # from 0.0113 to 8.711 and stage two then calibrated eps at 10.89 -- 773x
    # the original -- which made the constraint vacuous and let OPTIMISE undo
    # everything stage one had achieved.  Pinning eps across stages is the fix.
    epsilon_override: float | None = None
    penalty_lambda: float = 1.0
    trainable: tuple[str, ...] = ("predictor", "pred_proj", "action_encoder")
    in_frame_filter: bool = True
    in_frame_margin: float = 0.0
    # Height of the object's centre when it rests on the floor, for the scenes
    # that have one.  ``None`` means no floor, which is every contact-free
    # dataset.  See ``contact_free_context`` for the other half of option A.
    floor_resting_z: float | None = None
    # Drop windows whose *context* spans a contact.  The position form's
    # implied velocity ``v = (p_t - p_{t-1})/dt + g dt/2`` is a free-fall
    # quantity and is wrong across a bounce, so such a window would enter the
    # constraint with a corrupted initial condition.
    contact_free_context: bool = True
    calibration_windows: int = 4096
    prime_batches: int = 8
    eval_every: int = 100
    seed: int = 0
    switching: SwitchingConfig = field(default_factory=SwitchingConfig)
    constraint: ConstraintSpec = field(default_factory=ConstraintSpec)

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.steps < 0 or self.batch_size < 1:
            raise ValueError("steps must be non-negative and batch_size positive")
        if self.gravity_low >= self.gravity_high:
            raise ValueError("gravity_low must be below gravity_high")
        if self.accel_x_index is not None and self.accel_x_low > self.accel_x_high:
            raise ValueError("accel_x_low must not exceed accel_x_high")
        if self.accel_x_index is None and (self.accel_x_low or self.accel_x_high):
            raise ValueError("accel_x range given without accel_x_index")
        if self.epsilon_override is not None and not self.epsilon_override > 0.0:
            raise ValueError("epsilon_override must be positive")
        if self.rho <= 0.0:
            raise ValueError("rho must be positive so that the tolerance exceeds E[c_real]")
        if self.epsilon_source not in EPSILON_SOURCES:
            raise ValueError(f"epsilon_source must be one of {EPSILON_SOURCES}")
        if self.epsilon_scale <= 0.0:
            raise ValueError("epsilon_scale must be positive")
        if self.penalty_lambda < 0.0:
            raise ValueError("penalty_lambda must be non-negative")
        if self.prime_batches < 1 or self.calibration_windows < 1:
            raise ValueError("prime_batches and calibration_windows must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trainable"] = list(self.trainable)
        payload["constraint"] = asdict(self.constraint)
        payload["switching"] = self.switching.to_dict()
        return payload


def jepa_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    spec: ConstraintSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``L(theta) = E[sum_s ||zhat_{t+s} - z_{t+s}||^2]`` on real windows.

    The rollout is conditioned on the window's own gravity, which is already
    the gravity coordinate of the stored action, and the targets come from the
    frozen encoder so they carry no gradient.  The sum over the horizon and the
    squared Euclidean norm are the form the theory analyses; the pretraining
    objective of ``sg_jepa`` additionally averages over latent dimensions and
    discounts the horizon, which only rescales ``L`` and is reported separately
    for comparability.
    """

    latents = batch["latents"]
    condition = model.action_encoder(batch["action"])
    predicted = rollout_latents(
        model, latents, condition, history=spec.history, horizon=spec.horizon
    )
    target = latents[:, spec.history : spec.history + spec.horizon].detach()
    squared = (predicted - target).pow(2)
    per_sample = squared.sum(dim=-1).sum(dim=-1)
    return per_sample.mean(), squared.mean()


def sample_counterfactual_gravity(
    batch: dict[str, torch.Tensor],
    *,
    low: float,
    high: float,
    spec: ConstraintSpec,
    dt: float,
    safe_box: tuple[float, float, float, float] | None,
    margin: float,
    generator: torch.Generator,
    max_resamples: int = 0,
    accel_x_low: float = 0.0,
    accel_x_high: float = 0.0,
    floor_resting_z: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Draw ``g' ~ U(low, high)`` per window and mask the out-of-frame pairs.

    The exact counterfactual trajectory is known in closed form from the
    simulator state at step ``t``, so whether a draw would push the object out
    of the rendered field of view -- and therefore out of the range of
    positions the probe was trained on -- can be decided before any network is
    run.

    Out-of-frame pairs are **rejected**, not resampled.  Resampling would
    replace ``g'`` for that particular window and so change the conditional
    law of ``g'`` given the window, systematically under-drawing the large
    shifts for windows near the edge of the view -- measured, it took the
    training acceptance rate to 1.00 while the evaluation grid rejected up to
    17% at the far end, and the training violation came out a factor 1.4 below
    the evaluated one.  Rejection keeps ``P`` equal to "real window times
    ``g' ~ U(G)``, restricted to in-frame pairs", which is the law the
    evaluation uses, so Corollary 2 applies with density one.  ``max_resamples``
    is kept for the biased variant but defaults to zero.
    """

    device = batch["latents"].device
    size = batch["latents"].shape[0]
    gravity = torch.empty(size, device=device).uniform_(float(low), float(high), generator=generator)
    # The horizontal component is drawn from its own uniform law and makes the
    # counterfactual an acceleration *vector*.  A degenerate range means the
    # scalar experiment, and ``None`` keeps every downstream call identical to
    # it rather than passing an all-zero tensor around.
    accel_x = None
    if float(accel_x_high) > float(accel_x_low):
        accel_x = torch.empty(size, device=device).uniform_(
            float(accel_x_low), float(accel_x_high), generator=generator
        )
    accepted = torch.ones(size, dtype=torch.bool, device=device)
    if safe_box is None and floor_resting_z is None:
        return gravity, accepted, accel_x

    pivot = spec.history - 1
    position = batch["state"][:, pivot, :2]
    velocity = batch["state"][:, pivot, 2:4]
    steps = torch.arange(1, spec.horizon + 1, device=device, dtype=position.dtype) * float(dt)

    def inside(candidate: torch.Tensor) -> torch.Tensor:
        horizontal = None if accel_x is None else accel_x.to(candidate.dtype)
        acceleration = acceleration_vector(candidate, horizontal).to(position.dtype)
        path = (
            position.unsqueeze(1)
            + velocity.unsqueeze(1) * steps.reshape(1, -1, 1)
            + 0.5 * acceleration.unsqueeze(1) * steps.reshape(1, -1, 1) ** 2
        )
        ok = torch.ones(path.shape[0], dtype=torch.bool, device=path.device)
        if safe_box is not None:
            left, right, bottom, top = safe_box
            ok = ok & (
                (path[..., 0] >= left + margin)
                & (path[..., 0] <= right - margin)
                & (path[..., 1] >= bottom + margin)
                & (path[..., 1] <= top - margin)
            ).all(dim=1)
        if floor_resting_z is not None:
            # The counterfactual, not the factual, decides this.  A window that
            # never touches the floor at ``g`` can bounce at ``g' = 9``, and the
            # free-fall identity the constraint compares against is false on
            # such a window just as it is on a real bounce.  Rejecting it is the
            # same move the in-frame filter makes, and like that one it rejects
            # without resampling so ``P`` stays the law Corollary 2 needs.
            ok = ok & (path[..., 1] > float(floor_resting_z)).all(dim=1)
        return ok

    ok = inside(gravity)
    for _ in range(int(max_resamples)):
        if bool(ok.all()):
            break
        redraw = torch.empty(size, device=device).uniform_(
            float(low), float(high), generator=generator
        )
        gravity = torch.where(ok, gravity, redraw)
        ok = ok | inside(gravity)
    return gravity, ok, accel_x


EPSILON_SOURCES = ("real_latents", "factual_rollout")


@torch.no_grad()
def calibrate_tolerance(
    probe: nn.Module,
    cache: LatentCache,
    *,
    spec: ConstraintSpec,
    windows: int,
    seed: int,
    batch_size: int = 512,
    source: str = "real_latents",
    model: nn.Module | None = None,
    contact_free_only: bool = True,
) -> dict[str, float]:
    """``eps = (1 + rho) E[c_real]``, measured on held-out real windows.

    Two sources, because they ask for different things and only one of them is
    always attainable.

    ``real_latents`` is the method note's definition: the violation with the
    real future latents in place of predictions and ``g' = g``.  Its
    accelerations are those of the true trajectory, so what remains is the
    probe's own noise, and the true dynamics satisfy the constraint in
    expectation -- enforcing it cannot bias the predictor away from them.  But
    the constrained quantity ``C(theta)`` is measured on *predicted* latents,
    whose rollout error adds to the violation irreducibly, so a tolerance set
    this way can lie below anything the predictor can reach.  When it does,
    Algorithm 1 never leaves RESTORE, ``IO`` stays empty and Theorem 1 says
    nothing -- which is what happens here with an accurate probe.

    ``factual_rollout`` measures the same violation on the predictor's own
    rollout under the real gravity.  The tolerance is then attainable at
    ``theta_0`` by construction, and the constraint reads "do not become less
    physically consistent when the gravity changes than you already are at the
    gravity you were trained on" -- which is the generalisation the method is
    claiming.  It also makes condition (20) of Theorem 2(c) hold by
    construction, so the factual ablation behaves as the theorem predicts.
    """

    if source not in EPSILON_SOURCES:
        raise ValueError(f"source must be one of {EPSILON_SOURCES}, got {source!r}")
    if source == "factual_rollout" and model is None:
        raise ValueError("the factual_rollout tolerance needs the pretrained model")
    length = spec.history + spec.horizon
    episode_index, start = cache.windows(length)
    total = episode_index.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(total, generator=generator)[: min(int(windows), total)]
    values: list[torch.Tensor] = []
    dropped = 0
    for begin in range(0, len(order), batch_size):
        chunk = order[begin : begin + batch_size].to(episode_index.device)
        batch = cache.gather_windows(episode_index[chunk], start[chunk], length)
        # eps must be calibrated on windows where the free-fall identity is
        # true, or a bounce inflates it and the constraint goes vacuous.  This
        # is the same failure the staged tolerance hit from the other
        # direction: a damaged factual rollout produced eps 773x the original
        # and OPTIMISE then undid everything the constraint had achieved.  Here
        # the whole window matters, not just the context: ``c_real`` is a
        # second difference over the full span.
        if contact_free_only and "contact" in batch:
            keep = ~batch["contact"][:, : length - 1].any(dim=1)
            dropped += int((~keep).sum())
            if not bool(keep.any()):
                continue
            batch = {
                key: value[keep] if torch.is_tensor(value) and value.shape[:1] == keep.shape
                else value
                for key, value in batch.items()
            }
        if source == "real_latents":
            sequence = batch["latents"][:, :length]
        else:
            condition = model.action_encoder(batch["action"])
            predicted = rollout_latents(
                model,
                batch["latents"],
                condition,
                history=spec.history,
                horizon=spec.horizon,
            )
            sequence = torch.cat([batch["latents"][:, : spec.history], predicted], dim=1)
        positions = decode_positions(probe, sequence)
        vectors = gravity_vector(batch["gravity"])
        per_sample, _accelerations, _targets = free_fall_violation(
            positions, gravity=vectors, counterfactual=vectors, spec=spec, dt=cache.dt
        )
        values.append(per_sample.detach().cpu())
    if not values:
        raise RuntimeError("every calibration window spanned a contact")
    stacked = torch.cat(values).double()
    mean = float(stacked.mean())
    return {
        "source": source,
        "c_real_mean": mean,
        "windows_dropped_for_contact": dropped,
        "windows_used": int(stacked.numel()),
        "c_real_std": float(stacked.std()),
        "c_real_median": float(stacked.median()),
        "c_real_p95": float(torch.quantile(stacked, 0.95)),
        "c_real_sem": float(stacked.std() / math.sqrt(len(stacked))),
        "windows": int(len(stacked)),
    }


@dataclass
class PostTrainResult:
    variant: str
    epsilon: float
    calibration: dict[str, Any]
    summary: dict[str, Any]
    history: list[dict[str, Any]]
    probes: dict[str, Any]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "epsilon": self.epsilon,
            "calibration": self.calibration,
            "summary": self.summary,
            "history": self.history,
            "probes": self.probes,
            "diagnostics": self.diagnostics,
        }


def _batch_sampler(cache: LatentCache, length: int, generator: torch.Generator):
    episode_index, start = cache.windows(length)
    total = episode_index.shape[0]

    def draw(size: int) -> dict[str, torch.Tensor]:
        chunk = torch.randint(0, total, (int(size),), generator=generator, device="cpu")
        chunk = chunk.to(episode_index.device)
        return cache.gather_windows(episode_index[chunk], start[chunk], length)

    return draw, total


def _snapshot(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


@torch.no_grad()
def _restore(parameters: list[torch.nn.Parameter], snapshot: list[torch.Tensor]) -> None:
    for parameter, value in zip(parameters, snapshot, strict=True):
        parameter.copy_(value)


def run_post_training(
    model: nn.Module,
    probe_train: nn.Module,
    probe_eval: nn.Module,
    cache: LatentCache,
    heldout_cache: LatentCache,
    *,
    config: PostTrainConfig,
    safe_box: tuple[float, float, float, float] | None,
    device: torch.device,
    monitor: Any | None = None,
) -> PostTrainResult:
    """Run one post-training variant and return everything needed to audit it.

    The iteration follows Algorithm 1 literally: one minibatch per step
    supplies both ``grad L_B`` and ``grad C_B``, the moving average ``Chat`` is
    updated from that same ``C_B``, the mode is selected from ``Chat``, and the
    step is ``theta <- theta - eta d``.  Both gradients are always computed,
    which is the analysed form; the method note's option of skipping the second
    backward pass below the gate is available through
    ``SwitchingOptimizer.needs_constraint_gradient`` but is not taken here,
    because with cached latents the saving is small and the extra branching
    would make the two branches see different data.

    The returned predictor is ``theta_{n*}`` of Algorithm 1 line 13: the
    OPTIMISE iterate with the smallest ``||d_n||``, which Theorem 1(c) makes the
    approximate-KKT point.  That iterate is snapshotted when it occurs, since
    the final iterate carries no such certificate.
    """

    spec = config.constraint
    length = spec.history + spec.horizon
    if cache.frames < length:
        raise ValueError(
            f"episodes have {cache.frames} frames but the constraint needs {length}"
        )
    torch.manual_seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    cf_generator = torch.Generator(device=device).manual_seed(config.seed + 2)

    shared = dict(spec=spec, windows=config.calibration_windows, seed=config.seed + 3)
    calibration = calibrate_tolerance(
        probe_train, heldout_cache, source=config.epsilon_source, model=model, **shared
    )
    calibration_eval = calibrate_tolerance(
        probe_eval, heldout_cache, source=config.epsilon_source, model=model, **shared
    )
    # Always report the other source too: the gap between them is the
    # predictor's own irreducible rollout error in probe coordinates, and it
    # decides whether the tolerance is attainable at all.
    alternative = next(s for s in EPSILON_SOURCES if s != config.epsilon_source)
    calibration_other = calibrate_tolerance(
        probe_train, heldout_cache, source=alternative, model=model, **shared
    )
    # The tolerance is the method's dial.  Algorithm 1 seeks feasibility, not
    # the smallest possible violation: once C <= eps the RESTORE mode stops
    # and the projection holds C roughly constant (Proposition 2), so the
    # accuracy the method reaches is set by eps rather than by the iteration
    # count.  ``epsilon_scale`` makes that explicit, so a run can ask for a
    # tighter constraint and still get the feasibility certificate -- which is
    # what a fixed penalty weight cannot give (Remark 10).
    if config.epsilon_override is not None:
        epsilon = float(config.epsilon_override) * config.epsilon_scale
    else:
        epsilon = config.epsilon_scale * (1.0 + config.rho) * calibration["c_real_mean"]
    if not epsilon > 0.0:
        raise ValueError("calibration produced a non-positive tolerance")

    draw, window_count = _batch_sampler(cache, length, generator)
    parameters = trainable_parameters(model)
    optimizer = SwitchingOptimizer(parameters, epsilon=epsilon, config=config.switching)

    use_constraint = config.variant in {"factual", "counterfactual", "penalty"}
    counterfactual = config.variant in {"counterfactual", "penalty"}
    tally = {"accepted": 0, "drawn": 0, "contact_context": 0}

    def constraint_value(
        batch: dict[str, torch.Tensor], *, grad: bool
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """``C_B`` on one minibatch, with or without a graph."""

        with torch.set_grad_enabled(grad):
            if counterfactual:
                target_gravity, accepted, target_accel_x = sample_counterfactual_gravity(
                    batch,
                    low=config.gravity_low,
                    high=config.gravity_high,
                    spec=spec,
                    dt=cache.dt,
                    safe_box=safe_box if config.in_frame_filter else None,
                    margin=config.in_frame_margin,
                    floor_resting_z=config.floor_resting_z,
                    generator=cf_generator,
                    accel_x_low=config.accel_x_low,
                    accel_x_high=config.accel_x_high,
                )
            else:
                target_gravity = batch["gravity"]
                target_accel_x = None
                accepted = torch.ones_like(target_gravity, dtype=torch.bool)
                if config.floor_resting_z is not None:
                    # The factual arm needs this filter too, and for the same
                    # reason: the free-fall identity is false across a bounce
                    # whatever gravity it is evaluated at.  Leaving it out made
                    # the factual control measure bounce residuals instead of
                    # gravity error -- C_0/eps came out at 621 where the
                    # counterfactual arm, which did filter, sat at 36.2, and
                    # the sharpest control in the study was reading the wrong
                    # quantity.
                    pivot = spec.history - 1
                    accepted = accepted & ~counterfactual_hits_floor(
                        batch["state"][:, pivot, :2],
                        batch["state"][:, pivot, 2:4],
                        target_gravity,
                        spec_resting_z=float(config.floor_resting_z),
                        dt=cache.dt,
                        horizon=spec.horizon,
                    )
            if config.contact_free_context and "contact" in batch:
                # The other half of option A.  ``contact[t]`` covers the
                # interval from frame ``t`` to ``t+1``, so a context of
                # ``history`` frames spans intervals ``0 .. history-2``.  A
                # bounce in there corrupts the implied velocity the position
                # form starts the counterfactual rollout from, and the window
                # would enter ``C`` measuring the wrong thing rather than a
                # gravity error -- measured at 12.4 px against a 1.008 px
                # signal on the bouncing scene.
                clean = ~batch["contact"][:, : spec.history - 1].any(dim=1)
                tally["contact_context"] += int((~clean).sum())
                accepted = accepted & clean
            tally["accepted"] += int(accepted.sum())
            tally["drawn"] += int(accepted.numel())
            if not bool(accepted.any()):
                raise RuntimeError(
                    "no window survived the counterfactual filters; "
                    "if this is a bouncing dataset, check floor_resting_z and the contact rate"
                )
            action = condition_on_acceleration(
                batch["action"],
                gravity=target_gravity,
                accel_x=target_accel_x,
                action_mean=cache.action_mean,
                action_std=cache.action_std,
                accel_x_index=config.accel_x_index,
            )
            condition = model.action_encoder(action)
            predicted = rollout_latents(
                model, batch["latents"], condition, history=spec.history, horizon=spec.horizon
            )
            sequence = torch.cat([batch["latents"][:, : spec.history], predicted], dim=1)
            positions = decode_positions(probe_train, sequence)
            # The factual branch has no horizontal acceleration: the data was
            # generated with a_x = 0, which is what makes the counterfactual
            # direction something the model has never been shown.
            per_sample, accelerations, targets = free_fall_violation(
                positions,
                gravity=acceleration_vector(batch["gravity"], None),
                counterfactual=acceleration_vector(target_gravity, target_accel_x),
                spec=spec,
                dt=cache.dt,
            )
            value = per_sample[accepted].mean()
            info = {
                "gravity_shift_mean": float((target_gravity - batch["gravity"]).abs().mean()),
                "acceptance": float(accepted.float().mean()),
                "acceleration_rmse": float(
                    (accelerations - targets).detach().pow(2).sum(-1).mean().sqrt()
                ),
            }
        return value, info

    def zero_grad() -> None:
        for parameter in parameters:
            parameter.grad = None

    history: list[dict[str, Any]] = []
    best_snapshot = _snapshot(parameters)
    best_step = -1

    # Draw the priming batches for *every* variant, even those that do not use
    # the constraint.  The draws advance the sampler, so skipping them would
    # give the unconstrained ablations a different batch sequence from the
    # constrained ones and the comparison would not be controlled.
    primed = []
    for _ in range(config.prime_batches):
        batch = draw(config.constraint_batch_size or config.batch_size)
        if use_constraint:
            primed.append(float(constraint_value(batch, grad=False)[0]))
    optimizer.prime(primed or [0.0])

    for step in range(config.steps):
        batch = draw(config.batch_size)

        zero_grad()
        loss, mean_squared = jepa_loss(model, batch, spec=spec)
        loss.backward()
        grad_l = flat_gradient(parameters)
        loss_value = float(loss.detach())
        info: dict[str, float] = {"jepa_mean_squared": float(mean_squared.detach())}

        grad_c = None
        violation = float("nan")
        if use_constraint:
            zero_grad()
            value, constraint_info = constraint_value(batch, grad=True)
            value.backward()
            grad_c = flat_gradient(parameters)
            violation = float(value.detach())
            info.update(constraint_info)
            optimizer.observe_violation(violation)
            optimizer.select_mode()
        else:
            optimizer.state.mode = OPTIMISE

        if config.variant == "penalty":
            optimizer.state.mode = OPTIMISE
            record = optimizer.step(
                grad_l=grad_l + config.penalty_lambda * grad_c,
                grad_c=None,
                jepa_loss=loss_value,
                constraint=violation,
            )
        elif config.variant == "jepa":
            record = optimizer.step(
                grad_l=grad_l, grad_c=None, jepa_loss=loss_value, constraint=violation
            )
        else:
            record = optimizer.step(
                grad_l=grad_l, grad_c=grad_c, jepa_loss=loss_value, constraint=violation
            )

        if record.mode == OPTIMISE and optimizer.state.best_step == record.step:
            best_snapshot = _snapshot(parameters)
            best_step = record.step

        if step % max(1, config.eval_every) == 0 or step == config.steps - 1:
            row = record.to_dict()
            row.update(info)
            if use_constraint:
                # Estimation accuracy of Assumption 2: compare the moving
                # average that selects the mode with a large-batch estimate of
                # C.  ``zeta_C`` is this gap relative to ``eps``, and the third
                # condition of (7) requires ``2 zeta_C < 1 - gamma``.
                with torch.no_grad():
                    big = config.constraint_batch_size or config.batch_size
                    reference = float(constraint_value(draw(4 * big), grad=False)[0])
                row["constraint_large_batch"] = reference
                row["zeta_C_estimate"] = abs(record.constraint_ema - reference) / epsilon
            history.append(row)
            if monitor is not None:
                monitor(step, row)

    zero_grad()
    returned = "theta_n_star"
    if config.variant in {"counterfactual", "factual"} and best_step >= 0:
        _restore(parameters, best_snapshot)
    else:
        returned = "theta_N"

    diagnostics = {
        "window_count": int(window_count),
        "counterfactual_acceptance": (
            tally["accepted"] / tally["drawn"] if tally["drawn"] else float("nan")
        ),
        "counterfactual_draws": tally["drawn"],
        "gravity_range": [config.gravity_low, config.gravity_high],
        "accel_x_range": (
            None
            if config.accel_x_index is None
            else [config.accel_x_low, config.accel_x_high]
        ),
        "accel_x_index": config.accel_x_index,
        "dt": cache.dt,
        "dt_effective": cache.dt * spec.stride,
        "constraint_terms": spec.num_terms,
        "constraint_offsets": list(spec.offsets),
        "safe_box": list(safe_box) if safe_box is not None else None,
        "returned_iterate": returned,
        "returned_step": best_step if returned == "theta_n_star" else config.steps - 1,
    }
    return PostTrainResult(
        variant=config.variant,
        epsilon=epsilon,
        calibration={
            "epsilon": epsilon,
            "rho": config.rho,
            "epsilon_scale": config.epsilon_scale,
            "epsilon_override": config.epsilon_override,
            "source": config.epsilon_source,
            "train_probe": calibration,
            "eval_probe": calibration_eval,
            "alternative_source": calibration_other,
            "attainability_gap": (
                calibration_other["c_real_mean"] / max(calibration["c_real_mean"], 1e-30)
            ),
        },
        summary={
            **optimizer.summary(),
            # A run that never reaches OPTIMISE has no certificate -- Theorem
            # 1(c) reads it off an OPTIMISE iterate -- and nothing holding the
            # JEPA loss down.  Measured across six base models, the only two
            # with zero OPTIMISE steps are the only two whose loss blew up, by
            # 6.3x and 92.9x; every run with at least 292 of them ended at or
            # below its starting loss.  The requirement is predictable before
            # training from C_0/eps, which correlates -0.88 with log eps: a
            # predictor that is *better* factually gets a tighter tolerance and
            # so starts relatively further outside it.
            # ``pretrained`` takes no steps at all, so "no OPTIMISE step" is
            # vacuously true there and would read as the failure this flags.
            "starved_of_optimise_steps": (
                optimizer.state.step > 0 and optimizer.state.optimise_steps == 0
            ),
            "constraint_at_start_over_epsilon": (
                history[0]["constraint"] / epsilon
                if history and history[0].get("constraint") == history[0].get("constraint")
                else None
            ),
        },
        history=history,
        probes={
            "train_probe_c_real": calibration["c_real_mean"],
            "eval_probe_c_real": calibration_eval["c_real_mean"],
        },
        diagnostics=diagnostics,
    )


def save_result(result: PostTrainResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
