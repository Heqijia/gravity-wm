"""The free-fall violation of Definition 1, and the rollouts it is measured on.

The constraint is the only place physics enters post-training.  Given a real
window with latents ``z_{<=t}`` and gravity ``g``, the predictor is rolled out
under a counterfactual gravity ``g'`` for which no video exists, object
positions are decoded from the predicted latents with a frozen probe, and their
finite-difference accelerations are compared with the targets that Lemma 1
prescribes for a trajectory whose gravity switches from ``g`` to ``g'`` at
step ``t``.

Two deliberate departures from the method note:

``c`` is squared.
    The note averages ``||a - gbar||``; the theory note averages
    ``||a - gbar||^2`` and says why: the unsquared norm is not differentiable
    where ``a = gbar``, so the smoothness of Assumption 1 cannot hold for it.
    :func:`free_fall_violation` therefore squares, and ``epsilon``,
    ``E[c_real]`` and ``C`` are all in squared units.

The finite difference may use a stride.
    Lemma 1 holds for any uniform spacing, so the second difference may be
    taken over a stride of ``m`` frames with interval ``m*dt``.  This matters
    in practice: the quantity the constraint has to resolve is the one-step
    gravity displacement ``g*dt^2``, which a stride magnifies by ``m^2``.  At
    the sampling rate of the Semigroup-JEPA planar datasets that displacement
    is 0.19 pixel, far below any achievable probe accuracy; a stride turns the
    same data into a measurable signal.  :func:`free_fall_targets` derives the
    exact target for every term at any stride, so no term is approximated and
    ``m = 1`` reproduces Lemma 1 verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

__all__ = [
    "CONSTRAINT_FORMS",
    "ConstraintSpec",
    "FreeFallBatch",
    "acceleration_vector",
    "condition_on_acceleration",
    "condition_on_gravity",
    "counterfactual_rollout",
    "decode_positions",
    "free_fall_targets",
    "free_fall_position_violation",
    "free_fall_violation",
    "gravity_blind_violation",
    "implied_velocity",
    "real_violation",
    "rollout_latents",
]


@dataclass(frozen=True)
class ConstraintSpec:
    """Shape of the free-fall constraint.

    ``horizon`` is the number of predicted steps ``H``; ``stride`` is the
    finite-difference spacing ``m`` in frames.  Accelerations are evaluated at
    every offset ``j = tau - t`` in ``[1 - m, H - m]``, which is the largest
    set for which both ``tau - m`` is available (within the context) and
    ``tau + m`` is predicted, and which ties every one of the ``H`` predicted
    positions to the real context -- so the identifiability of Proposition 1
    carries over to any stride.
    """

    horizon: int = 6
    stride: int = 3
    history: int = 8
    form: str = "acceleration"

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.stride < 1 or self.history < 1:
            raise ValueError("horizon, stride and history must be positive")
        if self.horizon < self.stride:
            raise ValueError(
                f"horizon={self.horizon} must be at least stride={self.stride}, "
                "otherwise no acceleration term involves a prediction"
            )
        if self.form not in ("acceleration", "position"):
            raise ValueError("form must be 'acceleration' or 'position'")
        if self.history < 2 * self.stride:
            raise ValueError(
                f"history={self.history} must be at least 2*stride={2 * self.stride} so that "
                "the earliest acceleration term has both of its context positions"
            )

    @property
    def offsets(self) -> tuple[int, ...]:
        """Offsets ``j = tau - t`` at which an acceleration is evaluated."""

        return tuple(range(1 - self.stride, self.horizon - self.stride + 1))

    @property
    def num_terms(self) -> int:
        return len(self.offsets)


def free_fall_targets(
    offsets: tuple[int, ...] | list[int],
    gravity: torch.Tensor,
    counterfactual: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Finite-difference acceleration of the exact switched trajectory.

    Lemma 1 writes the trajectory that moves under ``g`` before step ``t`` and
    under ``g'`` after it as a single expression,

    ``p_{t+k} = p_t + v k dt + lambda(k) k^2 dt^2 / 2``,

    with ``lambda(k) = g'`` for ``k >= 0`` and ``g`` for ``k < 0`` -- the two
    cases agree at ``k = 0``.  Substituting it into the strided second
    difference at ``tau = t + j`` makes the ``p_t`` and velocity terms cancel
    identically, independently of ``v``, and leaves

    ``abar_{t+j} = [lambda(j+m)(j+m)^2 - 2 lambda(j) j^2
                    + lambda(j-m)(j-m)^2] / (2 m^2)``.

    For ``m = 1`` this is exactly Lemma 1: ``(g + g')/2`` at ``j = 0`` and
    ``g'`` for ``j >= 1``.  For any ``m`` it reduces to ``g'`` once
    ``j >= m`` (all three points lie after the switch) and to ``g`` once
    ``j <= -m``, and interpolates in between.

    ``gravity`` and ``counterfactual`` are gravity *vectors* of shape
    ``(B, 2)``.  Returns ``(B, len(offsets), 2)``.
    """

    if gravity.shape != counterfactual.shape or gravity.ndim != 2 or gravity.shape[-1] != 2:
        raise ValueError(
            "gravity and counterfactual must both have shape (B, 2); got "
            f"{tuple(gravity.shape)} and {tuple(counterfactual.shape)}"
        )
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be positive")

    def law(offset: int) -> torch.Tensor:
        return counterfactual if offset >= 0 else gravity

    rows = []
    for offset in offsets:
        value = (
            law(offset + stride) * float((offset + stride) ** 2)
            - 2.0 * law(offset) * float(offset**2)
            + law(offset - stride) * float((offset - stride) ** 2)
        ) / (2.0 * float(stride) ** 2)
        rows.append(value)
    return torch.stack(rows, dim=1)


def gravity_blind_violation(
    spec: ConstraintSpec,
    gravity: torch.Tensor,
    counterfactual: torch.Tensor,
) -> torch.Tensor:
    """``c`` of a predictor that ignores the conditioning gravity.

    Such a predictor continues the motion under the episode's own gravity, so
    by Lemma 1 every one of its accelerations equals ``g`` while the targets
    interpolate towards ``g'``.  The resulting violation is a closed-form
    function of the gravity shift, the stride and the offsets -- no model is
    involved -- which makes it the natural reference scale for the tolerance:
    the constraint can only teach the predictor about ``g'`` if this quantity
    is well above ``epsilon``.  It is also the mechanism behind Theorem 2's
    separation, where the counterfactual error of a gravity-blind predictor is
    bounded below by a quantity growing with the gravity shift.
    """

    targets = free_fall_targets(spec.offsets, gravity, counterfactual, spec.stride)
    blind = gravity.unsqueeze(1).expand_as(targets)
    return (blind - targets).pow(2).sum(dim=-1).mean(dim=-1)


def acceleration_vector(
    gravity: torch.Tensor, accel_x: torch.Tensor | None = None
) -> torch.Tensor:
    """The planar acceleration ``(a_x, -g)`` the constraint is stated in.

    Everything downstream of here -- Lemma 1's targets, the violation in either
    form, Proposition 1's exact path -- is already a function of a planar
    *vector*, because the second difference of a two-dimensional position is.
    Only this conversion and the action conditioning know that the vector is
    usually ``(0, -g)``: free fall under the simulator's ``(0, 0, -g)``.

    With ``accel_x`` supplied the counterfactual becomes a vector rather than a
    scalar, which is the two-dimensional experiment: the model is conditioned
    on an acceleration direction it never saw, not merely a magnitude.  No
    other part of the method changes.
    """

    scalar = torch.as_tensor(gravity)
    if scalar.ndim != 1:
        raise ValueError(f"expected a batch of scalars, got shape {tuple(scalar.shape)}")
    horizontal = (
        torch.zeros_like(scalar) if accel_x is None else torch.as_tensor(accel_x).to(scalar)
    )
    if horizontal.shape != scalar.shape:
        raise ValueError(
            f"accel_x must match gravity's shape {tuple(scalar.shape)}; "
            f"got {tuple(horizontal.shape)}"
        )
    return torch.stack((horizontal, -scalar), dim=-1)


def gravity_vector(gravity: torch.Tensor) -> torch.Tensor:
    """``acceleration_vector`` with no horizontal component: ``(0, -g)``."""

    return acceleration_vector(gravity, None)


def condition_on_gravity(
    action: torch.Tensor,
    gravity: torch.Tensor,
    *,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    gravity_index: int = 2,
) -> torch.Tensor:
    """Replace the gravity coordinate of a normalized action sequence.

    Gravity reaches the predictor as the last coordinate of the action, after
    the dataset's train-split z-scoring; conditioning on a different gravity
    means writing the normalized value of that gravity into every step of the
    window, because gravity is constant within an episode and the predictor was
    trained on constant sequences.  Nothing else about the action changes, so
    the counterfactual branch differs from the factual one *only* in gravity.
    """

    if action.ndim != 3:
        raise ValueError(f"action must have shape (B, T, A); got {tuple(action.shape)}")
    if gravity.ndim != 1 or gravity.shape[0] != action.shape[0]:
        raise ValueError("gravity must be a scalar per batch element")
    normalized = (gravity - action_mean[gravity_index]) / action_std[gravity_index]
    out = action.clone()
    out[:, :, gravity_index] = normalized.to(out.dtype).unsqueeze(-1)
    return out


def condition_on_acceleration(
    action: torch.Tensor,
    *,
    gravity: torch.Tensor,
    accel_x: torch.Tensor | None = None,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    gravity_index: int = 2,
    accel_x_index: int | None = None,
) -> torch.Tensor:
    """``condition_on_gravity`` for a counterfactual acceleration *vector*.

    Writes the normalised gravity into ``gravity_index`` exactly as before, and
    the normalised horizontal acceleration into ``accel_x_index`` when the
    dataset carries one.  With ``accel_x`` or ``accel_x_index`` absent this is
    ``condition_on_gravity``, element for element, so the scalar experiments are
    unaffected by the generalisation.
    """

    out = condition_on_gravity(
        action,
        gravity,
        action_mean=action_mean,
        action_std=action_std,
        gravity_index=gravity_index,
    )
    if accel_x is None or accel_x_index is None:
        return out
    horizontal = torch.as_tensor(accel_x)
    if horizontal.ndim != 1 or horizontal.shape[0] != action.shape[0]:
        raise ValueError("accel_x must be a scalar per batch element")
    normalized = (horizontal - action_mean[accel_x_index]) / action_std[accel_x_index]
    out[:, :, accel_x_index] = normalized.to(out.dtype).unsqueeze(-1)
    return out


def rollout_latents(
    model: nn.Module,
    latents: torch.Tensor,
    condition: torch.Tensor,
    *,
    history: int,
    horizon: int,
) -> torch.Tensor:
    """Autoregressive latent rollout, shape ``(B, horizon, D)``.

    Uses the sliding-window contract of ``sg_jepa.losses.rollout``: start from
    the first ``history`` real latents, predict one step, append it, and slide.
    """

    if latents.ndim != 3 or condition.ndim != 3:
        raise ValueError("latents and condition must have shape (B, T, D)")
    if latents.shape[1] < history:
        raise ValueError(
            f"need at least history={history} context latents, got {latents.shape[1]}"
        )
    if condition.shape[1] < history + horizon:
        raise ValueError(
            f"need history+horizon={history + horizon} condition steps, got {condition.shape[1]}"
        )
    rolled = latents[:, :history]
    predictions = []
    for step in range(int(horizon)):
        window = rolled[:, -history:]
        window_condition = condition[:, step : step + history]
        nxt = model.predict(window, window_condition)[:, -1:]
        predictions.append(nxt)
        rolled = torch.cat([rolled, nxt], dim=1)
    return torch.cat(predictions, dim=1)


def decode_positions(probe: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """Decode object positions from latents, keeping the graph.

    The probe is frozen but must stay differentiable with respect to its input:
    the constraint gradient reaches the predictor along
    ``c -> a -> h -> zhat -> P_theta``.  ``FrozenStateProbe.predict_features``
    runs under ``torch.inference_mode`` and would sever that path, so the probe
    module is called directly here.
    """

    if latents.ndim != 3:
        raise ValueError(f"latents must have shape (B, T, D); got {tuple(latents.shape)}")
    batch, steps, _ = latents.shape
    flat = probe(latents.reshape(batch * steps, -1))
    return flat.reshape(batch, steps, -1)


@dataclass
class FreeFallBatch:
    """Everything one constraint evaluation produced, for logging and evaluation."""

    violation: torch.Tensor
    per_sample: torch.Tensor
    accelerations: torch.Tensor
    targets: torch.Tensor
    positions: torch.Tensor
    predicted_latents: torch.Tensor

    def detached(self) -> dict[str, torch.Tensor]:
        return {
            "violation": self.violation.detach(),
            "per_sample": self.per_sample.detach(),
            "accelerations": self.accelerations.detach(),
            "targets": self.targets.detach(),
            "positions": self.positions.detach(),
        }


def implied_velocity(
    previous: torch.Tensor, current: torch.Tensor, gravity: torch.Tensor, dt: float
) -> torch.Tensor:
    """Lemma 1's velocity at step ``t`` from two decoded context positions.

    ``p_{t-1} = p_t - v dt + g dt^2 / 2`` gives ``v = (p_t - p_{t-1})/dt + g dt/2``.
    Uses only positions decoded from *real* latents, so it needs no future frame.
    """

    return (current - previous) / float(dt) + 0.5 * gravity * float(dt)


def free_fall_position_violation(
    positions: torch.Tensor,
    *,
    gravity: torch.Tensor,
    counterfactual: torch.Tensor,
    spec: ConstraintSpec,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Proposition 1's violation: distance to the exact counterfactual path.

    Definition 1 constrains the *second differences* of the decoded positions.
    Proposition 1 says that driving that to zero puts the decoded positions on
    the counterfactual trajectory through the decoded context, so the two have
    the same zero set -- but not the same optimisation landscape, and the
    difference matters.

    By Lemma 7 the position error at step ``s`` is a *signed, time-weighted*
    sum of the acceleration residuals, ``dt^2 sum_j (s-j) (a_j - gbar_j)``,
    whereas Definition 1 averages their *squares*.  Minimising the mean square
    removes whatever part of the residual is unsystematic and leaves the part
    that is -- measured here, driving the violation from 57.7 down to 19.2 cut
    the residual's RMS almost in half while its signed mean barely moved, and
    the terminal position error got *worse*, from 11.2 to 15.2 pixels.  The
    squared-residual constraint is blind to exactly the structure that decides
    position accuracy.

    This form constrains the thing Corollary 2 actually bounds.  It still uses
    no future frames: the trajectory is built from the two decoded *context*
    positions and the two gravities, which is all Proposition 1 needs.
    """

    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(f"positions must have shape (B, T, 2); got {tuple(positions.shape)}")
    total = spec.history + spec.horizon
    if positions.shape[1] != total:
        raise ValueError(
            f"expected history+horizon={total} decoded positions, got {positions.shape[1]}"
        )
    if spec.history < 2:
        raise ValueError("the implied velocity needs two context positions")
    pivot = spec.history - 1
    anchor = positions[:, pivot]
    velocity = implied_velocity(positions[:, pivot - 1], anchor, gravity, dt)
    steps = torch.arange(
        1, spec.horizon + 1, device=positions.device, dtype=positions.dtype
    ) * float(dt)
    target = (
        anchor.unsqueeze(1)
        + velocity.unsqueeze(1) * steps.reshape(1, -1, 1)
        + 0.5 * counterfactual.unsqueeze(1) * steps.reshape(1, -1, 1) ** 2
    )
    predicted = positions[:, spec.history :]
    per_sample = (predicted - target).pow(2).sum(dim=-1).mean(dim=-1)
    return per_sample, predicted, target


CONSTRAINT_FORMS = ("acceleration", "position")


def free_fall_violation(
    positions: torch.Tensor,
    *,
    gravity: torch.Tensor,
    counterfactual: torch.Tensor,
    spec: ConstraintSpec,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Free-fall violation, in whichever form ``spec.form`` selects.

    ``acceleration`` is Definition 1 verbatim; ``position`` is Proposition 1's
    equivalent-zero-set form, which constrains the quantity Corollary 2 bounds.
    See :func:`free_fall_position_violation` for why they behave differently.

    Squared free-fall violation of Definition 1.

    ``positions`` has shape ``(B, history + horizon, 2)``: the first
    ``history`` entries are decoded from real latents and the rest from the
    rollout.  Index ``history - 1`` is step ``t``.  Returns the per-sample
    violation ``c``, the accelerations and their targets.
    """

    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(f"positions must have shape (B, T, 2); got {tuple(positions.shape)}")
    total = spec.history + spec.horizon
    if positions.shape[1] != total:
        raise ValueError(
            f"expected history+horizon={total} decoded positions, got {positions.shape[1]}"
        )
    if spec.form == "position":
        return free_fall_position_violation(
            positions, gravity=gravity, counterfactual=counterfactual, spec=spec, dt=dt
        )
    pivot = spec.history - 1
    interval = float(dt) * float(spec.stride)
    centre, plus, minus = [], [], []
    for offset in spec.offsets:
        index = pivot + offset
        if index - spec.stride < 0 or index + spec.stride >= total:
            raise ValueError(f"offset {offset} leaves the decoded window")
        centre.append(index)
        plus.append(index + spec.stride)
        minus.append(index - spec.stride)
    accelerations = (
        positions[:, plus] - 2.0 * positions[:, centre] + positions[:, minus]
    ) / interval**2
    targets = free_fall_targets(spec.offsets, gravity, counterfactual, spec.stride)
    per_sample = (accelerations - targets).pow(2).sum(dim=-1).mean(dim=-1)
    return per_sample, accelerations, targets




def counterfactual_rollout(
    model: nn.Module,
    probe: nn.Module,
    batch: dict[str, Any],
    *,
    spec: ConstraintSpec,
    dt: float,
    counterfactual_gravity: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> FreeFallBatch:
    """Roll the predictor out under ``counterfactual_gravity`` and score it.

    The context latents are real, so the predictor and the probe both receive
    in-distribution inputs at the start of the rollout; only the conditioning
    gravity changes.  No future frames are used anywhere in this function,
    which is what lets the constraint be evaluated at gravities that have no
    video.
    """

    latents = batch["latents"]
    action = condition_on_gravity(
        batch["action"],
        counterfactual_gravity,
        action_mean=action_mean,
        action_std=action_std,
    )
    condition = model.action_encoder(action)
    predicted = rollout_latents(
        model,
        latents,
        condition,
        history=spec.history,
        horizon=spec.horizon,
    )
    sequence = torch.cat([latents[:, : spec.history], predicted], dim=1)
    positions = decode_positions(probe, sequence)
    per_sample, accelerations, targets = free_fall_violation(
        positions,
        gravity=gravity_vector(batch["gravity"]),
        counterfactual=gravity_vector(counterfactual_gravity),
        spec=spec,
        dt=dt,
    )
    return FreeFallBatch(
        violation=per_sample.mean(),
        per_sample=per_sample,
        accelerations=accelerations,
        targets=targets,
        positions=positions,
        predicted_latents=predicted,
    )


def real_violation(
    probe: nn.Module,
    batch: dict[str, Any],
    *,
    spec: ConstraintSpec,
    dt: float,
) -> FreeFallBatch:
    """``c_real``: the violation of real motion, which calibrates the tolerance.

    Every position is decoded from a real latent and the conditioning gravity
    equals the episode gravity, so the accelerations are those of the true
    trajectory and the residual is pure probe noise.  ``epsilon`` is set to
    ``(1 + rho) E[c_real]`` so that the true dynamics satisfy the constraint in
    expectation and enforcing it does not bias the predictor away from them.
    """

    latents = batch["latents"]
    total = spec.history + spec.horizon
    if latents.shape[1] < total:
        raise ValueError(
            f"c_real needs history+horizon={total} real latents, got {latents.shape[1]}"
        )
    positions = decode_positions(probe, latents[:, :total])
    vectors = gravity_vector(batch["gravity"])
    per_sample, accelerations, targets = free_fall_violation(
        positions,
        gravity=vectors,
        counterfactual=vectors,
        spec=spec,
        dt=dt,
    )
    return FreeFallBatch(
        violation=per_sample.mean(),
        per_sample=per_sample,
        accelerations=accelerations,
        targets=targets,
        positions=positions,
        predicted_latents=latents[:, spec.history : total],
    )
