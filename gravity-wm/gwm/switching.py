"""The switching gradient method of Algorithm 1, and its KKT certificates.

For ``min L(theta)`` subject to ``C(theta) <= eps`` the method steps along
``-grad C`` while the constraint is violated (RESTORE) and along ``-grad L``
otherwise (OPTIMISE), removing from ``grad L`` the component that conflicts
with ``grad C`` when the constraint is close to active.

Three choices here differ from the method note, each for a reason the theory
note itself gives.

The projection gate is ``gamma = 0.9``, not ``0.5``.
    Remark 1 recommends ``(gamma - zeta_C)(1 + rho) > 1`` so that the
    constraint qualification is only needed above the real-data violation
    level; with ``rho = 0.25`` and ``zeta_C <= 0.05`` that means
    ``gamma > 0.85``.  The note's ``0.5`` projects long before the constraint
    is active, and Remark 6 and Proposition 1 of the theory note show that
    early projection can stop learning at a strictly feasible point with a
    complementarity residual of at least ``(1 - gamma)/7``.

The update is plain SGD.
    Theorem 1 analyses ``theta <- theta - eta d`` exactly.  An adaptive
    preconditioner would rescale ``d`` coordinate-wise, which destroys both the
    descent inequalities of Lemma 5 and the meaning of the ``1/eps`` scaling of
    the RESTORE direction, and would make ``||d_n||`` useless as the
    stationarity certificate of Theorem 1(c).  :class:`SwitchingOptimizer`
    therefore applies the step itself.

RESTORE uses a Polyak step.
    The note's RESTORE direction is ``grad C_B / eps``.  Because ``eps`` is
    calibrated to probe noise it can be small, and that direction
    correspondingly huge, so the note permits RESTORE its own step size
    explicitly.  A *fixed* step size is still fragile: ``||grad C||`` grows as
    the constraint gets harder, so a step calibrated at the first RESTORE
    iteration becomes too long later, and RESTORE begins increasing the very
    quantity it is meant to reduce while the JEPA loss blows up -- which is
    what happens here in practice.  The default is therefore Polyak's step for
    reaching the level set ``{C = 0.8 eps}``,

        ``d = grad C_B (C_B - 0.8 eps) / ||grad C_B||^2`` with step size one,

    whose length adapts to the gradient norm by construction.  This is the step
    the switching-subgradient analysis the algorithm follows uses on constraint
    steps.  ``restore_direction="eps"`` recovers the note's literal direction
    and ``"normalized"`` a fixed-length one; both are kept as ablations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

__all__ = [
    "SwitchingConfig",
    "SwitchingOptimizer",
    "SwitchingState",
    "StepRecord",
    "flat_gradient",
]

RESTORE = "restore"
OPTIMISE = "optimise"


def flat_gradient(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    """Concatenate ``.grad`` over ``parameters``, treating missing grads as zero."""

    chunks = []
    for parameter in parameters:
        if parameter.grad is None:
            chunks.append(parameter.new_zeros(parameter.numel()))
        else:
            chunks.append(parameter.grad.detach().reshape(-1))
    if not chunks:
        raise ValueError("no trainable parameters")
    return torch.cat(chunks)


@dataclass(frozen=True)
class SwitchingConfig:
    """Hyper-parameters of Algorithm 1.

    ``gate`` is ``gamma``; ``exit_factor`` is the hysteresis exit level as a
    multiple of ``eps``.  ``restore_lr`` defaults to ``None``, meaning "match
    the typical OPTIMISE step length", which :class:`SwitchingOptimizer`
    calibrates once from the observed gradient norms.
    """

    lr: float = 1.0e-3
    restore_lr: float | None = None
    gate: float = 0.9
    exit_factor: float = 0.9
    ema_decay: float = 0.9
    momentum: float = 0.0
    use_ema_for_mode: bool = True
    hysteresis: bool = True
    project: bool = True
    grad_clip: float | None = 1.0e4
    restore_direction: str = "polyak"
    restore_target_factor: float = 0.8
    # ``grad_clip`` is a safety net against a blown-up step, not part of the
    # analysed algorithm, so it defaults high enough to be inactive.  A clip
    # that engages on every RESTORE step would silently turn the analysed
    # direction ``grad C / eps`` into a normalised one, so the fraction of
    # clipped steps is reported in :meth:`SwitchingOptimizer.summary`.

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.restore_lr is not None and self.restore_lr <= 0.0:
            raise ValueError("restore_lr must be positive when given")
        if not 0.0 < self.gate < 1.0:
            raise ValueError("gate (gamma) must lie strictly in (0, 1)")
        if not 0.0 < self.exit_factor <= 1.0:
            raise ValueError("exit_factor must lie in (0, 1]")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if self.grad_clip is not None and self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive when given")
        if self.restore_direction not in {"eps", "normalized", "polyak"}:
            raise ValueError("restore_direction must be 'eps', 'normalized' or 'polyak'")
        if not 0.0 < self.restore_target_factor <= 1.0:
            raise ValueError("restore_target_factor must lie in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepRecord:
    """What one iteration did, enough to audit the switching rule afterwards."""

    step: int
    mode: str
    projected: bool
    exceptional: bool
    jepa_loss: float
    constraint: float
    constraint_ema: float
    epsilon: float
    direction_norm: float
    grad_l_norm: float
    grad_c_norm: float
    inner_product: float
    multiplier: float
    step_size: float
    lagrangian_residual: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SwitchingState:
    step: int = 0
    ema: float = float("nan")
    mode: str = OPTIMISE
    restore_lr: float | None = None
    momentum_buffer: torch.Tensor | None = None
    restore_steps: int = 0
    optimise_steps: int = 0
    projected_steps: int = 0
    exceptional_steps: int = 0
    best_direction_norm: float = float("inf")
    best_step: int = -1
    clipped_steps: int = 0
    history: list[StepRecord] = field(default_factory=list)


class SwitchingOptimizer:
    """Algorithm 1 over a fixed list of trainable parameters.

    The caller supplies, per iteration, the two minibatch gradients as flat
    vectors together with the minibatch loss and violation; this class owns the
    mode, the exponential moving average, the projection and the parameter
    update.  Keeping the gradient computation outside makes it possible to skip
    the constraint backward pass when the theory says it is not needed
    (``Chat <= gamma*eps`` and no projection can trigger).
    """

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        *,
        epsilon: float,
        config: SwitchingConfig,
    ) -> None:
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive; calibrate it from c_real first")
        self.parameters = [p for p in parameters if p.requires_grad]
        if not self.parameters:
            raise ValueError("no trainable parameters were passed to the optimizer")
        self.epsilon = float(epsilon)
        self.config = config
        self.state = SwitchingState(restore_lr=config.restore_lr)
        self._last_constraint = float("nan")
        self.dimension = sum(p.numel() for p in self.parameters)

    # -- mode selection -------------------------------------------------

    _last_constraint: float = float("nan")

    def observe_violation(self, violation: float) -> float:
        """Fold ``C_B`` into the estimate ``Chat`` used to select the mode.

        A single minibatch ``C_B`` is too noisy to switch on, so the mode is
        selected by an exponential moving average.  Remark 13 bounds the
        resulting relative error ``zeta_C`` by a noise term plus a lag
        ``2 eta M^2 beta / (1 - beta)``; a larger ``beta`` trades noise for lag.
        """

        value = float(violation)
        if not math.isfinite(value):
            raise FloatingPointError("constraint value is not finite")
        self._last_constraint = value
        if not self.config.use_ema_for_mode:
            self.state.ema = value
        elif math.isnan(self.state.ema):
            self.state.ema = value
        else:
            decay = self.config.ema_decay
            self.state.ema = decay * self.state.ema + (1.0 - decay) * value
        return self.state.ema

    def prime(self, violations: list[float]) -> float:
        """Initialise ``Chat`` from a few minibatches and pick the first mode."""

        if not violations:
            raise ValueError("priming needs at least one minibatch violation")
        self.state.ema = float(sum(violations) / len(violations))
        self._last_constraint = violations[-1]
        self.state.mode = RESTORE if self.state.ema > self.epsilon else OPTIMISE
        return self.state.ema

    def select_mode(self) -> str:
        """Apply the switching rule, with hysteresis if enabled.

        Switch to RESTORE when ``Chat > eps``, and back to OPTIMISE only once
        ``Chat < exit_factor * eps``.  Theorem 1 analyses the single-threshold
        rule; hysteresis only prevents switching at every step and, as Remark 1
        notes, does not improve the bound.
        """

        estimate = self.state.ema
        if math.isnan(estimate):
            raise RuntimeError("call prime() or observe_violation() before select_mode()")
        if not self.config.hysteresis:
            self.state.mode = RESTORE if estimate > self.epsilon else OPTIMISE
            return self.state.mode
        if estimate > self.epsilon:
            self.state.mode = RESTORE
        elif estimate < self.config.exit_factor * self.epsilon:
            self.state.mode = OPTIMISE
        return self.state.mode

    def needs_constraint_gradient(self) -> bool:
        """Whether this iteration needs ``grad C_B`` at all.

        In RESTORE it is the direction.  In OPTIMISE it is only needed when the
        projection can trigger, i.e. when ``Chat > gamma * eps``; below that
        the second backward pass can be skipped, as the method note observes.
        """

        if self.state.mode == RESTORE:
            return True
        return self.state.ema > self.config.gate * self.epsilon

    # -- the update -----------------------------------------------------

    def direction(
        self, grad_l: torch.Tensor | None, grad_c: torch.Tensor | None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Build ``d_n`` and report how it was built."""

        info: dict[str, Any] = {
            "projected": False,
            "exceptional": False,
            "multiplier": 0.0,
            "inner_product": 0.0,
            "grad_l_norm": float("nan"),
            "grad_c_norm": float("nan"),
        }
        if grad_l is not None:
            info["grad_l_norm"] = float(grad_l.norm())
        if grad_c is not None:
            info["grad_c_norm"] = float(grad_c.norm())
        if self.state.mode == RESTORE:
            if grad_c is None:
                raise ValueError("RESTORE needs the constraint gradient")
            norm = max(float(grad_c.norm()), 1.0e-12)
            if self.config.restore_direction == "normalized":
                return grad_c / norm, info
            if self.config.restore_direction == "polyak":
                # Polyak's step for reaching the level set {C = target}: with
                # ``d = grad C (C_B - target) / ||grad C||^2`` and step size
                # one, the linearisation of ``C`` at the new point equals the
                # target exactly.  Its length adapts to ||grad C||, which the
                # ``grad C / eps`` direction with a fixed step size does not:
                # that norm grows as the constraint gets harder, so a step
                # calibrated once becomes too long later and RESTORE starts
                # increasing the very quantity it is meant to reduce.  The
                # switching-subgradient analysis this algorithm follows uses
                # Polyak step sizes on constraint steps for the same reason.
                target = self.config.restore_target_factor * self.epsilon
                gap = max(self._last_constraint - target, 0.0)
                info["polyak_gap"] = gap
                return grad_c * (gap / norm**2), info
            return grad_c / self.epsilon, info

        if grad_l is None:
            raise ValueError("OPTIMISE needs the JEPA gradient")
        direction = grad_l
        if grad_c is None:
            return direction, info
        inner = float(torch.dot(grad_c, grad_l))
        info["inner_product"] = inner
        if inner >= 0.0:
            return direction, info
        gated = self.state.ema > self.config.gate * self.epsilon
        if not (self.config.project and gated):
            # A conflicting OPTIMISE iteration that is not projected is what
            # Lemma 3(iv) calls exceptional; it is the only case in which
            # <grad C, d> may be negative.
            info["exceptional"] = True
            return direction, info
        squared = float(grad_c.pow(2).sum())
        if squared <= 0.0:
            info["exceptional"] = True
            return direction, info
        multiplier = -inner / squared
        info["projected"] = True
        info["multiplier"] = multiplier
        return grad_l + multiplier * grad_c, info

    def _step_size(self, grad_l_norm: float, grad_c_norm: float) -> float:
        if self.state.mode != RESTORE:
            return self.config.lr
        if self.config.restore_direction == "polyak":
            # The direction already carries the step length.
            return 1.0
        if self.state.restore_lr is not None:
            return self.state.restore_lr
        # Calibrate once: make the first RESTORE step the same length as a
        # typical OPTIMISE step, so that ``eta_R * ||grad C / eps||`` is
        # comparable to ``eta * ||grad L||`` instead of larger by the factor
        # ``1/eps``.
        reference = grad_l_norm if math.isfinite(grad_l_norm) and grad_l_norm > 0.0 else 1.0
        scaled = max(grad_c_norm / self.epsilon, 1.0e-12)
        self.state.restore_lr = float(self.config.lr * reference / scaled)
        return self.state.restore_lr

    @torch.no_grad()
    def step(
        self,
        *,
        grad_l: torch.Tensor | None,
        grad_c: torch.Tensor | None,
        jepa_loss: float,
        constraint: float,
    ) -> StepRecord:
        """Take one iteration of Algorithm 1 and record it."""

        direction, info = self.direction(grad_l, grad_c)
        if not torch.isfinite(direction).all():
            raise FloatingPointError(f"non-finite update direction at step {self.state.step}")
        norm = float(direction.norm())
        if self.config.grad_clip is not None and norm > self.config.grad_clip:
            direction = direction * (self.config.grad_clip / norm)
            norm = self.config.grad_clip
            self.state.clipped_steps += 1
        grad_l_norm = info["grad_l_norm"]
        grad_c_norm = info["grad_c_norm"]
        step_size = self._step_size(grad_l_norm, grad_c_norm)

        if self.config.momentum > 0.0:
            if self.state.momentum_buffer is None:
                self.state.momentum_buffer = torch.zeros_like(direction)
            self.state.momentum_buffer.mul_(self.config.momentum).add_(direction)
            applied = self.state.momentum_buffer
        else:
            applied = direction

        offset = 0
        for parameter in self.parameters:
            count = parameter.numel()
            parameter.add_(applied[offset : offset + count].view_as(parameter), alpha=-step_size)
            offset += count

        # Theorem 1(c): on an OPTIMISE iteration ``d_n`` is exactly the
        # minibatch Lagrangian gradient with a non-negative multiplier, so
        # ``||d_n||`` is the stationarity certificate and the returned iterate
        # is the OPTIMISE step with the smallest one.
        residual = norm if self.state.mode == OPTIMISE else float("nan")
        record = StepRecord(
            step=self.state.step,
            mode=self.state.mode,
            projected=bool(info["projected"]),
            exceptional=bool(info["exceptional"]),
            jepa_loss=float(jepa_loss),
            constraint=float(constraint),
            constraint_ema=float(self.state.ema),
            epsilon=self.epsilon,
            direction_norm=norm,
            grad_l_norm=float(grad_l_norm),
            grad_c_norm=float(grad_c_norm),
            inner_product=float(info["inner_product"]),
            multiplier=float(info["multiplier"]),
            step_size=float(step_size),
            lagrangian_residual=float(residual),
        )
        if self.state.mode == RESTORE:
            self.state.restore_steps += 1
        else:
            self.state.optimise_steps += 1
            if info["projected"]:
                self.state.projected_steps += 1
            if info["exceptional"]:
                self.state.exceptional_steps += 1
            if norm < self.state.best_direction_norm:
                self.state.best_direction_norm = norm
                self.state.best_step = self.state.step
        self.state.history.append(record)
        self.state.step += 1
        return record

    def summary(self) -> dict[str, Any]:
        total = max(1, self.state.step)
        return {
            "steps": self.state.step,
            "epsilon": self.epsilon,
            "restore_steps": self.state.restore_steps,
            "optimise_steps": self.state.optimise_steps,
            "projected_steps": self.state.projected_steps,
            "exceptional_steps": self.state.exceptional_steps,
            "restore_fraction": self.state.restore_steps / total,
            "projected_fraction": (
                self.state.projected_steps / max(1, self.state.optimise_steps)
            ),
            "final_constraint_ema": self.state.ema,
            "best_direction_norm": self.state.best_direction_norm,
            "best_step": self.state.best_step,
            "restore_lr": self.state.restore_lr,
            "clipped_steps": self.state.clipped_steps,
            "clipped_fraction": self.state.clipped_steps / total,
            "config": self.config.to_dict(),
            "parameter_dimension": self.dimension,
        }
