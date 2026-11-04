"""Algorithm 1 must do what Theorem 1 says, on problems with known answers."""

from __future__ import annotations

import math

import pytest
import torch

from gwm.switching import OPTIMISE, RESTORE, SwitchingConfig, SwitchingOptimizer, flat_gradient


def quadratic_problem(dimension=3, target=3.0):
    """``min 0.5||theta - a||^2`` s.t. ``C(theta) = ||theta||^2 <= eps``.

    The solution is the projection of ``a`` onto the feasible ball,
    ``theta* = a * min(1, sqrt(eps)/||a||)``, with Lagrange multiplier
    ``nu = (||a||/sqrt(eps) - 1)/2`` when the constraint binds.  Everything in
    Definition 1 can therefore be checked against a closed form.
    """

    anchor = torch.full((dimension,), float(target))
    theta = torch.nn.Parameter(torch.zeros(dimension) + 0.1)
    return anchor, theta


def gradients(theta, anchor):
    loss = 0.5 * (theta - anchor).pow(2).sum()
    constraint = theta.pow(2).sum()
    grad_l = torch.autograd.grad(loss, theta, retain_graph=True)[0].reshape(-1)
    grad_c = torch.autograd.grad(constraint, theta)[0].reshape(-1)
    return float(loss), float(constraint), grad_l, grad_c


def test_projection_satisfies_proposition_two():
    """``<grad C, d> = 0`` and ``<grad L, d> = ||d||^2 >= 0``."""

    torch.manual_seed(0)
    theta = torch.nn.Parameter(torch.randn(16))
    optimizer = SwitchingOptimizer([theta], epsilon=1.0, config=SwitchingConfig(gate=0.5))
    optimizer.prime([0.9])
    optimizer.select_mode()
    assert optimizer.state.mode == OPTIMISE
    grad_l = torch.randn(16)
    grad_c = torch.randn(16)
    if torch.dot(grad_c, grad_l) > 0:
        grad_l = grad_l - 2.0 * torch.dot(grad_c, grad_l) / grad_c.pow(2).sum() * grad_c
    direction, info = optimizer.direction(grad_l, grad_c)
    assert info["projected"]
    assert abs(float(torch.dot(grad_c, direction))) < 1e-4
    torch.testing.assert_close(
        torch.dot(grad_l, direction), direction.pow(2).sum(), atol=1e-4, rtol=1e-4
    )
    assert float(torch.dot(grad_l, direction)) >= 0.0


def test_mode_switching_uses_hysteresis():
    theta = torch.nn.Parameter(torch.zeros(2))
    config = SwitchingConfig(gate=0.9, exit_factor=0.9, use_ema_for_mode=False)
    optimizer = SwitchingOptimizer([theta], epsilon=1.0, config=config)
    optimizer.prime([2.0])
    assert optimizer.select_mode() == RESTORE
    # Between the exit level and eps the previous mode is kept.
    optimizer.observe_violation(0.95)
    assert optimizer.select_mode() == RESTORE
    optimizer.observe_violation(0.5)
    assert optimizer.select_mode() == OPTIMISE
    optimizer.observe_violation(0.95)
    assert optimizer.select_mode() == OPTIMISE
    optimizer.observe_violation(1.5)
    assert optimizer.select_mode() == RESTORE


def test_constraint_gradient_is_skipped_below_the_gate():
    theta = torch.nn.Parameter(torch.zeros(2))
    config = SwitchingConfig(gate=0.9, use_ema_for_mode=False)
    optimizer = SwitchingOptimizer([theta], epsilon=1.0, config=config)
    optimizer.prime([0.2])
    optimizer.select_mode()
    assert not optimizer.needs_constraint_gradient()
    optimizer.observe_violation(0.95)
    optimizer.select_mode()
    assert optimizer.needs_constraint_gradient()
    optimizer.observe_violation(5.0)
    assert optimizer.select_mode() == RESTORE
    assert optimizer.needs_constraint_gradient()


def test_ema_tracks_a_constant_violation():
    theta = torch.nn.Parameter(torch.zeros(2))
    optimizer = SwitchingOptimizer(
        [theta], epsilon=1.0, config=SwitchingConfig(ema_decay=0.9)
    )
    optimizer.prime([0.0])
    for _ in range(300):
        optimizer.observe_violation(2.0)
    assert abs(optimizer.state.ema - 2.0) < 1e-3


@pytest.mark.parametrize("epsilon", [0.5, 2.0])
def test_switching_converges_to_the_constrained_optimum(epsilon):
    """The iterates become feasible and approach the closed-form solution."""

    anchor, theta = quadratic_problem(dimension=3, target=3.0)
    norm_anchor = float(anchor.norm())
    optimal = anchor * min(1.0, math.sqrt(epsilon) / norm_anchor)
    config = SwitchingConfig(
        lr=0.02, restore_lr=0.02 * epsilon, gate=0.9, exit_factor=0.9, ema_decay=0.8
    )
    optimizer = SwitchingOptimizer([theta], epsilon=epsilon, config=config)
    _loss, constraint, grad_l, grad_c = gradients(theta, anchor)
    optimizer.prime([constraint])
    for _ in range(4000):
        loss, constraint, grad_l, grad_c = gradients(theta, anchor)
        optimizer.observe_violation(constraint)
        optimizer.select_mode()
        optimizer.step(grad_l=grad_l, grad_c=grad_c, jepa_loss=loss, constraint=constraint)
    _loss, constraint, grad_l, grad_c = gradients(theta, anchor)
    assert constraint <= epsilon * 1.10, f"infeasible: C={constraint} eps={epsilon}"
    assert float((theta.detach() - optimal).norm()) < 0.1 * norm_anchor
    # Complementary slackness: the constraint binds, so the multiplier is
    # positive and ``C`` sits at the tolerance rather than strictly inside.
    assert constraint > 0.5 * epsilon


def test_switching_beats_a_fixed_penalty_at_certifying_feasibility():
    """Remark 10: a fixed ``lambda`` certifies feasibility only if it exceeds
    the problem's multiplier; the switching rule needs no such knowledge."""

    epsilon = 0.25
    anchor, theta = quadratic_problem(dimension=3, target=3.0)
    multiplier = (float(anchor.norm()) / math.sqrt(epsilon) - 1.0) / 2.0
    penalty_lambda = 0.1 * multiplier

    penalty_theta = torch.nn.Parameter(torch.zeros(3) + 0.1)
    for _ in range(6000):
        loss = 0.5 * (penalty_theta - anchor).pow(2).sum()
        constraint = penalty_theta.pow(2).sum()
        total = loss + penalty_lambda * constraint
        grad = torch.autograd.grad(total, penalty_theta)[0]
        with torch.no_grad():
            penalty_theta -= 0.02 * grad
    penalty_violation = float(penalty_theta.detach().pow(2).sum())

    config = SwitchingConfig(lr=0.02, restore_lr=0.02 * epsilon, gate=0.9, ema_decay=0.8)
    optimizer = SwitchingOptimizer([theta], epsilon=epsilon, config=config)
    _loss, constraint, _gl, _gc = gradients(theta, anchor)
    optimizer.prime([constraint])
    for _ in range(4000):
        loss, constraint, grad_l, grad_c = gradients(theta, anchor)
        optimizer.observe_violation(constraint)
        optimizer.select_mode()
        optimizer.step(grad_l=grad_l, grad_c=grad_c, jepa_loss=loss, constraint=constraint)
    switching_violation = float(theta.detach().pow(2).sum())

    assert penalty_violation > epsilon, "the under-weighted penalty should stay infeasible"
    assert switching_violation <= epsilon * 1.10
    assert switching_violation < penalty_violation


def test_restore_step_size_is_calibrated_when_not_given():
    """The note's literal RESTORE direction, with its step size calibrated.

    ``restore_direction='eps'`` is the method note's ``grad C_B / eps``, which
    is huge for a small ``eps``; when no ``restore_lr`` is given it is
    calibrated so the first RESTORE step has the length of an OPTIMISE step
    instead of diverging.  (The default direction is Polyak's, which needs no
    such calibration -- see ``test_polyak_restore_step_lands_on_the_target_level_set``.)
    """

    theta = torch.nn.Parameter(torch.ones(4))
    epsilon = 1.0e-4
    optimizer = SwitchingOptimizer(
        [theta],
        epsilon=epsilon,
        config=SwitchingConfig(
            lr=0.01, restore_lr=None, grad_clip=None, restore_direction="eps"
        ),
    )
    optimizer.prime([1.0])
    assert optimizer.select_mode() == RESTORE
    grad_l = torch.full((4,), 0.5)
    grad_c = torch.full((4,), 2.0)
    before = theta.detach().clone()
    optimizer.step(grad_l=grad_l, grad_c=grad_c, jepa_loss=1.0, constraint=1.0)
    moved = float((theta.detach() - before).norm())
    reference = 0.01 * float(grad_l.norm())
    assert moved == pytest.approx(reference, rel=1e-5)


def test_flat_gradient_handles_missing_grads():
    a = torch.nn.Parameter(torch.zeros(3))
    b = torch.nn.Parameter(torch.zeros(2))
    a.grad = torch.ones(3)
    flat = flat_gradient([a, b])
    assert flat.shape == (5,)
    torch.testing.assert_close(flat, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))


def test_history_records_every_step():
    theta = torch.nn.Parameter(torch.zeros(2))
    optimizer = SwitchingOptimizer([theta], epsilon=1.0, config=SwitchingConfig())
    optimizer.prime([0.1])
    for _ in range(5):
        optimizer.select_mode()
        optimizer.step(
            grad_l=torch.ones(2), grad_c=torch.ones(2), jepa_loss=1.0, constraint=0.1
        )
    assert len(optimizer.state.history) == 5
    assert optimizer.summary()["optimise_steps"] == 5
    assert all(record.mode == OPTIMISE for record in optimizer.state.history)


def test_polyak_restore_step_lands_on_the_target_level_set():
    """The default RESTORE step reaches ``{C = 0.8 eps}`` to first order.

    With ``d = grad C (C - target)/||grad C||^2`` and step size one, the
    linearisation ``C - <grad C, eta d>`` equals the target exactly, whatever
    ``||grad C||`` is.  That is the property a fixed step size lacks: its
    effective length tracks ``||grad C||``, which grows as the constraint gets
    harder.
    """

    theta = torch.nn.Parameter(torch.zeros(5))
    epsilon = 2.0
    config = SwitchingConfig(restore_direction="polyak", restore_target_factor=0.8)
    optimizer = SwitchingOptimizer([theta], epsilon=epsilon, config=config)
    optimizer.prime([10.0])
    assert optimizer.select_mode() == RESTORE
    for scale in (1.0e-3, 1.0, 1.0e3):
        constraint = 10.0
        grad_c = torch.full((5,), scale)
        optimizer.observe_violation(constraint)
        optimizer.select_mode()
        direction, info = optimizer.direction(torch.ones(5), grad_c)
        predicted = constraint - float(torch.dot(grad_c, direction))
        assert predicted == pytest.approx(0.8 * epsilon, rel=1e-6)


def test_polyak_restore_step_stops_at_the_target():
    """Once the minibatch violation is below the target the step is zero."""

    theta = torch.nn.Parameter(torch.zeros(3))
    optimizer = SwitchingOptimizer(
        [theta], epsilon=1.0, config=SwitchingConfig(restore_direction="polyak")
    )
    optimizer.prime([5.0])
    optimizer.observe_violation(0.5)
    optimizer.state.mode = RESTORE
    direction, _info = optimizer.direction(torch.ones(3), torch.ones(3))
    assert float(direction.abs().max()) == 0.0


def test_fixed_restore_step_overshoots_when_the_gradient_grows():
    """Why the default changed: a step calibrated once does not stay calibrated.

    With ``restore_direction='eps'`` the step length is proportional to
    ``||grad C||``, so a gradient that grows during RESTORE lengthens every
    step; the Polyak step's length is independent of it.
    """

    lengths = {}
    for direction_mode in ("eps", "polyak"):
        theta = torch.nn.Parameter(torch.zeros(4))
        config = SwitchingConfig(
            lr=1.0e-3, restore_direction=direction_mode, grad_clip=None
        )
        optimizer = SwitchingOptimizer([theta], epsilon=1.0e-2, config=config)
        optimizer.prime([1.0])
        optimizer.select_mode()
        moved = []
        for scale in (1.0, 10.0):
            optimizer.observe_violation(1.0)
            optimizer.select_mode()
            before = theta.detach().clone()
            optimizer.step(
                grad_l=torch.ones(4) * 0.1,
                grad_c=torch.full((4,), scale),
                jepa_loss=1.0,
                constraint=1.0,
            )
            moved.append(float((theta.detach() - before).norm()))
        lengths[direction_mode] = moved
    assert lengths["eps"][1] > 5.0 * lengths["eps"][0]
    assert lengths["polyak"][1] < 0.2 * lengths["polyak"][0]


def test_epsilon_sources_differ_by_the_predictor_error():
    """``factual_rollout`` must exceed ``real_latents`` by the rollout error.

    ``real_latents`` measures only the probe's noise, because it uses the real
    future latents.  ``factual_rollout`` measures the same violation on the
    predictor's own rollout, so it also carries the predictor's error -- which
    is what decides whether ``C(theta) <= eps`` is attainable at all.
    """


    from gwm.constraint import ConstraintSpec
    from gwm.latents import LatentCache
    from gwm.posttrain import EPSILON_SOURCES, calibrate_tolerance

    torch.manual_seed(0)
    spec = ConstraintSpec(horizon=4, stride=1, history=4)
    episodes, frames, dim = 24, 8, 16
    dt = 0.2
    # Latents that are an exact affine image of a free-fall path, so a linear
    # probe is exact and ``real_latents`` isolates the predictor's error.
    times = torch.arange(frames, dtype=torch.float32) * dt
    paths = []
    for index in range(episodes):
        gravity = 3.0 + 0.1 * index
        x = 0.1 * index + 0.5 * times
        z = 2.0 + 1.5 * times - 0.5 * gravity * times**2
        paths.append(torch.stack([x, z], dim=-1))
    state = torch.zeros(episodes, frames, 8)
    state[..., :2] = torch.stack(paths)
    weights = torch.randn(2, dim)
    cache = LatentCache(
        latents=state[..., :2] @ weights,
        state=state,
        action=torch.zeros(episodes, frames, 3),
        gravity=torch.tensor([3.0 + 0.1 * i for i in range(episodes)]),
        split=torch.ones(episodes, dtype=torch.long),
        source_id=torch.arange(episodes),
        # Contact-free by construction.
        contact=torch.zeros(state.shape[:2], dtype=torch.bool),
        dt=dt,
        pixels_per_unit=10.0,
        action_mean=torch.zeros(3),
        action_std=torch.ones(3),
        metadata={},
    )
    # positions @ weights = latents, so latents @ pinv(weights) recovers positions
    inverse = torch.linalg.pinv(weights)

    class ExactProbe(torch.nn.Module):
        def forward(self, latents):
            return latents @ inverse

    class WrongPredictor(torch.nn.Module):
        """A predictor whose rollout is offset, so its error is visible."""

        def __init__(self) -> None:
            super().__init__()
            self.action_encoder = torch.nn.Identity()

        def predict(self, window, condition):
            return window + 0.05

    probe = ExactProbe()
    real = calibrate_tolerance(probe, cache, spec=spec, windows=64, seed=0, source="real_latents")
    rollout = calibrate_tolerance(
        probe, cache, spec=spec, windows=64, seed=0, source="factual_rollout",
        model=WrongPredictor(),
    )
    assert real["source"] == "real_latents"
    assert rollout["source"] == "factual_rollout"
    # An exact probe on an exact path leaves essentially nothing.
    assert real["c_real_mean"] < 1e-6
    assert rollout["c_real_mean"] > real["c_real_mean"]
    assert set(EPSILON_SOURCES) == {"real_latents", "factual_rollout"}
    with pytest.raises(ValueError, match="needs the pretrained model"):
        calibrate_tolerance(probe, cache, spec=spec, windows=8, seed=0, source="factual_rollout")
    with pytest.raises(ValueError, match="source must be one of"):
        calibrate_tolerance(probe, cache, spec=spec, windows=8, seed=0, source="nonsense")


def test_cli_defaults_track_the_library_defaults():
    """A CLI default that drifts from the library's is a silent regression.

    ``restore_direction`` defaulted to the method note's ``eps`` in the CLI
    long after the library had moved to Polyak's step, so every run through
    the CLI used the fixed-step direction that is known to diverge -- and the
    traces showed it, with the violation rising and the JEPA loss exploding.
    Deriving the CLI default from the dataclass keeps them in step.
    """

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from posttrain import build_parser

    args = build_parser().parse_args(
        ["--checkpoint", "a", "--cache", "b", "--probe-dir", "c", "--output-dir", "d"]
    )
    defaults = SwitchingConfig()
    assert args.restore_direction == defaults.restore_direction
    assert args.gate == defaults.gate
    assert args.grad_clip == defaults.grad_clip
    assert args.ema_decay == defaults.ema_decay


def test_starved_flag_is_about_restore_not_about_doing_nothing() -> None:
    """``starved_of_optimise_steps`` must distinguish 0/4000 from 0/0.

    A run that spends every step restoring feasibility has no certificate --
    Theorem 1(c) reads it off an OPTIMISE iterate -- and nothing holding the
    JEPA loss down; measured on ball_ax the loss went from 13.4 to 1.2e6.  The
    ``pretrained`` variant takes no steps at all, so the same condition is
    vacuously true there and must not be reported as that failure.
    """

    torch.manual_seed(0)
    parameters = [torch.zeros(4, requires_grad=True)]

    # zero steps: not starved, just idle
    idle = SwitchingOptimizer(parameters, epsilon=1.0, config=SwitchingConfig(lr=1e-3))
    assert idle.state.step == 0
    assert not (idle.state.step > 0 and idle.state.optimise_steps == 0)

    # every step in RESTORE: starved
    busy = SwitchingOptimizer(parameters, epsilon=1.0e-6, config=SwitchingConfig(lr=1e-3))
    busy.prime([1.0])
    for _ in range(5):
        busy.step(
            grad_l=torch.ones(4),
            grad_c=torch.ones(4),
            jepa_loss=1.0,
            constraint=1.0,          # far above epsilon, so RESTORE every time
        )
    assert busy.state.optimise_steps == 0
    assert busy.state.restore_steps == 5
    assert busy.state.step > 0 and busy.state.optimise_steps == 0
