"""The constraint must be exact on exact physics, at every stride."""

from __future__ import annotations

import math

import pytest
import torch

from gwm.constraint import (
    ConstraintSpec,
    acceleration_vector,
    condition_on_acceleration,
    condition_on_gravity,
    free_fall_targets,
    free_fall_violation,
    gravity_vector,
)


def switched_path(p_t, velocity, gravity, counterfactual, *, offsets, dt):
    """Lemma 1's trajectory: gravity ``gravity`` before ``t``, ``counterfactual`` after."""

    rows = []
    for offset in offsets:
        law = counterfactual if offset >= 0 else gravity
        time = offset * dt
        rows.append(p_t + velocity * time + 0.5 * law * time**2)
    return torch.stack(rows, dim=1)


def test_targets_reduce_to_lemma_one_at_unit_stride():
    g = torch.tensor([[0.0, -4.0]])
    gp = torch.tensor([[0.0, -9.0]])
    offsets = (0, 1, 2, 3)
    targets = free_fall_targets(offsets, g, gp, stride=1)
    torch.testing.assert_close(targets[0, 0], 0.5 * (g + gp)[0])
    for index in range(1, len(offsets)):
        torch.testing.assert_close(targets[0, index], gp[0])


@pytest.mark.parametrize("stride", [1, 2, 3, 4])
def test_targets_are_the_accelerations_of_the_exact_switched_path(stride):
    """The target of every term is, by construction, that term's true value."""

    dt = 0.16
    g = torch.tensor([[0.3, -4.0]])
    gp = torch.tensor([[-0.7, -9.5]])
    offsets = tuple(range(1 - stride, 7 - stride + 1))
    span = tuple(range(min(offsets) - stride, max(offsets) + stride + 1))
    path = switched_path(
        torch.tensor([[1.5, 2.5]]),
        torch.tensor([[0.8, 3.1]]),
        g,
        gp,
        offsets=span,
        dt=dt,
    )
    index = {offset: position for position, offset in enumerate(span)}
    measured = torch.stack(
        [
            (path[:, index[o + stride]] - 2.0 * path[:, index[o]] + path[:, index[o - stride]])
            / (stride * dt) ** 2
            for o in offsets
        ],
        dim=1,
    )
    torch.testing.assert_close(
        measured, free_fall_targets(offsets, g, gp, stride=stride), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("horizon", [4, 6, 8])
def test_violation_vanishes_on_the_exact_counterfactual_trajectory(stride, horizon):
    if horizon < stride:
        pytest.skip("horizon must cover the stride")
    dt = 0.16
    spec = ConstraintSpec(horizon=horizon, stride=stride, history=2 * stride + 2)
    g_scalar = torch.tensor([4.0, 2.0])
    gp_scalar = torch.tensor([9.0, 0.5])
    g, gp = gravity_vector(g_scalar), gravity_vector(gp_scalar)
    offsets = tuple(range(-(spec.history - 1), spec.horizon + 1))
    path = switched_path(
        torch.tensor([[1.0, 3.0], [-2.0, 4.0]]),
        torch.tensor([[0.5, 2.0], [-1.0, 1.0]]),
        g,
        gp,
        offsets=offsets,
        dt=dt,
    )
    violation, _, _ = free_fall_violation(
        path, gravity=g, counterfactual=gp, spec=spec, dt=dt
    )
    assert violation.shape == (2,)
    assert float(violation.max()) < 1e-6


@pytest.mark.parametrize("stride", [1, 2])
def test_gravity_blind_rollout_violation_grows_with_the_gravity_shift(stride):
    """A predictor that ignores ``g'`` is penalised in proportion to the shift.

    This is the mechanism behind Theorem 2: the counterfactual error of a
    gravity-blind predictor is bounded below by the displacement the change of
    gravity causes, which grows with ``||g' - g||``.
    """

    dt = 0.16
    spec = ConstraintSpec(horizon=6, stride=stride, history=2 * stride + 2)
    offsets = tuple(range(-(spec.history - 1), spec.horizon + 1))
    g_scalar = torch.tensor([4.0])
    g = gravity_vector(g_scalar)
    previous = -1.0
    for shift in (0.0, 1.0, 2.0, 4.0, 8.0):
        gp = gravity_vector(g_scalar + shift)
        # The path keeps moving under ``g``: it is blind to ``g'``.
        path = switched_path(
            torch.tensor([[1.0, 3.0]]),
            torch.tensor([[0.5, 2.0]]),
            g,
            g,
            offsets=offsets,
            dt=dt,
        )
        violation, _, _ = free_fall_violation(
            path, gravity=g, counterfactual=gp, spec=spec, dt=dt
        )
        value = float(violation)
        assert value > previous
        previous = value
    assert previous > 1.0


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_zero_violation_pins_every_predicted_position(stride):
    """Proposition 1 carries over to a strided difference.

    ``c = 0`` fixes every strided second difference, and the resulting
    recursion determines ``p_{t+1}, ..., p_{t+H}`` uniquely from the context.
    The offsets of :class:`ConstraintSpec` start at ``1 - stride`` precisely so
    that each residue class modulo ``stride`` is seeded inside the context;
    this test checks that the recursion reaches every predicted index.
    """

    spec = ConstraintSpec(horizon=8, stride=stride, history=2 * stride + 2)
    pivot = spec.history - 1
    known = set(range(0, pivot + 1))
    for offset in spec.offsets:
        index = pivot + offset
        assert index - spec.stride in known, "a term reaches outside the context"
        if index in known:
            known.add(index + spec.stride)
    assert set(range(pivot + 1, pivot + 1 + spec.horizon)) <= known


def test_condition_on_gravity_rewrites_only_the_gravity_coordinate():
    action = torch.randn(3, 7, 3)
    mean = torch.tensor([0.5, -0.25, 4.0])
    std = torch.tensor([2.0, 1.5, 0.5])
    target = torch.tensor([9.0, 1.0, 4.0])
    out = condition_on_gravity(action, target, action_mean=mean, action_std=std)
    torch.testing.assert_close(out[:, :, :2], action[:, :, :2])
    expected = ((target - mean[2]) / std[2]).unsqueeze(-1).expand(3, 7)
    torch.testing.assert_close(out[:, :, 2], expected)


def test_constraint_spec_rejects_unusable_shapes():
    with pytest.raises(ValueError, match="at least stride"):
        ConstraintSpec(horizon=1, stride=2, history=8)
    with pytest.raises(ValueError, match="2\\*stride"):
        ConstraintSpec(horizon=6, stride=3, history=5)


def test_offsets_cover_the_documented_range():
    spec = ConstraintSpec(horizon=6, stride=2, history=8)
    assert spec.offsets == (-1, 0, 1, 2, 3, 4)
    assert spec.num_terms == spec.horizon


def test_stride_scales_the_resolvable_gravity_signal():
    """The displacement the constraint must resolve grows as ``m^2``."""

    dt, gravity = 0.16, 4.0
    for stride in (1, 2, 3):
        assert math.isclose(gravity * (stride * dt) ** 2, gravity * dt**2 * stride**2)


@pytest.mark.parametrize("form", ["acceleration", "position"])
def test_both_forms_vanish_on_the_exact_switched_trajectory(form):
    """Proposition 1: the two forms have the same zero set."""

    dt = 0.16
    spec = ConstraintSpec(horizon=6, stride=1, history=8, form=form)
    g_scalar = torch.tensor([4.0, 2.0])
    gp_scalar = torch.tensor([9.0, 0.5])
    g, gp = gravity_vector(g_scalar), gravity_vector(gp_scalar)
    offsets = tuple(range(-(spec.history - 1), spec.horizon + 1))
    path = switched_path(
        torch.tensor([[1.0, 3.0], [-2.0, 4.0]]),
        torch.tensor([[0.5, 2.0], [-1.0, 1.0]]),
        g,
        gp,
        offsets=offsets,
        dt=dt,
    )
    violation, _a, _b = free_fall_violation(
        path, gravity=g, counterfactual=gp, spec=spec, dt=dt
    )
    assert float(violation.max()) < 1e-6


def test_position_form_sees_a_sign_structure_the_squared_form_misses():
    """Why the two forms optimise differently despite sharing a zero set.

    Definition 1 averages the *squares* of the acceleration residuals; by
    Lemma 7 the position error is their signed, time-weighted sum.  The two
    therefore weight the same residual completely differently: measured below,
    an unsystematic residual and a systematic one that cost the *same* in
    position differ by more than two orders of magnitude in the squared form.

    The consequence is that the squared form's gradient is dominated by the
    unsystematic component, which is cheap to remove and nearly irrelevant to
    position -- which is what the real runs showed, where driving the squared
    violation from 57.7 to 19.2 left the residual's signed part untouched and
    made the terminal position error worse.
    """

    from gwm.constraint import free_fall_position_violation

    dt = 0.16
    accel_spec = ConstraintSpec(horizon=6, stride=1, history=8, form="acceleration")
    pos_spec = ConstraintSpec(horizon=6, stride=1, history=8, form="position")
    g = gravity_vector(torch.tensor([4.0]))
    gp = gravity_vector(torch.tensor([9.0]))
    offsets = tuple(range(-(accel_spec.history - 1), accel_spec.horizon + 1))
    exact = switched_path(
        torch.tensor([[1.0, 6.0]]), torch.tensor([[0.5, 2.0]]), g, gp, offsets=offsets, dt=dt
    )
    pivot = accel_spec.history - 1

    def scores(path):
        accel, _a, _b = free_fall_violation(
            path, gravity=g, counterfactual=gp, spec=accel_spec, dt=dt
        )
        position, _c, _d = free_fall_position_violation(
            path, gravity=g, counterfactual=gp, spec=pos_spec, dt=dt
        )
        return float(accel), float(position)

    assert max(scores(exact)) < 1e-8, "both forms must vanish on the exact trajectory"

    # A systematic residual: a constant extra acceleration after the switch.
    times = torch.clamp(
        torch.arange(-pivot, accel_spec.horizon + 1, dtype=torch.float32) * dt, min=0.0
    )
    systematic = exact.clone()
    systematic[0, :, 1] -= 0.5 * 0.6 * times**2
    accel_sys, pos_sys = scores(systematic)

    # An unsystematic residual chosen to cost a comparable amount in position.
    torch.manual_seed(0)
    noise = torch.randn(1, exact.shape[1], 2)
    noise[:, : pivot + 1] = 0.0
    noisy = exact + noise * 0.12
    accel_noisy, pos_noisy = scores(noisy)

    assert pos_noisy == pytest.approx(pos_sys, rel=0.5), "the two must be comparable in position"
    assert accel_noisy > 100.0 * accel_sys, (
        "the squared form must over-penalise the unsystematic residual by orders of "
        f"magnitude; got {accel_noisy:.3f} against {accel_sys:.3f}"
    )


def _descend(form: str, lr: float, iterations: int = 4000) -> tuple[float, float]:
    """Optimise the predicted positions directly and report the distance ratio.

    The cleanest possible stand-in for a predictor: the decoded positions are
    the free variables, so what is measured is the constraint's own landscape
    rather than any network's.
    """

    dt = 0.16
    spec = ConstraintSpec(horizon=6, stride=1, history=8, form=form)
    gravity = gravity_vector(torch.tensor([4.0]))
    counterfactual = gravity_vector(torch.tensor([9.0]))
    offsets = tuple(range(-(spec.history - 1), spec.horizon + 1))
    exact = switched_path(
        torch.tensor([[1.0, 6.0]]),
        torch.tensor([[0.5, 2.0]]),
        gravity,
        counterfactual,
        offsets=offsets,
        dt=dt,
    )
    pivot = spec.history - 1
    context = exact[:, : pivot + 1].clone()
    target = exact[:, pivot + 1 :]
    torch.manual_seed(0)
    predicted = torch.nn.Parameter(target.clone() + torch.randn_like(target) * 0.4)
    before = float((predicted.detach() - target).pow(2).sum(-1).mean().sqrt())
    optimizer = torch.optim.SGD([predicted], lr=lr)
    for _ in range(iterations):
        violation, _a, _b = free_fall_violation(
            torch.cat([context, predicted], dim=1),
            gravity=gravity,
            counterfactual=counterfactual,
            spec=spec,
            dt=dt,
        )
        optimizer.zero_grad(set_to_none=True)
        violation.sum().backward()
        optimizer.step()
        if not torch.isfinite(predicted).all():
            return before, float("inf")
    return before, float((predicted.detach() - target).pow(2).sum(-1).mean().sqrt())


@pytest.mark.parametrize("form", ["acceleration", "position"])
def test_both_forms_descend_towards_the_exact_trajectory(form):
    """Descending either form must move the positions towards the truth."""

    lr = 2.0e-4 if form == "acceleration" else 5.0e-2
    before, after = _descend(form, lr)
    assert after < before


def test_the_position_form_is_far_better_conditioned():
    """The two forms share a zero set but not a landscape, by a wide margin.

    Definition 1's second difference squares the ill-conditioning of
    integration; Proposition 1's form inverts it in closed form.  Measured on
    the same problem with the same number of steps: the squared form is stable
    only up to a step size of about 2e-4 and gets within 4% of the exact
    trajectory, while the position form is stable at 250x that step and gets
    within 0.001%.  That three-to-four order of magnitude gap is why
    optimising Definition 1 barely improves position accuracy in practice,
    even though driving it to zero would.
    """

    before, accel_best = _descend("acceleration", 2.0e-4)
    _b, accel_unstable = _descend("acceleration", 5.0e-4)
    _b, position_best = _descend("position", 5.0e-2)

    assert accel_unstable == float("inf"), "the squared form should diverge at 5e-4"
    assert accel_best < before, "it must still make progress at a stable step"
    assert position_best < accel_best / 100.0, (
        "the position form must converge orders of magnitude further; got "
        f"{position_best:.6f} against {accel_best:.6f}"
    )


def test_acceleration_vector_generalises_the_scalar_case_exactly() -> None:
    """``acceleration_vector(g, None)`` must be the old ``(0, -g)``, bit for bit.

    The two-dimensional experiment conditions the model on an acceleration
    *direction* it never saw, not just a magnitude.  The generalisation is only
    sound if the scalar path it replaces is untouched, so this pins that.
    """

    gravity = torch.tensor([1.0, 4.0, 9.0])
    assert torch.equal(acceleration_vector(gravity), gravity_vector(gravity))
    assert torch.equal(
        acceleration_vector(gravity, torch.zeros_like(gravity)), gravity_vector(gravity)
    )

    accel_x = torch.tensor([-3.0, 0.5, 2.0])
    vector = acceleration_vector(gravity, accel_x)
    assert torch.equal(vector[:, 0], accel_x)
    assert torch.equal(vector[:, 1], -gravity)

    with pytest.raises(ValueError, match="accel_x must match"):
        acceleration_vector(gravity, torch.zeros(2))


def test_condition_on_acceleration_reduces_to_condition_on_gravity() -> None:
    """With no horizontal coordinate it must equal the scalar conditioning."""

    torch.manual_seed(0)
    action = torch.randn(5, 7, 4)
    mean = torch.tensor([0.0, 0.3, 4.0, 0.0])
    std = torch.tensor([0.3, 1.5, 2.309, 1.2])
    gravity = torch.tensor([1.0, 3.0, 4.0, 7.0, 9.0])

    scalar = condition_on_gravity(action, gravity, action_mean=mean, action_std=std)
    for accel_x, index in ((None, None), (None, 3), (torch.zeros(5), None)):
        assert torch.equal(
            condition_on_acceleration(
                action,
                gravity=gravity,
                accel_x=accel_x,
                action_mean=mean,
                action_std=std,
                accel_x_index=index,
            ),
            scalar,
        )

    accel_x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    both = condition_on_acceleration(
        action,
        gravity=gravity,
        accel_x=accel_x,
        action_mean=mean,
        action_std=std,
        accel_x_index=3,
    )
    # gravity coordinate unchanged from the scalar call, horizontal written
    assert torch.equal(both[:, :, 2], scalar[:, :, 2])
    assert torch.allclose(both[:, :, 3], ((accel_x - mean[3]) / std[3]).unsqueeze(-1).expand(5, 7))
    # the untouched coordinates are the original action's
    assert torch.equal(both[:, :, :2], action[:, :, :2])


def test_free_fall_violation_vanishes_on_an_exact_two_dimensional_switch() -> None:
    """Proposition 1's zero set must hold for a vector counterfactual too.

    The targets are a function of the planar acceleration already, so a path
    built from a counterfactual with a horizontal component must have zero
    violation in both forms -- otherwise the two-dimensional experiment would
    be measuring an artefact of the scalar assumption.
    """

    dt = 0.16
    for stride in (1, 2, 3):
        spec_a = ConstraintSpec(horizon=6, stride=stride, history=8, form="acceleration")
        spec_p = ConstraintSpec(horizon=6, stride=stride, history=8, form="position")
        torch.manual_seed(stride)
        batch = 9
        gravity = torch.full((batch,), 4.0)
        accel_x = torch.linspace(-3.0, 3.0, batch)
        g_vec = acceleration_vector(gravity, torch.zeros(batch))
        c_vec = acceleration_vector(torch.linspace(1.0, 9.0, batch), accel_x)

        anchor = torch.randn(batch, 2)
        velocity = torch.randn(batch, 2)
        offsets = torch.arange(-(spec_a.history - 1), spec_a.horizon + 1, dtype=torch.float64)
        times = offsets * dt
        law = torch.where(
            (offsets >= 0).reshape(1, -1, 1), c_vec.unsqueeze(1).double(), g_vec.unsqueeze(1).double()
        )
        path = (
            anchor.unsqueeze(1).double()
            + velocity.unsqueeze(1).double() * times.reshape(1, -1, 1)
            + 0.5 * law * times.reshape(1, -1, 1) ** 2
        ).float()

        for spec in (spec_a, spec_p):
            value, _a, _b = free_fall_violation(
                path, gravity=g_vec, counterfactual=c_vec, spec=spec, dt=dt
            )
            assert value.abs().max() < 1e-8, (spec.form, stride, float(value.abs().max()))
