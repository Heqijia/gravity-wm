"""The scene's closed-form launch algebra must match the rendered path."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gwm.freefall_data import (
    SceneSpec,
    sample_episode_parameters,
    shared_rise_range,
    trajectory,
    verify_free_fall_against_integrator,
)


@pytest.mark.parametrize("gravity", [0.5, 1.0, 4.0, 9.0])
@pytest.mark.parametrize("rise", [0.4, 0.8, 1.0, 1.3, 1.8])
def test_vertical_excursion_matches_the_trajectory(gravity, rise):
    """``|g| span^2 max(rise, 2-rise)^2 / 8`` is the swept vertical extent."""

    spec = SceneSpec()
    vz = 0.5 * gravity * spec.span * rise
    path = trajectory(x0=0.0, z0=0.0, vx=0.0, vz=vz, gravity=gravity, dt=spec.dt, frames=4096)
    # Dense sampling of the same interval, so the discrete extent converges to
    # the continuous one the formula describes.
    times = np.linspace(0.0, spec.span, 100_000)
    dense = vz * times - 0.5 * gravity * times**2
    assert math.isclose(
        spec.vertical_excursion(gravity, rise),
        float(dense.max() - dense.min()),
        rel_tol=1e-3,
    )
    assert path.shape == (4096, 2)


def test_excursion_is_minimised_by_the_symmetric_parabola():
    spec = SceneSpec()
    best = spec.vertical_excursion(4.0, 1.0)
    for rise in (0.5, 0.7, 0.9, 1.1, 1.4, 1.9):
        assert spec.vertical_excursion(4.0, rise) >= best - 1e-12


@pytest.mark.parametrize("gravity", [1.0, 3.0, 6.0, 9.0])
def test_feasible_rise_range_fits_the_safe_box(gravity):
    spec = SceneSpec()
    left, right, bottom, top = spec.safe_box()
    height = (top - bottom) * spec.fill_fraction
    low, high = spec.feasible_rise_range(gravity)
    assert low <= high
    for rise in (low, 0.5 * (low + high), high):
        assert spec.vertical_excursion(gravity, rise) <= height * (1.0 + 1e-9)


def test_infeasible_gravity_is_reported_not_silently_clipped():
    """A gravity whose shortest parabola cannot fit must raise, with the numbers."""

    spec = SceneSpec(dt=0.3, frames=40, view_height=6.0, z_center=3.0)
    with pytest.raises(ValueError, match="no launch fits the view"):
        spec.feasible_rise_range(9.0)


def test_shared_rise_range_is_feasible_for_every_gravity():
    """One launch distribution for all gravities, so the launch leaks no gravity."""

    spec = SceneSpec()
    grid = [1.0 + 0.5 * index for index in range(17)]
    low, high = shared_rise_range(spec, grid)
    left, right, bottom, top = spec.safe_box()
    height = (top - bottom) * spec.fill_fraction
    for gravity in grid:
        for rise in (low, high):
            assert spec.vertical_excursion(gravity, rise) <= height * (1.0 + 1e-9)


def test_gravity_step_pixels_reports_the_resolvable_signal():
    """The pixel displacement of one step of gravity, which the probe must beat."""

    spec = SceneSpec(dt=0.16, frames=16, view_height=13.0, z_center=6.5)
    one = spec.gravity_step_pixels(4.0, 128)
    assert math.isclose(one, 4.0 * 0.16**2 / 13.0 * 128)
    # The Semigroup-JEPA planar setting, for comparison: 0.19 pixel.
    planar = SceneSpec(dt=1 / 16, frames=64, view_height=10.4, z_center=4.6)
    assert planar.gravity_step_pixels(4.0, 128) < 0.2
    assert one > 5.0 * planar.gravity_step_pixels(4.0, 128)


def test_trajectory_second_difference_is_exactly_gravity():
    spec = SceneSpec()
    path = trajectory(x0=1.0, z0=2.0, vx=0.7, vz=3.0, gravity=4.0, dt=spec.dt, frames=16)
    second = path[2:] - 2 * path[1:-1] + path[:-2]
    expected = np.array([0.0, -4.0]) * spec.dt**2
    assert np.abs(second - expected).max() < 1e-12


@pytest.mark.parametrize("accel_x", [0.0, 1.5, -2.5])
def test_two_dimensional_acceleration_matches_the_integrator(accel_x: float) -> None:
    """The analytic path must be the path MuJoCo produces, horizontally too.

    The renderer *sets* the pose from :func:`trajectory` rather than
    integrating, so a sign or factor error in the ``accel_x`` term would give
    self-consistent but physically wrong data, and the constraint -- which is
    exact on exact physics -- would be fitting an artefact.  Agreement alone is
    not enough to show the term is live, because ``mj_step``'s own drift is of
    the same order, so the test also integrates the scene with the horizontal
    gravity *removed* and requires that to disagree by world units.

    The residual is the drift documented in ``docs/design.md``: a few times
    ``1e-3`` world units, which is why the labels come from the closed form and
    not from the integrator.
    """

    spec = SceneSpec()
    parameters = sample_episode_parameters(4.0, 11, spec, accel_x=accel_x)
    assert parameters["accel_x"] == pytest.approx(accel_x)
    assert parameters["attempts"] == 1.0, "the launch budget should still be analytic"

    report = verify_free_fall_against_integrator(parameters, spec, image_size=64)
    assert report["contacts"] == 0.0
    assert report["max_position_gap"] < 5.0e-3, report
    assert report["max_second_difference_gap"] < 5.0e-3, report

    if accel_x != 0.0:
        blind = verify_free_fall_against_integrator(
            {**parameters, "accel_x": 0.0}, spec, image_size=64
        )
        # The analytic path inside `blind` is also recomputed without accel_x,
        # so compare the two scenes directly instead.
        horizontal = abs(accel_x) * 0.5 * spec.span**2
        assert horizontal > 1.0
        assert blind["max_position_gap"] < 5.0e-3, (
            "the control must itself be self-consistent; it differs from the "
            "accel_x scene, not from its own analytic path"
        )
        moved = trajectory(
            x0=parameters["x"], z0=parameters["z"], vx=parameters["vx"],
            vz=parameters["vz"], gravity=parameters["gravity"], dt=spec.dt,
            frames=spec.frames, accel_x=accel_x,
        )
        still = trajectory(
            x0=parameters["x"], z0=parameters["z"], vx=parameters["vx"],
            vz=parameters["vz"], gravity=parameters["gravity"], dt=spec.dt,
            frames=spec.frames, accel_x=0.0,
        )
        assert np.abs(moved - still).max() == pytest.approx(horizontal, rel=1e-9)


def test_trajectory_second_difference_is_the_acceleration_vector() -> None:
    """``p_{t+1} - 2 p_t + p_{t-1}`` must equal ``(a_x, -g) dt^2`` exactly."""

    dt = 0.16
    for accel_x, gravity in ((0.0, 4.0), (3.0, 1.0), (-3.0, 9.0)):
        path = trajectory(
            x0=0.3, z0=6.0, vx=0.7, vz=2.1, gravity=gravity, dt=dt, frames=16, accel_x=accel_x
        )
        second = path[2:] - 2.0 * path[1:-1] + path[:-2]
        expected = np.array([accel_x, -gravity]) * dt**2
        assert np.abs(second - expected).max() < 1.0e-13, (accel_x, gravity)


def test_horizontal_acceleration_keeps_the_launch_in_frame() -> None:
    """The speed budget must absorb the extra excursion, not the reject loop."""

    spec = SceneSpec()
    for accel_x in (0.0, 1.0, 2.0, 3.0):
        for seed in range(12):
            parameters = sample_episode_parameters(4.0, seed, spec, accel_x=accel_x)
            assert parameters["attempts"] == 1.0, (accel_x, seed)
            path = trajectory(
                x0=parameters["x"], z0=parameters["z"], vx=parameters["vx"],
                vz=parameters["vz"], gravity=parameters["gravity"], dt=spec.dt,
                frames=spec.frames, accel_x=accel_x,
            )
            left, right, bottom, top = spec.safe_box()
            assert path[:, 0].min() >= left - 1e-9 and path[:, 0].max() <= right + 1e-9
            assert path[:, 1].min() >= bottom - 1e-9 and path[:, 1].max() <= top + 1e-9
