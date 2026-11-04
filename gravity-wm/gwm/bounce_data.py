"""Bouncing projectile generator: the first scene in this study that has contacts.

Every dataset up to now is contact-free by construction, because the free-fall
identity of Lemma 1 -- the identity the constraint is the residual of -- holds
only under uniform acceleration.  That makes the physics a single family:
parabolas.  This generator adds a floor so the world model has to learn
something the constraint says nothing about, while the constraint keeps its
exact meaning on the windows between bounces.

Three measurements decided the design, taken before any of it was built:

* MuJoCo's own free-fall residual between contacts, integrating with ``RK4`` at
  a 0.0005 substep, is ``4e-13`` world units -- as exact as the analytic
  generator, and eleven orders of magnitude below the ``1.008`` px one-step
  gravity displacement.  So ``mj_step`` can be the source of truth here and the
  constraint loses no signal to integrator drift.
* A window that spans a contact scores ``2.5-5.1`` px against that same
  ``1.008`` px signal.  Such windows must be *excluded* from the constraint,
  not counted as violations, which is why every frame carries a contact flag.
* The restitution has to be measured rather than assumed.  ``solref="0.002
  0.05"`` returns ``e = 1.6`` -- the ball leaves faster than it arrives, which
  is not physics.  ``solref="-5000 -5"`` returns ``e = 0.49``.  The generator
  measures what it actually produced and records it in the manifest.

The floor is rendered.  A world model cannot predict a bounce it cannot see
coming, so hiding the floor would charge the JEPA loss for an unpredictable
event and confound every comparison the loss appears in.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from gwm.freefall_data import (
    SHAPE_IDS,
    STATE_NAMES,
    _format,
    action_names,
)

TASK = "bounce_ball"
PHYSICS_SUBSTEP = 0.0005
# Direct stiffness/damping.  The positive-pair forms of ``solref`` were measured
# to return a restitution above one on this scene; see the module docstring.
CONTACT_SOLREF = "-5000 -5"
CONTACT_SOLIMP = "0.99 0.999 0.0001"
TEST_GRAVITIES = tuple(float(g) for g in np.arange(1.0, 9.01, 0.5))


@dataclass(frozen=True)
class BounceSpec:
    """Geometry and sampling ranges of the bouncing scene.

    ``floor_z`` is the top surface of the floor, so an object of half-height
    ``half_size`` rests with its centre at ``floor_z + half_size``.  The view
    spans ``[z_center - view_height/2, z_center + view_height/2]``, and the
    defaults put the floor exactly at the bottom edge of the frame so the whole
    bounce is visible.
    """

    dt: float = 0.16
    frames: int = 32
    view_height: float = 13.0
    z_center: float = 6.5
    shape: str = "ball"
    half_size: float = 0.45
    # Above the bottom of the view, so the floor is *rendered*.  A world model
    # cannot predict a bounce it cannot see coming, and an invisible floor
    # would charge the JEPA loss for an unpredictable event in every
    # comparison the loss appears in.  The band spans
    # ``[floor_z - floor_thickness, floor_z]`` and the view starts at
    # ``z_center - view_height/2 = 0``.
    floor_z: float = 0.80
    floor_thickness: float = 0.55
    mass_range: tuple[float, float] = (0.5, 2.0)
    spin_range: tuple[float, float] = (-0.6, 0.6)
    angle_range: tuple[float, float] = (-math.pi, math.pi)
    speed_x_range: tuple[float, float] = (-1.6, 1.6)
    # Measured, not guessed.  At 16 frames a 14-frame window has only three
    # starts, so a bounce anywhere in the context kills most of them: pushing
    # the bounce into the context took the constraint-usable share from 93% to
    # 19%.  Lengthening the episode is the knob that resolves it -- at 32
    # frames there are 19 window starts, 52% are usable by the constraint, 38%
    # contain a bounce the world model has to predict, and 37% are both.
    launch_height_range: tuple[float, float] = (3.5, 7.0)
    launch_speed_z_range: tuple[float, float] = (-1.0, 3.0)
    margin: float = 0.35
    min_contacts: int = 1
    max_rejections: int = 400

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.frames < 3:
            raise ValueError("dt must be positive and frames at least 3")
        if self.shape not in SHAPE_IDS:
            raise ValueError(f"unknown shape {self.shape!r}")
        if self.half_size <= 0.0 or self.view_height <= 0.0:
            raise ValueError("half_size and view_height must be positive")
        if self.min_contacts < 0:
            raise ValueError("min_contacts must be non-negative")
        if abs(self.dt / PHYSICS_SUBSTEP - round(self.dt / PHYSICS_SUBSTEP)) > 1e-9:
            raise ValueError(f"dt={self.dt} is not a multiple of the physics substep")

    @property
    def substeps(self) -> int:
        return int(round(self.dt / PHYSICS_SUBSTEP))

    @property
    def half_view(self) -> float:
        return 0.5 * self.view_height

    @property
    def resting_z(self) -> float:
        return self.floor_z + self.half_size

    def safe_box(self) -> tuple[float, float, float, float]:
        """Positions whose rendered object is wholly inside the frame.

        The lower bound is ``resting_z`` rather than the usual frame margin:
        the floor stops the object there, so a margin below it would reject
        every episode that actually bounces -- which is every episode this
        generator is for.
        """

        reach = self.half_size + self.margin
        return (
            -self.half_view + reach,
            self.half_view - reach,
            self.resting_z,
            self.z_center + self.half_view - reach,
        )


def model_xml(parameters: Mapping[str, float], spec: BounceSpec, image_size: int) -> str:
    """The free body, a collidable floor, a white backdrop and the same camera.

    Unlike :func:`gwm.freefall_data._model_xml` the object and the floor both
    collide, which is the entire point of this scene.  The backdrop keeps
    ``contype=0`` so it can never take part in a contact.
    """

    root = ET.Element("mujoco", model="bounce_projectile")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(
        root,
        "option",
        timestep=_format(PHYSICS_SUBSTEP),
        gravity=_format((0.0, 0.0, -float(parameters["gravity"]))),
        integrator="RK4",
        solver="Newton",
        iterations="100",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(image_size), offheight=str(image_size))
    ET.SubElement(visual, "quality", offsamples="8")
    ET.SubElement(visual, "headlight", ambient="1 1 1", diffuse="0 0 0", specular="0 0 0")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "camera",
        name="render_camera",
        pos=_format((0.0, -20.0, spec.z_center)),
        xyaxes="1 0 0 0 0 1",
        projection="orthographic",
        fovy=_format(spec.view_height),
        resolution=f"{image_size} {image_size}",
    )
    ET.SubElement(
        world,
        "geom",
        name="background_panel",
        type="box",
        pos=_format((0.0, 20.0, spec.z_center)),
        size=_format((spec.view_height, 0.02, spec.view_height)),
        rgba="1 1 1 1",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="box",
        pos=_format((0.0, 0.0, spec.floor_z - spec.floor_thickness)),
        size=_format((spec.view_height, 0.40, spec.floor_thickness)),
        contype="1",
        conaffinity="1",
        rgba=_format((0.20, 0.20, 0.24, 1.0)),
        condim="1",
        solref=CONTACT_SOLREF,
        solimp=CONTACT_SOLIMP,
    )

    body = ET.SubElement(world, "body", name="moving_object", pos="0 0 0")
    ET.SubElement(body, "joint", name="slide_x", type="slide", axis="1 0 0")
    ET.SubElement(body, "joint", name="slide_z", type="slide", axis="0 0 1")
    ET.SubElement(body, "joint", name="hinge_y", type="hinge", axis="0 1 0")
    geometry = (
        {"type": "sphere", "size": _format(spec.half_size)}
        if spec.shape == "ball"
        else {"type": "box", "size": _format((spec.half_size, 0.35, spec.half_size))}
    )
    ET.SubElement(
        body,
        "geom",
        name="moving_object",
        **geometry,
        mass=_format(parameters["mass"]),
        rgba=_format((0.55, 0.12, 0.12, 1.0)),
        condim="1",
        solref=CONTACT_SOLREF,
        solimp=CONTACT_SOLIMP,
    )
    return ET.tostring(root, encoding="unicode")


def simulate_episode(
    parameters: Mapping[str, float], spec: BounceSpec, *, image_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Integrate and render one episode, recording where the contacts are.

    The pose is *stepped*, not set: the contacts are the reason this scene
    exists, and they only happen if the solver is allowed to produce them.
    ``contact[t]`` is True when any substep between frame ``t`` and frame
    ``t + 1`` had a contact, so a window ``[t, t+2]`` is usable by the
    constraint exactly when ``contact[t:t+2]`` is all False.
    """

    import mujoco

    model = mujoco.MjModel.from_xml_string(model_xml(parameters, spec, image_size))
    data = mujoco.MjData(model)
    slide_x = int(model.joint("slide_x").qposadr[0])
    slide_z = int(model.joint("slide_z").qposadr[0])
    hinge = int(model.joint("hinge_y").qposadr[0])
    dof_x = int(model.joint("slide_x").dofadr[0])
    dof_z = int(model.joint("slide_z").dofadr[0])
    dof_hinge = int(model.joint("hinge_y").dofadr[0])

    mujoco.mj_resetData(model, data)
    data.qpos[slide_x] = float(parameters["x"])
    data.qpos[slide_z] = float(parameters["z"])
    data.qpos[hinge] = float(parameters["angle"])
    data.qvel[dof_x] = float(parameters["vx"])
    data.qvel[dof_z] = float(parameters["vz"])
    data.qvel[dof_hinge] = float(parameters["spin"])
    mujoco.mj_forward(model, data)

    frames = int(spec.frames)
    pixels = np.empty((frames, image_size, image_size, 3), dtype=np.uint8)
    state = np.empty((frames, len(STATE_NAMES)), dtype=np.float32)
    contact = np.zeros(frames, dtype=bool)
    names = action_names(False)
    action = np.zeros((frames, len(names)), dtype=np.float32)
    action[:, 2] = float(parameters["gravity"])
    action[0, :2] = float(parameters["mass"]) * np.asarray(
        (parameters["vx"], parameters["vz"]), dtype=np.float32
    )

    renderer = mujoco.Renderer(model, height=image_size, width=image_size)
    try:
        for frame in range(frames):
            state[frame] = (
                data.qpos[slide_x],
                data.qpos[slide_z],
                data.qvel[dof_x],
                data.qvel[dof_z],
                data.qpos[hinge],
                data.qvel[dof_hinge],
                0.0,
                spec.z_center,
            )
            renderer.update_scene(data, camera="render_camera")
            renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 0
            pixels[frame] = renderer.render()
            if frame + 1 < frames:
                touched = False
                for _ in range(spec.substeps):
                    mujoco.mj_step(model, data)
                    if data.ncon > 0:
                        touched = True
                contact[frame] = touched
    finally:
        renderer.close()

    path = state[:, :2].astype(np.float64)
    second = path[2:] - 2.0 * path[1:-1] + path[:-2]
    target = np.asarray((0.0, -float(parameters["gravity"])), dtype=np.float64) * spec.dt**2
    residual = np.linalg.norm(second - target, axis=1)
    clean = ~(contact[:-2] | contact[1:-1])
    diagnostics = {
        "contacts": int(contact.sum()),
        "clean_windows": int(clean.sum()),
        "total_windows": int(len(residual)),
        # The constraint's own noise floor on this episode: how exactly the
        # integrator reproduces ``g dt**2`` where Lemma 1 applies.
        "free_fall_residual_max": float(residual[clean].max()) if clean.any() else float("nan"),
        "contact_window_residual_max": (
            float(residual[~clean].max()) if (~clean).any() else float("nan")
        ),
        "x_min": float(path[:, 0].min()),
        "x_max": float(path[:, 0].max()),
        "z_min": float(path[:, 1].min()),
        "z_max": float(path[:, 1].max()),
    }
    return pixels, action, state, contact, diagnostics


def measure_restitution(spec: BounceSpec, gravity: float, *, trials: int = 24) -> dict[str, float]:
    """Drop the object and measure the restitution the solver actually produces.

    Worth its own function because the first ``solref`` tried on this scene
    returned ``e = 1.6``: the object left the floor faster than it arrived.
    Anything at or above one is rejected rather than recorded, since a dataset
    that gains energy at every bounce is not the physics this is meant to test.
    """

    import mujoco

    speeds_in, speeds_out = [], []
    for trial in range(trials):
        height = spec.resting_z + 2.0 + 4.0 * trial / max(trials - 1, 1)
        parameters = {"gravity": gravity, "mass": 1.0, "x": 0.0, "z": height,
                      "vx": 0.0, "vz": 0.0, "angle": 0.0, "spin": 0.0}
        model = mujoco.MjModel.from_xml_string(model_xml(parameters, spec, 64))
        data = mujoco.MjData(model)
        slide_z = int(model.joint("slide_z").qposadr[0])
        dof_z = int(model.joint("slide_z").dofadr[0])
        mujoco.mj_resetData(model, data)
        data.qpos[slide_z] = height
        mujoco.mj_forward(model, data)
        # The speed *entering* the first contact and the speed *leaving* the
        # last substep of it.  A contact spans many substeps and the velocity
        # decays through them, so sampling it anywhere but the first substep
        # reports a restitution far above one.
        approach, departure, entered = 0.0, 0.0, False
        for _ in range(int(6.0 / PHYSICS_SUBSTEP)):
            previous = float(data.qvel[dof_z])
            mujoco.mj_step(model, data)
            if data.ncon > 0:
                if not entered and previous < -0.2:
                    approach = -previous
                    entered = True
            elif entered and data.qvel[dof_z] > 0.0:
                departure = float(data.qvel[dof_z])
                break
        if entered and approach > 0.0 and departure > 0.0:
            speeds_in.append(approach)
            speeds_out.append(departure)
    if not speeds_in:
        raise RuntimeError("no clean bounce was observed; check the scene geometry")
    ratio = np.asarray(speeds_out) / np.asarray(speeds_in)
    return {
        "restitution_median": float(np.median(ratio)),
        "restitution_min": float(ratio.min()),
        "restitution_max": float(ratio.max()),
        "samples": int(len(ratio)),
    }


def sample_episode_parameters(
    rng: np.random.Generator, spec: BounceSpec, gravity: float
) -> dict[str, float]:
    """Draw a launch that stays in frame and bounces at least ``min_contacts`` times.

    Rejection only, never resampling of a *rejected* quantity into an accepted
    one: the acceptance rate is recorded and the counterfactual protocol needs
    the training and evaluation distributions to differ by nothing else
    (Corollary 2's ``kappa = 1``).  Unlike the contact-free generator the
    feasible launch set has no closed form here -- a bounce changes the
    subsequent path -- so the trial is an actual simulation, which is why this
    is the expensive part of generation.
    """

    low_x, high_x, low_z, high_z = spec.safe_box()
    for attempt in range(int(spec.max_rejections)):
        parameters = {
            "gravity": float(gravity),
            "mass": float(rng.uniform(*spec.mass_range)),
            "x": float(rng.uniform(low_x, high_x)),
            "z": float(rng.uniform(*spec.launch_height_range)),
            "vx": float(rng.uniform(*spec.speed_x_range)),
            "vz": float(rng.uniform(*spec.launch_speed_z_range)),
            "angle": float(rng.uniform(*spec.angle_range)),
            "spin": float(rng.uniform(*spec.spin_range)),
        }
        path, contact = _dry_run(parameters, spec)
        if int(contact.sum()) < spec.min_contacts:
            continue
        if not (
            low_x <= path[:, 0].min() and path[:, 0].max() <= high_x
            and low_z <= path[:, 1].min() and path[:, 1].max() <= high_z
        ):
            continue
        parameters["attempts"] = float(attempt + 1)
        return parameters
    raise RuntimeError(
        f"no launch bounced and stayed in frame within {spec.max_rejections} draws; "
        "widen launch_height_range or lower min_contacts"
    )


def _dry_run(
    parameters: Mapping[str, float], spec: BounceSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate without rendering, to test a candidate launch cheaply."""

    import mujoco

    model = mujoco.MjModel.from_xml_string(model_xml(parameters, spec, 64))
    data = mujoco.MjData(model)
    slide_x = int(model.joint("slide_x").qposadr[0])
    slide_z = int(model.joint("slide_z").qposadr[0])
    dof_x = int(model.joint("slide_x").dofadr[0])
    dof_z = int(model.joint("slide_z").dofadr[0])
    mujoco.mj_resetData(model, data)
    data.qpos[slide_x] = float(parameters["x"])
    data.qpos[slide_z] = float(parameters["z"])
    data.qvel[dof_x] = float(parameters["vx"])
    data.qvel[dof_z] = float(parameters["vz"])
    mujoco.mj_forward(model, data)

    frames = int(spec.frames)
    path = np.empty((frames, 2), dtype=np.float64)
    contact = np.zeros(frames, dtype=bool)
    for frame in range(frames):
        path[frame] = (data.qpos[slide_x], data.qpos[slide_z])
        if frame + 1 < frames:
            touched = False
            for _ in range(spec.substeps):
                mujoco.mj_step(model, data)
                if data.ncon > 0:
                    touched = True
            contact[frame] = touched
    return path, contact


def counterfactual_hits_floor(
    position: np.ndarray,
    velocity: np.ndarray,
    gravity: np.ndarray,
    *,
    spec_resting_z: float,
    dt: float,
    horizon: int,
) -> np.ndarray:
    """Would the counterfactual free-fall path touch the floor inside the horizon?

    This is the filter option A needs, and it has to be evaluated under ``g'``
    rather than under ``g``.  A window that is contact-free in the data can
    still bounce in the counterfactual -- at ``g' = 9`` the object falls more
    than twice as far per step as at ``g = 4`` -- and the free-fall identity
    the constraint compares against does not hold on such a window either.
    Rejecting it is the same move the in-frame filter already makes, for the
    same reason, and like that one it rejects without resampling.

    ``position`` and ``velocity`` are the state at the last context frame,
    ``(B, 2)``; ``gravity`` is ``(B,)``.  Solving
    ``z0 + vz s - g' s^2 / 2 = resting_z`` for the first positive root and
    comparing it against ``horizon * dt`` is exact, because between contacts
    the counterfactual path *is* a parabola.
    """

    z0 = np.asarray(position, dtype=np.float64)[..., 1] - float(spec_resting_z)
    vz = np.asarray(velocity, dtype=np.float64)[..., 1]
    g = np.asarray(gravity, dtype=np.float64)
    span = float(horizon) * float(dt)

    already = z0 <= 0.0
    # z(s) = z0 + vz s - g s^2 / 2; the first positive root of a downward
    # parabola through a positive z0 is (vz + sqrt(vz^2 + 2 g z0)) / g.
    with np.errstate(invalid="ignore", divide="ignore"):
        discriminant = vz**2 + 2.0 * g * z0
        root = np.where(
            (discriminant >= 0.0) & (g > 0.0),
            (vz + np.sqrt(np.maximum(discriminant, 0.0))) / np.where(g > 0.0, g, 1.0),
            np.inf,
        )
    return already | (root <= span)


def window_statistics(
    contact: np.ndarray, *, history: int, horizon: int
) -> tuple[int, int, int]:
    """``(usable, with_bounce, total)`` over the windows the constraint sees.

    The constraint rolls out from ``history`` context frames over ``horizon``
    steps, so its unit is a ``history + horizon`` frame window, not the
    three-frame triple a second difference spans.  A window is usable when its
    *context* is contact-free: the implied velocity
    ``v = (p_t - p_{t-1})/dt + g dt/2`` that the position form needs is a
    free-fall quantity, and it is wrong across a bounce.  Whether the window's
    horizon bounces is a separate matter -- that is what the world model has to
    learn, and :func:`counterfactual_hits_floor` handles it at constraint time.
    """

    span = int(history) + int(horizon)
    usable = bounce = total = 0
    for start in range(len(contact) - span + 1):
        total += 1
        usable += int(not contact[start : start + history].any())
        bounce += int(contact[start + history : start + span].any())
    return usable, bounce, total


def _generate_one(payload):
    """Sample, simulate and render one episode.  Module level so it pickles."""

    index, split, gravity, spec, image_size, seed, history, horizon = payload
    rng = np.random.default_rng(int(seed) + 7919 * int(index))
    parameters = sample_episode_parameters(rng, spec, gravity)
    attempts = parameters.pop("attempts")
    pixels, action, state, contact, diagnostics = simulate_episode(
        parameters, spec, image_size=image_size
    )
    usable, bounce, total = window_statistics(contact, history=history, horizon=horizon)
    diagnostics.update(
        {"usable_windows": usable, "bounce_windows": bounce, "window_count": total,
         "attempts": attempts, "mass": parameters["mass"]}
    )
    return index, split, float(gravity), pixels, action, state, contact, diagnostics


def generate_bounce_dataset(
    output_dir,
    *,
    spec: BounceSpec | None = None,
    image_size: int = 128,
    history: int = 8,
    horizon: int = 6,
    train_episodes: int = 16000,
    test_gravities=TEST_GRAVITIES,
    test_episodes_per_gravity: int = 32,
    train_mean: float = 4.0,
    train_std: float = 0.0,
    heldout_train_gravity_episodes: int = 800,
    seed: int = 20260101,
    image_format: str = "png",
    workers: int = 0,
) -> dict:
    """Render the bouncing dataset and record what the constraint needs to know.

    The manifest carries three things the contact-free generator has no need
    for: the measured restitution, the share of windows that are usable by the
    constraint, and the free-fall residual on those windows.  All three are
    properties of the data rather than of a run, and all three were the
    quantities that decided whether this experiment was worth doing.
    """

    import json
    import os
    import socket
    from pathlib import Path

    from gwm.freefall_data import (
        EpisodeRecord,
        gravity_assignments,
        write_lance_episodes,
    )

    spec = spec or BounceSpec()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    restitution = measure_restitution(spec, float(train_mean))
    if restitution["restitution_max"] >= 1.0:
        raise RuntimeError(
            f"the scene gains energy at a bounce (e={restitution['restitution_max']:.3f}); "
            "pick a solref that dissipates"
        )

    assignments = gravity_assignments(
        train_episodes=train_episodes,
        test_gravities=test_gravities,
        test_episodes_per_gravity=test_episodes_per_gravity,
        train_mean=train_mean,
        train_std=train_std,
        seed=seed,
        heldout_train_gravity_episodes=heldout_train_gravity_episodes,
    )

    usable_windows = bounce_windows = window_count = 0
    residual, contact_residual, attempts = [], [], []

    def episodes():
        nonlocal usable_windows, bounce_windows, window_count
        payloads = [
            (index, split, gravity, spec, image_size, seed, history, horizon)
            for index, split, gravity in assignments
        ]
        if int(workers) > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            # Spawn, not the Linux default fork: this generator is consumed
            # *inside* the Lance writer's record-batch stream, so forking here
            # would copy a live Lance handle into every worker, which Lance
            # warns is not fork-safe.  Spawn also hands each worker a clean EGL
            # context instead of an inherited one.
            with ProcessPoolExecutor(
                max_workers=int(workers),
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                # Ordered map: the Lance writer needs contiguous episode indices,
                # and each worker builds its own MuJoCo model and renderer.
                results = pool.map(_generate_one, payloads, chunksize=4)
                yield from _collect(results)
        else:
            yield from _collect(_generate_one(payload) for payload in payloads)

    def _collect(results):
        nonlocal usable_windows, bounce_windows, window_count
        for index, split, gravity, pixels, action, state, contact, diagnostics in results:
            usable_windows += int(diagnostics["usable_windows"])
            bounce_windows += int(diagnostics["bounce_windows"])
            window_count += int(diagnostics["window_count"])
            attempts.append(diagnostics["attempts"])
            if diagnostics["free_fall_residual_max"] == diagnostics["free_fall_residual_max"]:
                residual.append(diagnostics["free_fall_residual_max"])
            if (
                diagnostics["contact_window_residual_max"]
                == diagnostics["contact_window_residual_max"]
            ):
                contact_residual.append(diagnostics["contact_window_residual_max"])
            physics = np.asarray(
                (gravity, diagnostics["mass"], spec.half_size, spec.half_size,
                 SHAPE_IDS[spec.shape], spec.dt, spec.view_height, spec.z_center),
                dtype=np.float32,
            )
            yield EpisodeRecord(
                episode_index=index,
                split=split,
                pixels=pixels,
                state=state,
                action=action,
                gravity=float(gravity),
                physics=physics,
                source_episode_index=index,
                contact=contact,
            )

    pixels_per_unit = image_size / spec.view_height
    write_lance_episodes(
        episodes(),
        output_dir / "data.lance",
        task=TASK,
        frames_per_episode=spec.frames,
        image_size=image_size,
        image_format=image_format,
        extra_metadata={"dt": spec.dt, "pixels_per_unit": pixels_per_unit},
    )

    low_x, high_x, low_z, high_z = spec.safe_box()
    manifest = {
        # The fields the upstream reader and the latent cache require.  The
        # dataset is unusable without "dataset", and the constraint calibrates
        # eps on the episodes named by "heldout_train_gravity_source_range" --
        # using the whole held-out split instead raises eps by a factor 1.7 and
        # the constraint then never activates (see RESULTS.md section 2).
        "schema_version": 1,
        "dataset": "data.lance",
        "generator": "gwm.bounce_data",
        "shape": spec.shape,
        "state_names": list(STATE_NAMES),
        "frames_per_episode": spec.frames,
        "image_format": image_format,
        "pixels_per_world_unit": image_size / spec.view_height,
        "safe_box": [low_x, high_x, low_z, high_z],
        "scene": {
            "dt": spec.dt, "frames": spec.frames, "view_height": spec.view_height,
            "z_center": spec.z_center, "shape": spec.shape, "half_size": spec.half_size,
            "margin": spec.margin, "floor_z": spec.floor_z,
            "floor_thickness": spec.floor_thickness, "resting_z": spec.resting_z,
        },
        "gravity_range": [float(min(test_gravities)), float(max(test_gravities))],
        "test_gravities": [float(g) for g in test_gravities],
        "test_episodes": sum(item[1] == "test" for item in assignments),
        "heldout_train_gravity_episodes": int(heldout_train_gravity_episodes),
        "heldout_train_gravity_source_range": [
            int(train_episodes),
            int(train_episodes) + int(heldout_train_gravity_episodes) - 1,
        ],
        "task": TASK,
        "dt": spec.dt,
        "frames": spec.frames,
        "image_size": image_size,
        "view_height": spec.view_height,
        "pixels_per_unit": pixels_per_unit,
        "floor_z": spec.floor_z,
        "resting_z": spec.resting_z,
        "half_size": spec.half_size,
        "episodes": len(assignments),
        "train_episodes": train_episodes,
        "train_gravity_mean": train_mean,
        "train_gravity_std": train_std,
        "action_names": list(action_names(False)),
        "physics_substep": PHYSICS_SUBSTEP,
        "contact_solref": CONTACT_SOLREF,
        "contact_solimp": CONTACT_SOLIMP,
        "restitution": restitution,
        # What option A actually gets to use.
        "history": history,
        "horizon": horizon,
        "window_count": window_count,
        "constraint_usable_windows": usable_windows,
        "constraint_usable_fraction": (usable_windows / window_count) if window_count else float("nan"),
        "windows_with_a_bounce_in_the_horizon": bounce_windows,
        "bounce_window_fraction": (bounce_windows / window_count) if window_count else float("nan"),
        "free_fall_residual_max": float(np.max(residual)) if residual else float("nan"),
        "free_fall_residual_median": float(np.median(residual)) if residual else float("nan"),
        "contact_window_residual_median": (
            float(np.median(contact_residual)) if contact_residual else float("nan")
        ),
        "gravity_step_pixels": float(train_mean) * spec.dt**2 * pixels_per_unit,
        "launch_acceptance_rate": float(1.0 / np.mean(attempts)) if attempts else float("nan"),
        "seed": seed,
        # Where the pixels came from.  The same scene rendered through two EGL
        # implementations is bit-identical in physics and *not* in pixels: a
        # login node and a GPU compute node differed by 91 non-white pixels out
        # of 35k on one episode, about 0.26%, while the state hash matched
        # exactly.  The position probe works at 0.1-0.2 px, so mixing renderers
        # across a dataset or across a comparison introduces a bias larger than
        # the signal.  Recording it makes a mismatch visible instead of silent.
        "render_host": socket.gethostname(),
        "render_backend": os.environ.get("MUJOCO_GL", "<unset>"),
    }
    (output_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
