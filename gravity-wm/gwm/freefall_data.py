"""Contact-free planar projectile generator for the gravity-constraint study.

The planar datasets of Semigroup-JEPA put the object in a closed box, so its
windows contain contacts and the free-fall identity of Lemma 1 does not hold on
them.  They are also sampled too finely for the constraint to carry signal: at
``fps=16`` and an orthographic view ten world units tall, one step of gravity
displaces the object by ``g dt**2 = 0.0156`` units, i.e. 0.19 pixel at 128x128,
which is far below any achievable position-probe accuracy.

This generator produces the regime the theory assumes instead:

* no walls and no floor, so every window of every episode is contact-free and
  gravity is the only force acting;
* a frame interval ``dt`` chosen so that ``g dt**2`` spans a few pixels, which
  is what makes the finite-difference acceleration measurable;
* a rejection-sampled launch that keeps the whole parabola inside the
  orthographic view for every gravity, including the out-of-distribution ones,
  so the position probe never has to extrapolate off-frame;
* lossless PNG frames, because JPEG ringing is a sub-pixel position error and
  the constraint is sensitive to exactly that.

The Lance schema, action layout ``(impulse_x, impulse_z, g)`` and state layout
``(x, z, vx, vz, theta, omega, anchor_x, anchor_z)`` match the planar tasks of
``sg_jepa`` so that its trainer, dataset reader and probe code apply unchanged.
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TASKS = {"ball": "freefall_ball", "square": "freefall_square"}
ACTION_NAMES = ("impulse_x", "impulse_z", "g")
# The two-dimensional experiment needs the horizontal acceleration to reach the
# predictor, which it can only do as an action coordinate.  It is appended
# rather than inserted so that the first three keep their indices and every
# existing cache, checkpoint and `gravity_index=2` stays correct.
ACTION_NAMES_WITH_ACCEL_X = ACTION_NAMES + ("a_x",)


def action_names(with_accel_x: bool = False) -> tuple[str, ...]:
    """The action layout, with or without the horizontal-acceleration slot."""

    return ACTION_NAMES_WITH_ACCEL_X if with_accel_x else ACTION_NAMES
STATE_NAMES = ("x", "z", "vx", "vz", "theta", "omega", "anchor_x", "anchor_z")
PHYSICS_NAMES = ("g", "mass", "size_x", "size_z", "shape_id", "dt", "view_height", "z_center")
SPLIT_TO_ID = {"train": 0, "test": 1}

BASE_SEED = 20261101
PHYSICS_TIMESTEP = 1.0 / 1000.0
SHAPE_IDS = {"ball": 2.0, "square": 1.0}


@dataclass(frozen=True)
class SceneSpec:
    """Geometry and sampling ranges of the contact-free projectile scene.

    ``dt`` is the interval between rendered frames; the simulator substeps at
    ``PHYSICS_TIMESTEP``.  ``view_height`` is the orthographic extent in world
    units, so one pixel is ``view_height / image_size`` units and one step of
    gravity moves the object ``g * dt**2`` units.
    """

    dt: float = 0.16
    frames: int = 16
    view_height: float = 13.0
    z_center: float = 6.5
    shape: str = "ball"
    half_size: float = 0.80
    margin: float = 0.45
    mass_range: tuple[float, float] = (0.5, 2.0)
    spin_range: tuple[float, float] = (-0.6, 0.6)
    angle_range: tuple[float, float] = (-math.pi, math.pi)
    speed_x_range: tuple[float, float] = (-1.6, 1.6)
    rise_fraction_range: tuple[float, float] = (0.70, 1.30)
    fill_fraction: float = 0.88
    max_rejections: int = 400

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.frames < 3:
            raise ValueError("dt must be positive and frames must be at least three")
        if self.view_height <= 0.0 or self.half_size <= 0.0:
            raise ValueError("view_height and half_size must be positive")
        if self.margin < 0.0:
            raise ValueError("margin must be non-negative")
        if self.max_rejections < 1:
            raise ValueError("max_rejections must be positive")
        if not 0.0 < self.fill_fraction <= 1.0:
            raise ValueError("fill_fraction must lie in (0, 1]")
        if self.shape not in TASKS:
            raise ValueError(f"shape must be one of {sorted(TASKS)}")

    @property
    def span(self) -> float:
        """Elapsed time between the first and the last rendered frame."""

        return (self.frames - 1) * self.dt

    @property
    def half_view(self) -> float:
        return 0.5 * self.view_height

    def gravity_step_pixels(self, gravity: float, image_size: int) -> float:
        """Pixels by which one step of ``gravity`` displaces the object."""

        return abs(gravity) * self.dt**2 / self.view_height * float(image_size)

    def safe_box(self) -> tuple[float, float, float, float]:
        """Axis-aligned world box the object centre must stay inside."""

        corner = 1.0 if self.shape == "ball" else math.sqrt(2.0)
        reach = self.half_view - self.half_size * corner - self.margin
        if reach <= 0.0:
            raise ValueError("view_height is too small for the object and margin")
        return (-reach, reach, self.z_center - reach, self.z_center + reach)

    def vertical_excursion(self, gravity: float, rise: float) -> float:
        """Vertical extent swept by the launch ``vz = rise * g * span / 2``.

        With ``z(s) = vz s - g s^2 / 2`` on ``s in [0, span]`` the swept extent
        is ``|g| span^2 max(rise, 2 - rise)^2 / 8`` for ``0 < rise < 2``: it is
        minimised by the symmetric parabola ``rise = 1``, which returns the
        object to its launch height at the final frame, and grows for launches
        that either fall through the start height or do not reach the apex.
        """

        factor = max(float(rise), 2.0 - float(rise))
        return abs(float(gravity)) * self.span**2 * factor**2 / 8.0

    def feasible_rise_range(self, gravity: float, *, height: float | None = None) -> tuple[float, float]:
        """Rise factors whose excursion fits ``height`` world units.

        Inverting :meth:`vertical_excursion` gives ``max(rise, 2 - rise) <= M``
        with ``M = sqrt(8 height / (|g| span^2))``, i.e. ``rise in [2-M, M]``,
        which is non-empty exactly when ``M >= 1``.
        """

        if height is None:
            left, right, bottom, top = self.safe_box()
            height = (top - bottom) * self.fill_fraction
        low, high = self.rise_fraction_range
        if abs(float(gravity)) < 1.0e-12:
            return (low, high)
        bound = math.sqrt(8.0 * float(height) / (abs(float(gravity)) * self.span**2))
        if bound < 1.0:
            raise ValueError(
                f"no launch fits the view at gravity={gravity}: the shortest possible "
                f"excursion is {self.vertical_excursion(gravity, 1.0):.2f} world units but "
                f"only {height:.2f} are available; shorten the episode, lower dt, or widen view_height"
            )
        return (max(low, 2.0 - bound), min(high, bound))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def trajectory(
    *,
    x0: float,
    z0: float,
    vx: float,
    vz: float,
    gravity: float,
    dt: float,
    frames: int,
    accel_x: float = 0.0,
) -> np.ndarray:
    """Analytic path under a uniform acceleration, ``(frames, 2)`` in ``(x, z)``.

    Gravity is the MuJoCo scalar, so the acceleration is ``(accel_x, -gravity)``
    and ``accel_x = 0`` is free fall.  A non-zero ``accel_x`` is the
    two-dimensional experiment: the counterfactual becomes an acceleration
    *vector* rather than a magnitude, and the constraint machinery already
    works in those terms (see ``gwm.constraint.acceleration_vector``).
    """

    times = np.arange(int(frames), dtype=np.float64) * float(dt)
    return np.stack(
        (
            float(x0) + float(vx) * times + 0.5 * float(accel_x) * times**2,
            float(z0) + float(vz) * times - 0.5 * float(gravity) * times**2,
        ),
        axis=-1,
    )


def _fits(path: np.ndarray, spec: SceneSpec) -> bool:
    left, right, bottom, top = spec.safe_box()
    return bool(
        path[:, 0].min() >= left
        and path[:, 0].max() <= right
        and path[:, 1].min() >= bottom
        and path[:, 1].max() <= top
    )


def shared_rise_range(
    spec: SceneSpec, gravities: Sequence[float]
) -> tuple[float, float]:
    """Rise window feasible for every gravity in ``gravities``.

    All episodes are drawn from one launch distribution so that the shape of
    the launch carries no information about the episode gravity.  Without this
    the feasible window narrows as ``|g|`` grows, and a held-out model could
    read gravity off the launch instead of off the motion.
    """

    lows, highs = [], []
    for gravity in gravities:
        low, high = spec.feasible_rise_range(gravity)
        lows.append(low)
        highs.append(high)
    low, high = max(lows), min(highs)
    if low >= high:
        raise ValueError(
            "no launch window is feasible for every requested gravity; "
            f"got [{low:.3f}, {high:.3f}] over {len(gravities)} values"
        )
    return (low, high)


def sample_episode_parameters(
    gravity: float,
    seed_sequence: np.random.Generator | int,
    spec: SceneSpec,
    *,
    rise_range: tuple[float, float] | None = None,
    accel_x: float = 0.0,
) -> dict[str, float]:
    """Draw one in-frame, contact-free launch for an arbitrary gravity.

    The launch is ``vz = rise * g * span / 2``; ``rise = 1`` is the symmetric
    parabola that returns to the launch height at the final frame.  ``rise`` is
    confined analytically to a window whose excursion fits the view
    (:meth:`SceneSpec.feasible_rise_range`), and ``vx`` likewise, so the path
    fits by construction for any gravity.  The parabola is then centred in
    :meth:`SceneSpec.safe_box` and jittered inside the remaining slack; the
    rejection loop is only a backstop against rounding at the boundary.
    """

    rng = (
        seed_sequence
        if isinstance(seed_sequence, np.random.Generator)
        else np.random.default_rng(int(seed_sequence))
    )
    left, right, bottom, top = spec.safe_box()
    height = (top - bottom) * spec.fill_fraction
    width = (right - left) * spec.fill_fraction
    rise_low, rise_high = (
        rise_range if rise_range is not None else spec.feasible_rise_range(gravity, height=height)
    )
    if rise_low > rise_high:
        raise ValueError(f"empty rise range [{rise_low}, {rise_high}]")
    # A horizontal acceleration adds |a_x| span^2 / 2 to the x-excursion, so the
    # speed budget has to leave room for it or the rejection loop -- which is
    # only meant to be a rounding backstop -- would start doing real work and
    # the launch distribution would stop being independent of the episode.
    horizontal_drift = 0.5 * abs(float(accel_x)) * spec.span**2
    speed_high = min(max(spec.speed_x_range), max(width - horizontal_drift, 0.0) / spec.span)

    for attempt in range(spec.max_rejections):
        shrink = 1.0 if attempt < spec.max_rejections // 2 else 0.8
        rise = float(rng.uniform(rise_low, rise_high))
        vz = 0.5 * float(gravity) * spec.span * rise
        vx = float(rng.uniform(-speed_high, speed_high)) * shrink
        spinning = spec.shape != "ball"
        angle = float(rng.uniform(*spec.angle_range)) if spinning else 0.0
        spin = float(rng.uniform(*spec.spin_range)) if spinning else 0.0
        mass = float(rng.uniform(*spec.mass_range))
        path = trajectory(
            x0=0.0, z0=0.0, vx=vx, vz=vz, gravity=gravity, dt=spec.dt,
            frames=spec.frames, accel_x=accel_x,
        )
        slack_x = max(0.0, (right - left) - (path[:, 0].max() - path[:, 0].min()))
        slack_z = max(0.0, (top - bottom) - (path[:, 1].max() - path[:, 1].min()))
        jitter_x = float(rng.uniform(-1.0, 1.0)) * 0.5 * slack_x * shrink
        jitter_z = float(rng.uniform(-1.0, 1.0)) * 0.5 * slack_z * shrink
        x0 = 0.5 * (left + right) - 0.5 * (path[:, 0].min() + path[:, 0].max()) + jitter_x
        z0 = 0.5 * (bottom + top) - 0.5 * (path[:, 1].min() + path[:, 1].max()) + jitter_z
        path = trajectory(
            x0=x0, z0=z0, vx=vx, vz=vz, gravity=gravity, dt=spec.dt,
            frames=spec.frames, accel_x=accel_x,
        )
        if _fits(path, spec):
            return {
                "x": x0,
                "z": z0,
                "vx": vx,
                "vz": vz,
                "gravity": float(gravity),
                "accel_x": float(accel_x),
                "mass": mass,
                "angle": angle,
                "spin": spin,
                "rise": rise,
                "attempts": float(attempt + 1),
            }
    raise RuntimeError(
        f"could not place an in-frame trajectory for gravity={gravity} after "
        f"{spec.max_rejections} draws; widen view_height or shorten the episode"
    )


def _format(values: Any) -> str:
    if np.isscalar(values):
        return f"{float(values):.10g}"
    return " ".join(f"{float(value):.10g}" for value in values)


def _model_xml(parameters: Mapping[str, float], spec: SceneSpec, image_size: int) -> str:
    """A single free body, a white backdrop and an orthographic camera.

    The scene deliberately contains no collidable geometry other than the
    object itself, so ``mj_step`` can never produce a contact and every window
    satisfies the contact-free hypothesis of Lemma 1.
    """

    root = ET.Element("mujoco", model="freefall_projectile")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(
        root,
        "option",
        timestep=_format(PHYSICS_TIMESTEP),
        gravity=_format(
            (
                float(parameters.get("accel_x", 0.0)),
                0.0,
                -float(parameters["gravity"]),
            )
        ),
        integrator="implicitfast",
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
        contype="0",
        conaffinity="0",
    )
    return ET.tostring(root, encoding="unicode")


def render_episode(
    parameters: Mapping[str, float],
    spec: SceneSpec,
    *,
    image_size: int,
    with_accel_x: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Render one episode from the closed-form free-fall solution.

    Contact-free motion under uniform gravity has an exact solution, so the
    pose at every frame is *set* from :func:`trajectory` rather than obtained
    by integrating.  This matters here: the constraint is a second difference
    of positions, and an integrator's drift of a few times ``1e-3`` world units
    is a few percent of the whole one-step gravity displacement ``g dt**2``
    this study is trying to measure.  Setting the pose makes both the physics
    and the position labels exact, and removes the substep cost.

    :func:`verify_free_fall_against_integrator` cross-checks the scene against
    ``mj_step`` to confirm that the rendered motion is the motion MuJoCo would
    itself produce in this contact-free scene.
    """

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_string(_model_xml(parameters, spec, image_size))
    data = mujoco.MjData(model)
    slide_x = int(model.joint("slide_x").qposadr[0])
    slide_z = int(model.joint("slide_z").qposadr[0])
    hinge = int(model.joint("hinge_y").qposadr[0])
    dof_x = int(model.joint("slide_x").dofadr[0])
    dof_z = int(model.joint("slide_z").dofadr[0])
    dof_hinge = int(model.joint("hinge_y").dofadr[0])

    frames = int(spec.frames)
    accel_x = float(parameters.get("accel_x", 0.0))
    path = trajectory(
        x0=parameters["x"],
        z0=parameters["z"],
        vx=parameters["vx"],
        vz=parameters["vz"],
        gravity=parameters["gravity"],
        dt=spec.dt,
        frames=frames,
        accel_x=accel_x,
    )
    times = np.arange(frames, dtype=np.float64) * float(spec.dt)
    velocity_z = float(parameters["vz"]) - float(parameters["gravity"]) * times
    velocity_x = float(parameters["vx"]) + accel_x * times
    angle = float(parameters["angle"]) + float(parameters["spin"]) * times

    pixels = np.empty((frames, image_size, image_size, 3), dtype=np.uint8)
    state = np.empty((frames, len(STATE_NAMES)), dtype=np.float32)
    names = action_names(with_accel_x)
    action = np.zeros((frames, len(names)), dtype=np.float32)
    action[:, 2] = parameters["gravity"]
    if with_accel_x:
        action[:, 3] = accel_x
    action[0, :2] = parameters["mass"] * np.asarray(
        (parameters["vx"], parameters["vz"]), dtype=np.float32
    )

    renderer = mujoco.Renderer(model, height=image_size, width=image_size)
    try:
        for frame in range(frames):
            data.qpos[slide_x] = path[frame, 0]
            data.qpos[slide_z] = path[frame, 1]
            data.qpos[hinge] = angle[frame]
            data.qvel[dof_x] = velocity_x[frame]
            data.qvel[dof_z] = velocity_z[frame]
            data.qvel[dof_hinge] = parameters["spin"]
            mujoco.mj_forward(model, data)
            if data.ncon:
                raise RuntimeError("the contact-free scene produced a contact")
            state[frame] = (
                path[frame, 0],
                path[frame, 1],
                velocity_x[frame],
                velocity_z[frame],
                angle[frame],
                parameters["spin"],
                0.0,
                spec.z_center,
            )
            renderer.update_scene(data, camera="render_camera")
            renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 0
            pixels[frame] = renderer.render()
    finally:
        renderer.close()

    # The labels are the analytic solution, so the only residual worth
    # reporting is how exactly the discrete second difference reproduces
    # ``g dt**2``; it bounds the ground-truth error of the constraint target.
    second = path[2:] - 2.0 * path[1:-1] + path[:-2]
    target = np.asarray((accel_x, -float(parameters["gravity"])), dtype=np.float64) * spec.dt**2
    diagnostics = {
        "free_fall_residual_max": float(np.abs(second - target).max()),
        "x_min": float(path[:, 0].min()),
        "x_max": float(path[:, 0].max()),
        "z_min": float(path[:, 1].min()),
        "z_max": float(path[:, 1].max()),
    }
    return pixels, action, state, diagnostics


def verify_free_fall_against_integrator(
    parameters: Mapping[str, float],
    spec: SceneSpec,
    *,
    image_size: int = 64,
) -> dict[str, float]:
    """Integrate the same scene with ``mj_step`` and compare with the analytic path.

    Used as a one-off audit that the generated scene really is contact-free
    uniform-gravity motion, independently of the kinematic rendering path.
    """

    import mujoco

    model = mujoco.MjModel.from_xml_string(_model_xml(parameters, spec, image_size))
    data = mujoco.MjData(model)
    substeps = int(round(spec.dt / PHYSICS_TIMESTEP))
    if abs(substeps * PHYSICS_TIMESTEP - spec.dt) > 1e-12:
        raise ValueError(f"dt={spec.dt} is not a multiple of the physics timestep")
    data.qpos[int(model.joint("slide_x").qposadr[0])] = parameters["x"]
    data.qpos[int(model.joint("slide_z").qposadr[0])] = parameters["z"]
    data.qpos[int(model.joint("hinge_y").qposadr[0])] = parameters["angle"]
    data.qvel[int(model.joint("slide_x").dofadr[0])] = parameters["vx"]
    data.qvel[int(model.joint("slide_z").dofadr[0])] = parameters["vz"]
    data.qvel[int(model.joint("hinge_y").dofadr[0])] = parameters["spin"]
    mujoco.mj_forward(model, data)
    observed = np.empty((spec.frames, 2), dtype=np.float64)
    contacts = 0
    for frame in range(int(spec.frames)):
        if frame:
            for _ in range(substeps):
                mujoco.mj_step(model, data)
        contacts += int(data.ncon)
        body = data.body("moving_object")
        observed[frame] = (body.xpos[0], body.xpos[2])
    analytic = trajectory(
        x0=parameters["x"],
        z0=parameters["z"],
        vx=parameters["vx"],
        vz=parameters["vz"],
        gravity=parameters["gravity"],
        dt=spec.dt,
        frames=spec.frames,
        accel_x=float(parameters.get("accel_x", 0.0)),
    )
    return {
        "contacts": float(contacts),
        "max_position_gap": float(np.abs(observed - analytic).max()),
        "max_second_difference_gap": float(
            np.abs(
                (observed[2:] - 2.0 * observed[1:-1] + observed[:-2])
                - (analytic[2:] - 2.0 * analytic[1:-1] + analytic[:-2])
            ).max()
        ),
    }


def _encode(frame: np.ndarray, image_format: str) -> bytes:
    buffer = BytesIO()
    image = Image.fromarray(frame, mode="RGB")
    if image_format == "png":
        image.save(buffer, format="PNG", optimize=False, compress_level=4)
    elif image_format == "jpeg":
        image.save(buffer, format="JPEG", quality=95)
    else:
        raise ValueError("image_format must be png or jpeg")
    return buffer.getvalue()


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    split: str
    pixels: np.ndarray
    state: np.ndarray
    action: np.ndarray
    gravity: float
    physics: np.ndarray
    source_episode_index: int
    # Per-frame contact flag, for the generators that admit contacts.  The
    # free-fall identity of Lemma 1 does not hold across a contact, so a window
    # that spans one must be excluded from the constraint rather than counted
    # as a violation -- measured on the bouncing scene, such a window scores
    # 2.5-5.1 px against a one-step gravity displacement of 1.008 px, so
    # including them would swamp the quantity the constraint is about.  ``None``
    # for the contact-free generator, whose scene cannot produce one.
    contact: np.ndarray | None = None


def write_lance_episodes(
    episodes: Iterable[EpisodeRecord],
    output: str | Path,
    *,
    task: str,
    frames_per_episode: int,
    image_size: int,
    image_format: str,
    extra_metadata: Mapping[str, Any],
    names: tuple[str, ...] = ACTION_NAMES,
) -> Path:
    """Stream episodes into the ``sg_jepa`` Lance contract.

    Mirrors ``data_generation.common.lance.write_lance_episodes`` but allows a
    lossless image codec and records the frame interval in the dataset
    metadata, which the constraint needs and the upstream writer does not keep.
    """

    import lance
    import pyarrow as pa

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "task_name": task,
        "frames_per_episode": int(frames_per_episode),
        "pixels_shape": [image_size, image_size, 3],
        "pixels_dtype": "uint8",
        "pixels_encoding": f"{image_format}_rgb",
        "state_schema": list(STATE_NAMES),
        "action_schema": list(names),
        "phys_schema": list(PHYSICS_NAMES),
        "split_to_id": dict(SPLIT_TO_ID),
    }
    metadata.update(extra_metadata)
    encoded_metadata = {
        str(key).encode(): (
            value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        ).encode()
        for key, value in metadata.items()
    }
    schema = pa.schema(
        [
            pa.field("episode_idx", pa.int32(), nullable=False),
            pa.field("step_idx", pa.int32(), nullable=False),
            pa.field("split_id", pa.int8(), nullable=False),
            pa.field("pixels", pa.binary(), nullable=False),
            pa.field("state", pa.list_(pa.float32(), len(STATE_NAMES)), nullable=False),
            pa.field("action", pa.list_(pa.float32(), len(names)), nullable=False),
            pa.field("reward", pa.float32(), nullable=False),
            pa.field("phys", pa.list_(pa.float32(), len(PHYSICS_NAMES)), nullable=False),
            pa.field("contact", pa.bool_(), nullable=False),
            pa.field("gravity", pa.float32(), nullable=False),
            pa.field("source_episode_index", pa.int32(), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("episode_metadata", pa.string(), nullable=False),
            pa.field("success", pa.bool_(), nullable=False),
        ],
        metadata=encoded_metadata,
    )

    def batches():
        for episode in episodes:
            count = int(episode.pixels.shape[0])
            if count != int(frames_per_episode):
                raise ValueError("episode length differs from frames_per_episode")
            if episode.state.shape != (count, len(STATE_NAMES)):
                raise ValueError("state shape differs from schema")
            if episode.action.shape != (count, len(names)):
                raise ValueError("action shape differs from schema")
            physics = np.repeat(
                np.asarray(episode.physics, dtype=np.float32).reshape(1, -1), count, axis=0
            )
            episode_metadata = json.dumps(
                {
                    "task": task,
                    "episode_index": episode.episode_index,
                    "source_episode_index": episode.source_episode_index,
                    "split": episode.split,
                    "gravity": float(episode.gravity),
                },
                sort_keys=True,
            )
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array([episode.episode_index] * count, type=pa.int32()),
                    pa.array(np.arange(count), type=pa.int32()),
                    pa.array([SPLIT_TO_ID[episode.split]] * count, type=pa.int8()),
                    pa.array(
                        [_encode(frame, image_format) for frame in episode.pixels],
                        type=pa.binary(),
                    ),
                    pa.array(episode.state.tolist(), type=schema.field("state").type),
                    pa.array(episode.action.tolist(), type=schema.field("action").type),
                    pa.array(np.zeros(count, dtype=np.float32), type=pa.float32()),
                    pa.array(physics.tolist(), type=schema.field("phys").type),
                    pa.array(
                        np.zeros(count, dtype=bool)
                        if episode.contact is None
                        else np.asarray(episode.contact, dtype=bool),
                        type=pa.bool_(),
                    ),
                    pa.array([float(episode.gravity)] * count, type=pa.float32()),
                    pa.array([episode.source_episode_index] * count, type=pa.int32()),
                    pa.array([task] * count, type=pa.string()),
                    pa.array([episode_metadata] * count, type=pa.string()),
                    pa.array([False] * count, type=pa.bool_()),
                ],
                schema=schema,
            )

    reader = pa.RecordBatchReader.from_batches(schema, batches())
    lance.write_dataset(reader, str(output), mode="create")
    return output


def gravity_assignments(
    *,
    train_episodes: int,
    test_gravities: Sequence[float],
    test_episodes_per_gravity: int,
    train_mean: float,
    train_std: float,
    seed: int,
    heldout_train_gravity_episodes: int = 0,
) -> list[tuple[int, str, float]]:
    """Interleave narrow-range training gravities with a wide held-out grid.

    Training gravity follows the Semigroup-JEPA planar protocol,
    ``g ~ N(train_mean, train_std**2)`` clipped away from zero, so the method is
    evaluated in the same narrow-training / wide-testing regime.

    The held-out split has two parts, because the two evaluations in the theory
    note need different things.  ``heldout_train_gravity_episodes`` are drawn
    from the *training* gravity distribution: these supply the real windows of
    the counterfactual protocol of Theorem 2, in which the context comes from
    the training distribution and only the conditioning gravity moves.  The
    ``test_gravities`` sweep supplies whole episodes whose context already
    evolves under an out-of-range gravity, which is the harder setting of
    Remark 8(iii) and is not covered by Corollary 2.
    """

    rng = np.random.default_rng(int(seed))
    assignments: list[tuple[int, str, float]] = []
    for index in range(int(train_episodes)):
        gravity = float(max(rng.normal(float(train_mean), float(train_std)), 0.1))
        assignments.append((index, "train", gravity))
    offset = int(train_episodes)
    for index in range(int(heldout_train_gravity_episodes)):
        gravity = float(max(rng.normal(float(train_mean), float(train_std)), 0.1))
        assignments.append((offset + index, "test", gravity))
    offset += int(heldout_train_gravity_episodes)
    for repeat in range(int(test_episodes_per_gravity)):
        for position, gravity in enumerate(test_gravities):
            index = offset + repeat * len(test_gravities) + position
            assignments.append((index, "test", float(gravity)))
    return assignments


def _render_one(
    payload: tuple[Mapping[str, float], SceneSpec, int, bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Module-level entry point so rendering can be mapped over processes."""

    parameters, spec, image_size, with_accel_x = payload
    return render_episode(
        parameters, spec, image_size=image_size, with_accel_x=with_accel_x
    )


def _render_stream(
    payloads: Sequence[tuple[Mapping[str, float], SceneSpec, int, bool]],
    workers: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]]:
    """Render in episode order, optionally across processes.

    Order is preserved because the Lance writer consumes a single ordered
    stream and the dataset reader requires contiguous, ordered episodes.  Each
    worker builds its own MuJoCo model and renderer, so each gets an
    independent EGL context.
    """

    if workers <= 1:
        for payload in payloads:
            yield _render_one(payload)
        return
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        yield from pool.map(_render_one, payloads, chunksize=4)


def generate_freefall_dataset(
    output_dir: str | Path,
    *,
    spec: SceneSpec | None = None,
    image_size: int = 128,
    train_episodes: int = 1200,
    test_gravities: Sequence[float] = tuple(np.arange(1.0, 9.01, 0.5)),
    test_episodes_per_gravity: int = 8,
    train_mean: float = 4.0,
    train_std: float = 0.5,
    heldout_train_gravity_episodes: int = 400,
    seed: int = BASE_SEED,
    image_format: str = "png",
    free_fall_tolerance: float = 1.0e-9,
    workers: int = 0,
    accel_x: float = 0.0,
    with_accel_x: bool = False,
) -> dict[str, Any]:
    """Render the contact-free projectile dataset and verify it is free fall.

    ``with_accel_x`` appends the horizontal acceleration to the action layout so
    the predictor can be conditioned on it; ``accel_x`` is its value in every
    training episode, and the default of zero is the regime the two-dimensional
    experiment wants -- the model never sees a horizontal acceleration, so only
    the constraint can teach the conditioning path what one does.
    """

    spec = spec or SceneSpec()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    assignments = gravity_assignments(
        train_episodes=train_episodes,
        test_gravities=test_gravities,
        test_episodes_per_gravity=test_episodes_per_gravity,
        train_mean=train_mean,
        train_std=train_std,
        seed=seed,
        heldout_train_gravity_episodes=heldout_train_gravity_episodes,
    )
    all_gravities = [value for _, _, value in assignments]
    rise_range = shared_rise_range(spec, sorted({round(value, 6) for value in all_gravities}))
    output_dir.mkdir(parents=True)
    worst = {"free_fall_residual_max": 0.0}
    extent = {"x_min": math.inf, "x_max": -math.inf, "z_min": math.inf, "z_max": -math.inf}
    attempts: list[float] = []
    launches = []
    for source_index, _split_name, gravity in assignments:
        parameters = sample_episode_parameters(
            gravity,
            np.random.default_rng([int(seed), int(source_index)]),
            spec,
            rise_range=rise_range,
            accel_x=float(accel_x),
        )
        attempts.append(parameters["attempts"])
        launches.append(parameters)
    payloads = [
        (parameters, spec, int(image_size), bool(with_accel_x)) for parameters in launches
    ]

    def records():
        stream = _render_stream(payloads, int(workers))
        for local_index, ((source_index, split_name, gravity), (pixels, action, state, diagnostics)) in enumerate(
            zip(assignments, stream, strict=True)
        ):
            if diagnostics["free_fall_residual_max"] > free_fall_tolerance:
                raise RuntimeError(
                    "rendered path departs from the analytic parabola by "
                    f"{diagnostics['free_fall_residual_max']:.3e} world units"
                )
            worst["free_fall_residual_max"] = max(
                worst["free_fall_residual_max"], diagnostics["free_fall_residual_max"]
            )
            for key in ("x_min", "z_min"):
                extent[key] = min(extent[key], diagnostics[key])
            for key in ("x_max", "z_max"):
                extent[key] = max(extent[key], diagnostics[key])
            physics = np.asarray(
                (
                    gravity,
                    launches[local_index]["mass"],
                    2.0 * spec.half_size,
                    2.0 * spec.half_size,
                    SHAPE_IDS[spec.shape],
                    spec.dt,
                    spec.view_height,
                    spec.z_center,
                ),
                dtype=np.float32,
            )
            yield EpisodeRecord(
                episode_index=local_index,
                split=split_name,
                pixels=pixels,
                state=state,
                action=action,
                gravity=float(gravity),
                physics=physics,
                source_episode_index=int(source_index),
            )

    dataset_path = write_lance_episodes(
        records(),
        output_dir / "data.lance",
        task=TASKS[spec.shape],
        frames_per_episode=spec.frames,
        image_size=image_size,
        image_format=image_format,
        names=action_names(bool(with_accel_x)),
        extra_metadata={
            "dt": spec.dt,
            "accel_x": float(accel_x),
            "fps": 1.0 / spec.dt,
            "scene": spec.to_dict(),
            "seed": int(seed),
            "contact_free": True,
            "train_gravity": {"mean": train_mean, "std": train_std, "clip_min": 0.1},
            "rise_range": list(rise_range),
            "test_gravities": [float(value) for value in test_gravities],
        },
    )
    manifest = {
        "schema_version": 1,
        "task": TASKS[spec.shape],
        "shape": spec.shape,
        "generator": "gwm.freefall_data",
        "dataset": dataset_path.name,
        "seed": int(seed),
        "episodes": len(assignments),
        "train_episodes": sum(item[1] == "train" for item in assignments),
        "test_episodes": sum(item[1] == "test" for item in assignments),
        "heldout_train_gravity_episodes": int(heldout_train_gravity_episodes),
        "heldout_train_gravity_source_range": [
            int(train_episodes),
            int(train_episodes) + int(heldout_train_gravity_episodes) - 1,
        ],
        "frames_per_episode": spec.frames,
        "image_size": image_size,
        "image_format": image_format,
        "dt": spec.dt,
        "scene": spec.to_dict(),
        "action_names": list(action_names(bool(with_accel_x))),
        "accel_x": float(accel_x),
        "state_names": list(STATE_NAMES),
        "test_gravities": [float(value) for value in test_gravities],
        "pixels_per_world_unit": image_size / spec.view_height,
        "gravity_step_pixels_at_train_mean": spec.gravity_step_pixels(train_mean, image_size),
        "max_free_fall_residual": worst["free_fall_residual_max"],
        "observed_extent": extent,
        "mean_rejection_attempts": float(np.mean(attempts)) if attempts else 0.0,
        "safe_box": list(spec.safe_box()),
        "rise_range": list(rise_range),
        "gravity_range": [float(min(all_gravities)), float(max(all_gravities))],
    }
    expected_test = int(heldout_train_gravity_episodes) + int(test_episodes_per_gravity) * len(
        test_gravities
    )
    if manifest["test_episodes"] != expected_test:
        raise RuntimeError(
            "held-out episode counts disagree: manifest says "
            f"{manifest['test_episodes']} but the request implies {expected_test}"
        )
    if manifest["episodes"] != manifest["train_episodes"] + manifest["test_episodes"]:
        raise RuntimeError("episode counts in the manifest do not add up")
    (output_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


__all__ = [
    "ACTION_NAMES",
    "ACTION_NAMES_WITH_ACCEL_X",
    "action_names",
    "BASE_SEED",
    "PHYSICS_NAMES",
    "SHAPE_IDS",
    "STATE_NAMES",
    "TASKS",
    "SceneSpec",
    "generate_freefall_dataset",
    "gravity_assignments",
    "render_episode",
    "verify_free_fall_against_integrator",
    "sample_episode_parameters",
    "shared_rise_range",
    "trajectory",
]
