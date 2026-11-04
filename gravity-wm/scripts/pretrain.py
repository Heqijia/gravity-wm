"""Pretrain a Semigroup-JEPA world model, with a choice of latent read-out.

``--readout cls`` is the published model and goes through
``sg_jepa.models.build_world_model`` untouched.  ``--readout attn_pool`` swaps
the ``[CLS]`` summary for a learned attention pool over the patch tokens, which
is the one change that lets the latent carry the object's location (see
``gwm.spatial_lewm`` for the measurement that motivates it).

Everything else -- the trainer, its validation, checkpointing, resume and
strict-reload check -- is the upstream code.  The builder is injected into
``sg_jepa.training`` rather than reimplemented, so the two read-outs are
trained by exactly the same loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sg_jepa.training as sg_training
from sg_jepa.config import load_experiment_config
from sg_jepa.data.dataset import ActionStatistics

from gwm.spatial_lewm import READOUTS, build_readout_world_model, write_readout_sidecar


def fix_action_scales(scales: dict[int, float]):
    """Pin chosen action coordinates' normalising scales; returns an undo call.

    See :func:`fix_gravity_scale` for why.  The two-dimensional experiment needs
    two of these: gravity at index 2 and the horizontal acceleration at index 3,
    both constant in the training data and both conditioned on values far from
    that constant.
    """

    original = ActionStatistics.fit

    def fit(_cls, action):
        statistics = original(action)
        std = list(statistics.std)
        for index, scale in scales.items():
            std[index] = float(scale)
        return ActionStatistics(statistics.mean, tuple(std))

    ActionStatistics.fit = classmethod(fit)
    return lambda: setattr(ActionStatistics, "fit", original)


def fix_gravity_scale(scale: float, index: int = -1):
    """Pin the gravity coordinate's normalising scale, returning an undo callable.

    The gravity coordinate is z-scored with the training data's own standard
    deviation, which is the wrong scale whenever the training gravity barely
    varies.  On the narrow dataset (``g ~ N(4, 0.05^2)``) training inputs span
    +-4 in normalised units while a counterfactual ``g'`` in [1, 9] arrives as
    -60 to +100: the conditioning path is evaluated 25x outside the range it
    ever saw, on top of the physics extrapolation being asked for.  With a
    constant training gravity the measured scale collapses to the ``1e-6``
    floor and the counterfactual input reaches ``5e6``, which is meaningless.

    Fixing the scale to the counterfactual law's own spread -- for
    ``g' ~ U(1, 9)`` that is ``8 / sqrt(12) = 2.309`` -- puts every gravity the
    model will be conditioned on inside a unit-ish range, and leaves the hard
    part where it belongs: the training data still carries no information about
    what gravity does, so the constraint is the only signal.  The statistics go
    into the checkpoint, so the cache, the post-training and the evaluation all
    inherit them.
    """

    return fix_action_scales({index: scale})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--readout", default="cls", choices=READOUTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--gravity-scale",
        type=float,
        default=None,
        help="fix the gravity action coordinate's normalising scale instead of "
        "using the training data's own standard deviation; required when the "
        "training gravity is constant and advisable whenever its spread is far "
        "narrower than the counterfactual range",
    )
    parser.add_argument(
        "--gravity-index",
        type=int,
        default=-1,
        help="which action coordinate is gravity (default: the last)",
    )
    parser.add_argument(
        "--accel-x-scale",
        type=float,
        default=None,
        help="the same fix for the horizontal-acceleration coordinate; needed "
        "whenever the training data holds it constant, which is the regime the "
        "two-dimensional experiment uses",
    )
    parser.add_argument(
        "--accel-x-index",
        type=int,
        default=3,
        help="which action coordinate is the horizontal acceleration",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    readout = args.readout

    scales: dict[int, float] = {}
    if args.gravity_scale is not None:
        scales[args.gravity_index] = args.gravity_scale
    if args.accel_x_scale is not None:
        scales[args.accel_x_index] = args.accel_x_scale
    if scales:
        fix_action_scales(scales)

    if readout != "cls":
        # The trainer calls build_world_model twice: once to create the model
        # and once for the strict-reload check at the end.  Replacing the name
        # it resolves covers both, and leaves the loop itself untouched.
        original = sg_training.build_world_model

        def builder(model_config):
            return build_readout_world_model(model_config, readout)

        sg_training.build_world_model = builder
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_readout_sidecar(args.output_dir, readout)
            report = sg_training.run_training(
                config,
                args.data,
                args.output_dir,
                device=args.device,
                resume=args.resume,
                max_steps=args.max_steps,
            )
        finally:
            sg_training.build_world_model = original
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_readout_sidecar(args.output_dir, readout)
        report = sg_training.run_training(
            config,
            args.data,
            args.output_dir,
            device=args.device,
            resume=args.resume,
            max_steps=args.max_steps,
        )

    report["readout"] = readout
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
