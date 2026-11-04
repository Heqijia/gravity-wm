"""Generate the contact-free projectile dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gwm.freefall_data import BASE_SEED, SceneSpec, generate_freefall_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--test-episodes-per-gravity", type=int, default=24)
    parser.add_argument("--test-gravity-min", type=float, default=1.0)
    parser.add_argument("--test-gravity-max", type=float, default=9.0)
    parser.add_argument("--test-gravity-step", type=float, default=0.5)
    parser.add_argument("--train-mean", type=float, default=4.0)
    parser.add_argument("--train-std", type=float, default=0.5)
    parser.add_argument(
        "--heldout-train-gravity-episodes",
        type=int,
        default=400,
        help="held-out episodes at the training gravity; context set for the counterfactual protocol",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.16)
    parser.add_argument("--view-height", type=float, default=13.0)
    parser.add_argument("--shape", default="ball", choices=("ball", "square"),
                        help="ball has no visible orientation; square rotates (harder variant)")
    parser.add_argument("--half-size", type=float, default=0.80)
    parser.add_argument("--image-format", default="png", choices=("png", "jpeg"))
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--with-accel-x",
        action="store_true",
        help="append the horizontal acceleration to the action layout so the "
        "predictor can be conditioned on it; the first three coordinates keep "
        "their indices so gravity stays at index 2",
    )
    parser.add_argument(
        "--accel-x",
        type=float,
        default=0.0,
        help="the horizontal acceleration in every episode; zero is the regime "
        "the two-dimensional experiment wants, where only the constraint can "
        "teach the conditioning path what a horizontal acceleration does",
    )
    args = parser.parse_args()

    spec = SceneSpec(
        dt=args.dt,
        frames=args.frames,
        view_height=args.view_height,
        z_center=0.5 * args.view_height,
        shape=args.shape,
        half_size=args.half_size,
    )
    grid = np.arange(
        args.test_gravity_min,
        args.test_gravity_max + 0.5 * args.test_gravity_step,
        args.test_gravity_step,
    )
    manifest = generate_freefall_dataset(
        args.output,
        spec=spec,
        image_size=args.image_size,
        train_episodes=args.train_episodes,
        test_gravities=tuple(float(value) for value in grid),
        test_episodes_per_gravity=args.test_episodes_per_gravity,
        train_mean=args.train_mean,
        train_std=args.train_std,
        heldout_train_gravity_episodes=args.heldout_train_gravity_episodes,
        seed=args.seed,
        image_format=args.image_format,
        workers=args.workers,
        accel_x=args.accel_x,
        with_accel_x=args.with_accel_x,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
