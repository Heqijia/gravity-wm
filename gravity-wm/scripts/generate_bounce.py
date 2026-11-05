"""Generate the bouncing dataset.

    python scripts/generate_bounce.py --output-dir DATA/bounce_single --train-episodes 16000
"""

from __future__ import annotations

import argparse
import json

from gwm.bounce_data import BounceSpec, generate_bounce_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-episodes", type=int, default=16000)
    parser.add_argument("--heldout-train-gravity-episodes", type=int, default=800)
    parser.add_argument("--test-episodes-per-gravity", type=int, default=32)
    parser.add_argument("--train-mean", type=float, default=4.0)
    parser.add_argument("--train-std", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.16)
    parser.add_argument(
        "--half-size",
        type=float,
        default=0.45,
        help="object half-extent; the contact-free datasets use 0.80, and a "
        "smaller object is harder for the position probe to localise",
    )
    parser.add_argument("--launch-low", type=float, default=3.5)
    parser.add_argument("--launch-high", type=float, default=7.0)
    parser.add_argument("--min-contacts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    spec = BounceSpec(
        dt=args.dt,
        frames=args.frames,
        min_contacts=args.min_contacts,
        half_size=args.half_size,
        launch_height_range=(args.launch_low, args.launch_high),
    )
    manifest = generate_bounce_dataset(
        args.output_dir,
        spec=spec,
        image_size=args.image_size,
        train_episodes=args.train_episodes,
        test_episodes_per_gravity=args.test_episodes_per_gravity,
        train_mean=args.train_mean,
        train_std=args.train_std,
        heldout_train_gravity_episodes=args.heldout_train_gravity_episodes,
        seed=args.seed,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
