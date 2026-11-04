"""Encode a dataset once with a pretrained frozen encoder and cache the latents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from gwm.latents import build_latent_cache, load_world_model, save_latent_cache


def checkpoint_fingerprint(path: Path, config: dict) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": digest.hexdigest(),
        "model_config": config["model"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-episodes", type=int, default=48)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, model_config, statistics, raw_config = load_world_model(args.checkpoint, device=device)
    cache = build_latent_cache(
        model, args.data, statistics, device=device, batch_episodes=args.batch_episodes
    )
    fingerprint = checkpoint_fingerprint(args.checkpoint, raw_config)
    fingerprint["dataset"] = str(args.data.resolve())
    save_latent_cache(cache, args.output, fingerprint)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": cache.episodes,
                "frames": cache.frames,
                "latent_dim": cache.latent_dim,
                "dt": cache.dt,
                "pixels_per_unit": cache.pixels_per_unit,
                "development_episodes": int((cache.split == 0).sum()),
                "heldout_episodes": int((cache.split == 1).sum()),
                "bytes": cache.latents.numel() * cache.latents.element_size(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
