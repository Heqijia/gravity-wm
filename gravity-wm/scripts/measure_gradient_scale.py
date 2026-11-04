"""Measure ``||grad L||`` at ``theta_0``, so the step size can be set per model.

Post-training uses a plain SGD step, so the step *length* is ``lr ||grad L||``
and a learning rate that is right for one predictor is not right for another.
The SSM predictor's gradient at ``theta_0`` exceeds the ``1e4`` clip, 20x the
GRU's median, and at ``lr = 1e-4`` continued JEPA training alone diverges on
it (``L`` 8.7 -> 1320 in 200 steps) -- a property of the step size, not of the
constraint.  This reports the norm and the learning rate that matches a chosen
reference step length.

    python scripts/measure_gradient_scale.py --checkpoint CKPT --cache CACHE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gwm.constraint import ConstraintSpec
from gwm.latents import load_latent_cache, load_world_model, set_posttraining_modes
from gwm.posttrain import jepa_loss

TRAINABLE = ("predictor", "pred_proj", "action_encoder")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reference-step",
        type=float,
        default=0.045,
        help="step length lr*||grad L|| to match; the default is the GRU's "
        "measured median (lr 1e-4 x 556)",
    )
    parser.add_argument(
        "--relative-step",
        type=float,
        default=1.5e-4,
        help="step length as a fraction of ||theta||; the default is the GRU's "
        "measured value and is the rule that transfers across architectures",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _config, _stats, _raw = load_world_model(args.checkpoint, device=device)
    set_posttraining_modes(model, TRAINABLE)
    parameters = [
        parameter
        for name, module in model.named_children()
        if name in TRAINABLE
        for parameter in module.parameters()
    ]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in parameters:
        parameter.requires_grad_(True)

    cache = load_latent_cache(args.cache).to(device)
    spec = ConstraintSpec(horizon=args.horizon, stride=1, history=args.history)
    development = cache.select(cache.split == 0)
    episode_index, start = development.windows(spec.history + spec.horizon)
    generator = torch.Generator().manual_seed(args.seed)

    norms = []
    losses = []
    for _ in range(args.batches):
        pick = torch.randint(
            episode_index.shape[0], (args.batch_size,), generator=generator
        ).to(episode_index.device)
        batch = development.gather_windows(episode_index[pick], start[pick], spec.history + spec.horizon)
        loss, _ = jepa_loss(model, batch, spec=spec)
        grads = torch.autograd.grad(loss, parameters, allow_unused=True)
        flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
        norms.append(float(flat.norm()))
        losses.append(float(loss.detach()))

    # The step length that matters is relative to the parameters it moves: a
    # step of 0.045 is a 1.5e-4 relative move for the GRU and a far larger one
    # for a predictor with a smaller parameter norm.  Matching the *relative*
    # step is what makes one rule serve every base model.
    parameter_norm = float(
        torch.cat([parameter.detach().reshape(-1) for parameter in parameters]).norm()
    )
    norms_tensor = torch.tensor(norms)
    median = float(norms_tensor.median())
    report = {
        "parameter_norm": parameter_norm,
        "relative_step_at_reference": args.reference_step / parameter_norm,
        "lr_for_relative_step": (
            args.relative_step * parameter_norm / median if median > 0 else None
        ),
        "relative_step": args.relative_step,
        "checkpoint": str(args.checkpoint),
        "batches": args.batches,
        "loss_mean": sum(losses) / len(losses),
        "grad_l_norm_median": median,
        "grad_l_norm_min": float(norms_tensor.min()),
        "grad_l_norm_max": float(norms_tensor.max()),
        "reference_step": args.reference_step,
        "lr_for_reference_step": args.reference_step / median if median > 0 else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
