"""Run one gravity-constrained post-training variant."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from sg_jepa.train_utils import load_training_checkpoint

from gwm.constraint import ConstraintSpec
from gwm.latents import load_latent_cache, load_world_model, set_posttraining_modes
from gwm.posttrain import PostTrainConfig, run_post_training, save_result
from gwm.probe import ProbeConfig, load_position_probe
from gwm.spatial_lewm import read_readout_sidecar, write_readout_sidecar
from gwm.switching import SwitchingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="counterfactual")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--constraint-batch-size", type=int, default=384)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument(
        "--constraint-form",
        default="acceleration",
        choices=("acceleration", "position"),
        help="acceleration is Definition 1; position is Proposition 1's equivalent-zero-set "
             "form, which constrains what Corollary 2 bounds and is not blind to the sign "
             "structure of the residual",
    )
    parser.add_argument("--gravity-low", type=float, default=1.0)
    parser.add_argument("--gravity-high", type=float, default=9.0)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument(
        "--epsilon-scale",
        type=float,
        default=1.0,
        help="multiply the calibrated tolerance; the method seeks feasibility at eps "
             "rather than the smallest violation, so this is the accuracy dial",
    )
    parser.add_argument(
        "--epsilon-source",
        default="real_latents",
        choices=("real_latents", "factual_rollout"),
        help="real_latents is the method note's c_real; factual_rollout also carries the "
             "predictor's own rollout error, which makes the tolerance attainable",
    )
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--restore-lr", type=float, default=None)
    parser.add_argument("--gate", type=float, default=0.9, help="projection gate gamma")
    parser.add_argument("--exit-factor", type=float, default=0.9)
    parser.add_argument("--ema-decay", type=float, default=0.9)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0e4)
    parser.add_argument("--penalty-lambda", type=float, default=1.0)
    parser.add_argument("--no-project", action="store_true")
    parser.add_argument("--no-hysteresis", action="store_true")
    parser.add_argument("--no-in-frame-filter", action="store_true")
    parser.add_argument(
        "--restore-direction",
        default=SwitchingConfig.__dataclass_fields__["restore_direction"].default,
        choices=("polyak", "eps", "normalized"),
        help="polyak adapts the RESTORE step to ||grad C||; eps is the method note's "
             "grad C / eps with a fixed step, which diverges once ||grad C|| grows",
    )
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--accel-x-index",
        type=int,
        default=None,
        help="action coordinate holding the horizontal acceleration; when given, "
        "the counterfactual is an acceleration vector rather than a gravity "
        "magnitude (the two-dimensional experiment)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="pin an absolute tolerance instead of calibrating one; required "
        "for a staged schedule, where calibrating per stage lets a stage that "
        "damages the model relax its own successor's tolerance",
    )
    parser.add_argument(
        "--floor-resting-z",
        type=float,
        default=None,
        help="height of the object's centre at rest on the floor; enables the "
        "counterfactual contact filter for a bouncing dataset",
    )
    parser.add_argument(
        "--allow-contact-context",
        action="store_true",
        help="keep windows whose context spans a bounce (the implied velocity "
        "the position form needs is wrong there; for ablation only)",
    )
    parser.add_argument("--accel-x-low", type=float, default=0.0)
    parser.add_argument("--accel-x-high", type=float, default=0.0)
    parser.add_argument(
        "--trainable",
        nargs="+",
        default=["predictor", "pred_proj", "action_encoder"],
        help="modules whose parameters post-training updates",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    model, model_config, statistics, raw_config = load_world_model(args.checkpoint, device=device)
    freeze_report = set_posttraining_modes(model, tuple(args.trainable))

    cache = load_latent_cache(args.cache).to(device)
    development = cache.select(cache.split == 0)
    heldout = cache.select(cache.split == 1)
    # The tolerance must be calibrated on held-out windows from the *same*
    # context distribution the constraint is trained and evaluated on.  The
    # held-out split also contains the wide gravity grid, whose contexts
    # themselves evolve outside the training range; including them measures
    # the predictor on inputs it never sees during post-training and inflated
    # the tolerance by about 1.7x here, enough to leave theta_0 feasible and
    # the constraint inactive.  The generator reserves episodes drawn from the
    # training gravity for exactly this purpose and records their range.
    source_range = cache.metadata.get("heldout_train_gravity_source_range")
    if source_range:
        low, high = int(source_range[0]), int(source_range[1])
        calibration_set = heldout.select(
            (heldout.source_id >= low) & (heldout.source_id <= high)
        )
    else:
        train_mean = float((cache.metadata.get("train_gravity") or {}).get("mean", 4.0))
        train_std = float((cache.metadata.get("train_gravity") or {}).get("std", 0.5))
        calibration_set = heldout.select(
            (heldout.gravity - train_mean).abs() <= 4.0 * train_std
        )
    if calibration_set.episodes == 0:
        raise ValueError(
            "no held-out episodes at the training gravity to calibrate the tolerance on; "
            "regenerate the dataset with --heldout-train-gravity-episodes"
        )
    scene = cache.metadata.get("scene") or {}
    half = 0.5 * float(scene["view_height"]) - float(scene["half_size"]) * 2**0.5 - float(scene["margin"])
    centre = float(scene["z_center"])
    safe_box = (-half, half, centre - half, centre + half)

    probe_config = ProbeConfig()
    probe_train = load_position_probe(
        args.probe_dir / "probe_train.pt",
        feature_dim=cache.latent_dim,
        config=probe_config,
        device=device,
    )
    probe_eval = load_position_probe(
        args.probe_dir / "probe_eval.pt",
        feature_dim=cache.latent_dim,
        config=probe_config,
        device=device,
    )

    spec = ConstraintSpec(
        horizon=args.horizon,
        stride=args.stride,
        history=args.history,
        form=args.constraint_form,
    )
    switching = SwitchingConfig(
        lr=args.lr,
        restore_lr=args.restore_lr,
        gate=args.gate,
        exit_factor=args.exit_factor,
        ema_decay=args.ema_decay,
        momentum=args.momentum,
        grad_clip=args.grad_clip,
        project=not args.no_project,
        hysteresis=not args.no_hysteresis,
        restore_direction=args.restore_direction,
    )
    config = PostTrainConfig(
        variant=args.variant,
        steps=0 if args.variant == "pretrained" else args.steps,
        batch_size=args.batch_size,
        constraint_batch_size=args.constraint_batch_size,
        gravity_low=args.gravity_low,
        gravity_high=args.gravity_high,
        accel_x_index=args.accel_x_index,
        accel_x_low=args.accel_x_low,
        accel_x_high=args.accel_x_high,
        rho=args.rho,
        epsilon_source=args.epsilon_source,
        epsilon_scale=args.epsilon_scale,
        epsilon_override=args.epsilon,
        floor_resting_z=args.floor_resting_z,
        contact_free_context=not args.allow_contact_context,
        penalty_lambda=args.penalty_lambda,
        trainable=tuple(args.trainable),
        in_frame_filter=not args.no_in_frame_filter,
        eval_every=args.eval_every,
        seed=args.seed,
        switching=switching,
        constraint=spec,
    )

    started = time.perf_counter()
    last = {"step": -1, "time": started}

    def monitor(step: int, row: dict) -> None:
        now = time.perf_counter()
        rate = (step - last["step"]) / max(1e-9, now - last["time"])
        last.update(step=step, time=now)
        print(
            f"[{args.variant}] step {step:>6} mode={row['mode']:<8} "
            f"L={row['jepa_loss']:.4e} C={row['constraint']:.4e} "
            f"Chat={row['constraint_ema']:.4e} eps={row['epsilon']:.4e} "
            f"|d|={row['direction_norm']:.3e} proj={int(row['projected'])} "
            f"{rate:.1f} it/s",
            flush=True,
        )

    result = run_post_training(
        model,
        probe_train,
        probe_eval,
        development,
        calibration_set,
        config=config,
        safe_box=safe_box,
        device=device,
        monitor=monitor,
    )
    elapsed = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The post-trained checkpoint is loaded back by the evaluation, which has
    # to rebuild the same architecture.  Carry the read-out forward both as a
    # sidecar and inside the payload, so the checkpoint is self-describing
    # even if it is moved away from its directory.
    readout = read_readout_sidecar(args.checkpoint)
    write_readout_sidecar(args.output_dir, readout)
    payload = load_training_checkpoint(args.checkpoint)
    torch.save(
        {
            "schema_version": 2,
            "kind": "native_world_model_training",
            "step": config.steps,
            "cursor": {"next_epoch": 0, "next_microbatch": 0, "completed_epochs": 0},
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": {},
            "scheduler": {},
            "config": raw_config,
            "action_statistics": statistics.to_dict(),
            "data_split": payload.get("data_split"),
            "best_validation_loss": None,
            "history": [],
            "rng_state": {},
            "gwm_post_training": {
                "readout": readout,
                "variant": args.variant,
                "config": config.to_dict(),
                "freeze_report": freeze_report,
                "source_checkpoint": str(args.checkpoint.resolve()),
            },
        },
        args.output_dir / "checkpoint.pt",
    )
    report = result.to_dict()
    report["elapsed_seconds"] = elapsed
    report["freeze_report"] = freeze_report
    report["safe_box"] = list(safe_box)
    report["command"] = vars(args) | {
        key: str(value) for key, value in vars(args).items() if isinstance(value, Path)
    }
    save_result(result, args.output_dir / "posttrain_result.json")
    (args.output_dir / "posttrain_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                "variant": args.variant,
                "epsilon": result.epsilon,
                "elapsed_seconds": elapsed,
                "summary": result.summary,
                "diagnostics": result.diagnostics,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
