"""Evaluate a (post-trained) world model: counterfactual sweep, bounds, ablation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gwm.constraint import ConstraintSpec
from gwm.evaluate import (
    corollary_two_bound,
    evaluate_counterfactual_sweep,
    evaluate_kkt_residuals,
    evaluate_latent_error,
    evaluate_true_ood,
    measure_probe_delta,
)
from gwm.latents import load_latent_cache, load_world_model, set_posttraining_modes
from gwm.probe import ProbeConfig, load_position_probe


def constraint_form_for(report: dict | None, args: argparse.Namespace) -> str:
    """The constraint form to evaluate in: the one the run was post-trained with.

    ``C`` in the acceleration form of Definition 1 and in the position form of
    Proposition 1 have the same zero set but different units, so a tolerance
    calibrated in one is meaningless against a violation measured in the other.
    The run records its own form; this returns that, lets an explicit flag
    stand when no report is supplied, and refuses a disagreement.
    """

    trained = ((report or {}).get("command") or {}).get("constraint_form")
    if trained is None:
        return args.constraint_form or "acceleration"
    if args.constraint_form is not None and args.constraint_form != trained:
        raise ValueError(
            f"--constraint-form {args.constraint_form} contradicts the post-training "
            f"report, which records {trained}; the KKT check would compare a "
            f"{args.constraint_form}-form violation against a {trained}-form epsilon"
        )
    return str(trained)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--posttrain-report", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument(
        "--constraint-form",
        default=None,
        choices=("acceleration", "position"),
        help="constraint form; taken from --posttrain-report when it records one, "
        "since the KKT check compares C against the epsilon that run was given "
        "and the two forms are not in the same units",
    )
    parser.add_argument("--sweep-min", type=float, default=1.0)
    parser.add_argument("--sweep-max", type=float, default=9.0)
    parser.add_argument("--sweep-step", type=float, default=0.5)
    parser.add_argument("--max-windows", type=int, default=6144)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--accel-x-index",
        type=int,
        default=None,
        help="action coordinate holding the horizontal acceleration; with it set "
        "the sweep runs over accel_x at a fixed gravity instead of over gravity",
    )
    parser.add_argument("--accel-x-min", type=float, default=-3.0)
    parser.add_argument("--accel-x-max", type=float, default=3.0)
    parser.add_argument("--accel-x-step", type=float, default=0.5)
    parser.add_argument(
        "--accel-x-sweep-gravity",
        type=float,
        default=None,
        help="the gravity to hold fixed while sweeping accel_x (default: the "
        "training mean, so the vertical motion is the one the model has seen)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _model_config, _statistics, _raw = load_world_model(args.checkpoint, device=device)
    set_posttraining_modes(model, ("predictor",))
    model.eval()
    model.requires_grad_(False)

    cache = load_latent_cache(args.cache).to(device)
    heldout = cache.select(cache.split == 1)
    scene = cache.metadata.get("scene") or {}
    half = 0.5 * float(scene["view_height"]) - float(scene["half_size"]) * 2**0.5 - float(scene["margin"])
    centre = float(scene["z_center"])
    safe_box = (-half, half, centre - half, centre + half)

    train_mean = float((cache.metadata.get("train_gravity") or {}).get("mean", 4.0))
    train_std = float((cache.metadata.get("train_gravity") or {}).get("std", 0.5))
    # The counterfactual protocol needs real windows from the training gravity
    # distribution.  The generator sets aside episodes drawn from exactly that
    # distribution and records their source range, so select by range: a
    # gravity filter would also admit the nearby points of the wide held-out
    # grid, whose contexts are not training-distribution windows.
    source_range = cache.metadata.get("heldout_train_gravity_source_range")
    if source_range:
        low, high = int(source_range[0]), int(source_range[1])
        is_context = (heldout.source_id >= low) & (heldout.source_id <= high)
        near_training = heldout.select(is_context)
        wide_grid = heldout.select(~is_context)
    else:
        near_training = heldout.select((heldout.gravity - train_mean).abs() <= 4.0 * train_std)
        wide_grid = heldout.select((heldout.gravity - train_mean).abs() > 4.0 * train_std)
    if near_training.episodes == 0:
        raise ValueError(
            "no held-out episodes at the training gravity; regenerate the dataset with "
            "--heldout-train-gravity-episodes so the counterfactual protocol has contexts"
        )

    probe_config = ProbeConfig()
    probe_train = load_position_probe(
        args.probe_dir / "probe_train.pt", feature_dim=cache.latent_dim, config=probe_config, device=device
    )
    probe_eval = load_position_probe(
        args.probe_dir / "probe_eval.pt", feature_dim=cache.latent_dim, config=probe_config, device=device
    )

    # The post-training report, read before the constraint spec is built: the
    # KKT block compares C(theta_out) against the epsilon that run was
    # calibrated with, and the acceleration and position forms differ by a
    # factor of order dt^-4.  Evaluating a position-form run with the
    # acceleration default made every certificate read infeasible at C/eps =
    # 240 while the run itself was inside its tolerance, so the form is taken
    # from the run rather than from a default, and an explicit disagreement is
    # an error rather than a silent rescaling.
    report = None
    if args.posttrain_report is not None and args.posttrain_report.is_file():
        report = json.loads(args.posttrain_report.read_text())
    spec = ConstraintSpec(
        horizon=args.horizon,
        stride=args.stride,
        history=args.history,
        form=constraint_form_for(report, args),
    )
    gravities = [
        float(value)
        for value in np.arange(args.sweep_min, args.sweep_max + 0.5 * args.sweep_step, args.sweep_step)
    ]

    delta_train = measure_probe_delta(probe_train, heldout)
    delta_eval = measure_probe_delta(probe_eval, heldout)

    if args.accel_x_index is None:
        sweep_gravities, sweep_accel_x = gravities, None
    else:
        # The horizontal slice: gravity held at the value the model has seen, so
        # the only thing it is extrapolating is the direction.
        held = args.accel_x_sweep_gravity
        sweep_gravities = [float(train_mean if held is None else held)]
        sweep_accel_x = [
            float(value)
            for value in np.arange(
                args.accel_x_min, args.accel_x_max + 0.5 * args.accel_x_step, args.accel_x_step
            )
        ]
    sweep = evaluate_counterfactual_sweep(
        model,
        probe_train,
        probe_eval,
        near_training,
        sweep_gravities,
        spec=spec,
        safe_box=safe_box,
        batch_size=args.batch_size,
        max_windows=args.max_windows,
        seed=args.seed,
        accel_x_values=sweep_accel_x,
        accel_x_index=args.accel_x_index,
    )
    latent = evaluate_latent_error(
        model, near_training, spec=spec, batch_size=args.batch_size,
        max_windows=args.max_windows, seed=args.seed
    )
    true_ood = (
        evaluate_true_ood(
            model, probe_eval, wide_grid, spec=spec, batch_size=args.batch_size,
            max_windows=1024, seed=args.seed
        )
        if wide_grid.episodes
        else []
    )

    # Corollary 2's bound, in the stride-one form in which it is stated, using
    # the measured probe accuracy and the measured stride-one violation.
    bounds = []
    for row in sweep:
        per_step = [
            corollary_two_bound(
                delta=delta_eval["delta_world"],
                constraint=row.violation_unit_stride,
                dt=cache.dt,
                horizon=spec.horizon,
                step=step,
                kappa=1.0,
            )
            for step in range(1, spec.horizon + 1)
        ]
        bounds.append(
            {
                "gravity": row.gravity,
                "bound_by_step": per_step,
                "measured_by_step": row.position_rmse_by_step,
                "slack_by_step": [
                    bound - measured
                    for bound, measured in zip(per_step, row.position_rmse_by_step, strict=True)
                ],
                "holds": all(
                    measured <= bound + 1e-9
                    for bound, measured in zip(per_step, row.position_rmse_by_step, strict=True)
                ),
            }
        )

    kkt = {"status": "no post-training report supplied"}
    if report is not None:
        history = report.get("history", [])
        epsilon = float(report["epsilon"])
        zeta_values = [row["zeta_C_estimate"] for row in history if "zeta_C_estimate" in row]
        grad_norms = [
            row["grad_l_norm"] for row in history if np.isfinite(row.get("grad_l_norm", np.nan))
        ]
        grad_c_norms = [
            row["grad_c_norm"] for row in history if np.isfinite(row.get("grad_c_norm", np.nan))
        ]
        at_training_gravity = min(sweep, key=lambda row: abs(row.gravity - train_mean))
        kkt = evaluate_kkt_residuals(
            history,
            epsilon=epsilon,
            gate=float(report["command"]["gate"]) if "command" in report else 0.9,
            constraint_at_output=float(
                np.mean([row.violation_stride for row in sweep])
            ),
            zeta_c=float(np.median(zeta_values)) if zeta_values else float("nan"),
            grad_l_bound=float(np.max(grad_norms)) if grad_norms else float("nan"),
            mu=float(np.min(grad_c_norms) / epsilon) if grad_c_norms else float("nan"),
        )
        kkt["constraint_at_training_gravity"] = at_training_gravity.violation_stride
        # The tolerance was calibrated under the run's own constraint spec, so
        # record whether this evaluation measured C under the same one.  The
        # form is reconciled above and cannot differ; horizon, history and
        # stride also change C's scale, and a mismatch makes C/eps -- and so
        # the feasibility certificate -- incomparable rather than merely noisy.
        trained_spec = {
            key: ((report.get("command") or {}).get(key))
            for key in ("history", "horizon", "stride", "constraint_form")
        }
        kkt["trained_constraint"] = trained_spec
        kkt["spec_matches_training"] = all(
            trained_spec[key] is None or int(trained_spec[key]) == getattr(spec, key)
            for key in ("history", "horizon", "stride")
        )
        if not kkt["spec_matches_training"]:
            kkt["feasible"] = None

    payload = {
        "label": args.label or args.checkpoint.parent.name,
        "checkpoint": str(args.checkpoint.resolve()),
        "cache": str(args.cache.resolve()),
        "constraint": {
            "history": spec.history,
            "horizon": spec.horizon,
            "stride": spec.stride,
            "form": spec.form,
        },
        "dt": cache.dt,
        "dt_effective": cache.dt * spec.stride,
        "pixels_per_unit": cache.pixels_per_unit,
        "safe_box": list(safe_box),
        "train_gravity": {"mean": train_mean, "std": train_std},
        "context_episodes_near_training": int(near_training.episodes),
        "wide_grid_episodes": int(wide_grid.episodes),
        "probe_delta": {"train": delta_train, "eval": delta_eval},
        "counterfactual_sweep": [row.to_dict() for row in sweep],
        "corollary_two": bounds,
        "latent_error": latent,
        "true_ood": true_ood,
        "kkt": kkt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    in_range = [row for row in sweep if abs(row.gravity - train_mean) <= 2.0 * train_std]
    out_range = [row for row in sweep if abs(row.gravity - train_mean) > 4.0 * train_std]
    print(
        json.dumps(
            {
                "label": payload["label"],
                "probe_delta_world": delta_eval["delta_world"],
                "probe_delta_pixels": delta_eval["delta_pixels"],
                "jepa_loss": latent["jepa_loss_sum_form"],
                "in_range_accel_rmse": float(np.mean([r.acceleration_rmse for r in in_range])) if in_range else None,
                "ood_accel_rmse": float(np.mean([r.acceleration_rmse for r in out_range])) if out_range else None,
                "in_range_final_position_rmse": float(np.mean([r.position_rmse_final for r in in_range])) if in_range else None,
                "ood_final_position_rmse": float(np.mean([r.position_rmse_final for r in out_range])) if out_range else None,
                "corollary_two_holds_everywhere": all(row["holds"] for row in bounds),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
