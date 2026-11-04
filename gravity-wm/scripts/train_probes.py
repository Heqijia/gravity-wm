"""Train the position probe ``h`` and the independent evaluation probe ``h_eval``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gwm.constraint import ConstraintSpec, gravity_blind_violation, gravity_vector
from gwm.latents import load_latent_cache
from gwm.posttrain import calibrate_tolerance
from gwm.probe import ProbeConfig, probe_report, save_position_probe, train_position_probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-episodes", type=int, default=6000)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--rho", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device(args.device)
    cache = load_latent_cache(args.cache)

    # Probes are fitted on development episodes only; the held-out episodes are
    # reserved for calibrating the tolerance and for evaluation.
    development = cache.select(cache.split == 0)
    if development.episodes > args.max_episodes:
        order = torch.randperm(
            development.episodes, generator=torch.Generator().manual_seed(args.seed)
        )[: args.max_episodes]
        development = development.select(order)
    heldout = cache.select(cache.split == 1).to(device)

    episodes, frames = development.episodes, development.frames
    features = development.latents.reshape(episodes * frames, -1).to(device)
    positions = development.state.reshape(episodes * frames, -1)[:, :2].to(device)
    episode_ids = np.repeat(np.arange(episodes), frames)

    config = ProbeConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )
    probe_train, probe_eval, summary = train_position_probes(
        features, positions, episode_ids, config=config, device=device
    )

    spec = ConstraintSpec(horizon=args.horizon, stride=args.stride, history=args.history)
    dt_effective = cache.dt * spec.stride
    train_gravity = float(cache.metadata.get("train_gravity", {}).get("mean", 4.0))

    reports = {}
    with torch.no_grad():
        held_features = heldout.latents.reshape(-1, heldout.latent_dim)
        held_positions = heldout.state.reshape(-1, heldout.state.shape[-1])[:, :2]
        for name, probe in (("train", probe_train), ("eval", probe_eval)):
            reports[name] = probe_report(
                probe(held_features),
                held_positions,
                dt_effective=dt_effective,
                gravity=train_gravity,
                pixels_per_unit=cache.pixels_per_unit,
            ).to_dict()

    # The tolerance that matters is the *measured* one: Lemma 2's
    # 16 delta^2 / dt_eff^4 treats per-frame probe errors as unrelated, whereas
    # a trained probe is a deterministic function of the state and its error
    # varies smoothly along a trajectory, so most of it cancels in the second
    # difference.  The table below reports, for each candidate stride, the
    # measured E[c_real], the tolerance it implies, and the margin against the
    # closed-form violation of a predictor that ignores the conditioning
    # gravity -- the signal the constraint has to work with.
    strides = [value for value in (1, 2, 3, 4) if args.history >= 2 * value and args.horizon >= value]
    stride_table = []
    reference_shift = 5.0
    for value in strides:
        candidate = ConstraintSpec(horizon=args.horizon, stride=value, history=args.history)
        measured = {
            name: calibrate_tolerance(probe, heldout, spec=candidate, windows=8192, seed=args.seed)
            for name, probe in (("train", probe_train), ("eval", probe_eval))
        }
        tolerance = (1.0 + args.rho) * measured["train"]["c_real_mean"]
        gravity = torch.full((1,), train_gravity, device=device)
        blind = float(
            gravity_blind_violation(
                candidate, gravity_vector(gravity), gravity_vector(gravity + reference_shift)
            )
        )
        lemma2 = 16.0 * reports["train"]["rmse_world"] ** 2 / (cache.dt * value) ** 4
        stride_table.append(
            {
                "stride": value,
                "dt_effective": cache.dt * value,
                "gravity_step_pixels": train_gravity
                * (cache.dt * value) ** 2
                * cache.pixels_per_unit,
                "c_real_measured": measured["train"]["c_real_mean"],
                "c_real_measured_eval_probe": measured["eval"]["c_real_mean"],
                "epsilon": tolerance,
                "lemma2_bound": lemma2,
                "lemma2_looseness": lemma2 / max(measured["train"]["c_real_mean"], 1e-30),
                "gravity_blind_violation": blind,
                "reference_shift": reference_shift,
                "margin": blind / tolerance if tolerance > 0 else float("inf"),
                "offsets": list(candidate.offsets),
            }
        )

    calibration = {
        name: calibrate_tolerance(probe, heldout, spec=spec, windows=8192, seed=args.seed)
        for name, probe in (("train", probe_train), ("eval", probe_eval))
    }
    epsilon = (1.0 + args.rho) * calibration["train"]["c_real_mean"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache": str(args.cache.resolve()),
        "feature_dim": development.latent_dim,
        "probe_config": config.to_dict(),
        "training_summary": summary,
        "heldout_reports": reports,
        "constraint": {"history": spec.history, "horizon": spec.horizon, "stride": spec.stride},
        "dt": cache.dt,
        "dt_effective": dt_effective,
        "calibration": calibration,
        "rho": args.rho,
        "epsilon": epsilon,
        "lemma2_bound_vs_measured": {
            name: {
                "bound": reports[name]["lemma2_bound_on_c_real"],
                "measured": calibration[name]["c_real_mean"],
            }
            for name in reports
        },
        "stride_table": stride_table,
    }
    save_position_probe(probe_train, args.output_dir / "probe_train.pt", metadata)
    save_position_probe(probe_eval, args.output_dir / "probe_eval.pt", metadata)
    (args.output_dir / "probe_report.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "epsilon": epsilon,
                "train_probe": {
                    k: reports["train"][k]
                    for k in ("rmse_world", "rmse_pixels", "gravity_step_pixels", "signal_to_noise")
                },
                "eval_probe": {
                    k: reports["eval"][k]
                    for k in ("rmse_world", "rmse_pixels", "gravity_step_pixels", "signal_to_noise")
                },
                "c_real": {k: v["c_real_mean"] for k, v in calibration.items()},
                "lemma2_bound": reports["train"]["lemma2_bound_on_c_real"],
                "stride_table": [
                    {
                        key: row[key]
                        for key in (
                            "stride",
                            "gravity_step_pixels",
                            "c_real_measured",
                            "epsilon",
                            "lemma2_looseness",
                            "gravity_blind_violation",
                            "margin",
                        )
                    }
                    for row in stride_table
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
