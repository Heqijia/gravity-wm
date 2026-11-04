"""A controlled test of the algorithm with the theory's assumptions satisfied.

Whether the constraint can teach a predictor about unseen gravities is a
question about the *algorithm*.  Whether a particular pretrained encoder
happens to encode position precisely enough for the position probe is a
separate question about that encoder.  This script separates them: it keeps the
real predictor, the real switching method, the real probes and the real
evaluation, and replaces only the encoder by an invertible random affine map of
the true dynamical state.  The probe is then accurate, ``eps`` is small, and
what the run measures is whether Algorithm 1 does what Theorem 1 and Theorem 2
say it does.

The pretrained predictor is made **gravity-blind**, which is the hypothesis
Theorem 2(b) and (c) are stated under: it is trained on the JEPA loss with the
gravity coordinate of the action held at zero, so it fits the real windows
without ever seeing which gravity produced them.  Post-training then restores
the gravity coordinate, exactly as adding a FiLM adapter initialised to the
identity would.  This is the configuration in which the theory predicts a
separation: continued JEPA training and the factual constraint cannot teach the
predictor about ``g'``, while the counterfactual constraint can.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sg_jepa.config import WorldModelConfig
from sg_jepa.data import ActionStatistics, open_trajectory_store
from sg_jepa.models.lewm import MLP, ActionConditionedGRUPredictor, Embedder, LeWM

from gwm.constraint import ConstraintSpec, gravity_blind_violation, gravity_vector
from gwm.evaluate import (
    corollary_two_bound,
    evaluate_counterfactual_sweep,
    evaluate_kkt_residuals,
    evaluate_latent_error,
    measure_probe_delta,
)
from gwm.latents import LatentCache, manifest_range, set_posttraining_modes
from gwm.posttrain import PostTrainConfig, calibrate_tolerance, jepa_loss, run_post_training
from gwm.probe import ProbeConfig, train_position_probes
from gwm.switching import SwitchingConfig


def synthetic_cache(
    dataset: str | Path,
    *,
    latent_dim: int,
    noise: float,
    seed: int,
    device: torch.device,
) -> LatentCache:
    """A latent space in which position is exactly decodable.

    ``z_t = W [x, z, vx, vz, 1] + sigma n``, with ``W`` a random matrix of full
    column rank.  Position is then an affine function of the latent up to the
    noise floor ``sigma``, so the probe can reach whatever accuracy the noise
    allows and ``delta`` becomes a knob rather than a property of a checkpoint.
    Velocity is included because the predictor needs it to roll out; the
    constant column makes the map affine.

    Everything else -- the episodes, ``dt``, the gravities, the safe box, the
    action normalisation -- comes from the real generated dataset, so the
    physics and the geometry of the control run are the real ones.
    """

    store = open_trajectory_store(dataset)
    metadata = dict(store.metadata)
    scene = metadata["scene"]
    episodes = len(store.lengths)
    lengths = np.unique(store.lengths)
    if len(lengths) != 1:
        raise ValueError(f"expected equal-length episodes, got {lengths}")
    frames = int(lengths[0])

    state = np.empty((episodes, frames, 8), dtype=np.float32)
    action = np.empty((episodes, frames, 3), dtype=np.float32)
    gravity = np.empty(episodes, dtype=np.float32)
    source = np.empty(episodes, dtype=np.int64)
    for index in range(episodes):
        episode = store.episode(index)
        state[index] = episode.state
        action[index] = episode.action
        gravity[index] = episode.gravity[0]
        source[index] = int(episode.episode_id)

    statistics = ActionStatistics.fit(action.reshape(-1, 3))
    mean = torch.tensor(statistics.mean, dtype=torch.float32)
    std = torch.tensor(statistics.std, dtype=torch.float32)
    normalized_action = (torch.from_numpy(action) - mean) / std

    generator = torch.Generator().manual_seed(int(seed))
    features = torch.cat(
        [torch.from_numpy(state[..., :4]), torch.ones(episodes, frames, 1)], dim=-1
    )
    scale = features.reshape(-1, 5).std(dim=0).clamp_min(1.0e-3)
    weights = torch.randn(5, int(latent_dim), generator=generator) / scale.unsqueeze(-1)
    latents = features @ weights
    flat = latents.reshape(-1, int(latent_dim))
    # Centre as well as scale.  SIGReg and the projector's BatchNorm leave real
    # Semigroup-JEPA latents roughly centred with unit scale per dimension; an
    # uncentred synthetic latent would carry the scene's offset (z_center) as a
    # large constant that ``pred_proj``'s BatchNorm removes, leaving the
    # predictor a constant error it cannot fit.
    latents = (latents - flat.mean(dim=0)) / flat.std(dim=0).clamp_min(1.0e-8)
    latents = latents + float(noise) * torch.randn(
        latents.shape, generator=generator, dtype=latents.dtype
    )

    cache = LatentCache(
        latents=latents,
        state=torch.from_numpy(state),
        action=normalized_action,
        gravity=torch.from_numpy(gravity),
        split=torch.from_numpy(np.asarray(store.split_id, dtype=np.int64)),
        source_id=torch.from_numpy(source),
        # The contact-free generator's scenes cannot produce one.
        contact=torch.zeros(latents.shape[:2], dtype=torch.bool),
        dt=float(metadata["dt"]),
        pixels_per_unit=int(metadata["pixels_shape"][0]) / float(scene["view_height"]),
        action_mean=mean,
        action_std=std,
        metadata={
            "dataset": str(Path(dataset).resolve()),
            "frames": frames,
            "dt": float(metadata["dt"]),
            "scene": scene,
            "train_gravity": metadata.get("train_gravity"),
            "test_gravities": metadata.get("test_gravities"),
            "heldout_train_gravity_source_range": manifest_range(dataset),
            "synthetic": {"latent_dim": latent_dim, "noise": noise, "seed": seed},
        },
    )
    return cache.to(device)


def build_predictor_only_model(
    *, latent_dim: int, history: int, device: torch.device, seed: int
) -> LeWM:
    """The real Semigroup-JEPA predictor stack, with no visual encoder.

    ``LeWM.predict`` -- predictor then ``pred_proj`` -- is the whole path
    post-training touches, so the encoder and projector are identities here and
    latents come from the synthetic cache instead.
    """

    torch.manual_seed(int(seed))
    config = WorldModelConfig(history_size=history, embed_dim=latent_dim, action_dim=3)
    predictor = ActionConditionedGRUPredictor(
        num_frames=history,
        depth=config.predictor_depth,
        input_dim=latent_dim,
        hidden_dim=config.predictor_hidden_dim,
        output_dim=latent_dim,
        mlp_dim=config.predictor_mlp_dim,
        dropout=0.0,
        cond_dim=latent_dim,
        conditioning=config.predictor_conditioning,
        residual_init=config.predictor_residual_init,
    )
    model = LeWM(
        encoder=torch.nn.Identity(),
        predictor=predictor,
        action_encoder=Embedder(input_dim=3, emb_dim=latent_dim),
        projector=torch.nn.Identity(),
        pred_proj=MLP(
            input_dim=latent_dim,
            output_dim=latent_dim,
            hidden_dim=config.projector_hidden_dim,
            norm_fn=torch.nn.BatchNorm1d,
        ),
    )
    return model.to(device)


def pretrain_gravity_blind(
    model: LeWM,
    cache: LatentCache,
    *,
    spec: ConstraintSpec,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> list[dict]:
    """Fit the predictor to real windows with the gravity coordinate zeroed.

    The result is Theorem 2's gravity-blind ``theta_0``: accurate on the windows
    it was trained on, and with a response to the conditioning gravity that
    carries no information, because it never saw one.
    """

    length = spec.history + spec.horizon
    episode_index, start = cache.windows(length)
    total = int(episode_index.shape[0])
    generator = torch.Generator().manual_seed(int(seed))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1.0e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(steps)))
    history: list[dict] = []
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    for step in range(int(steps)):
        chunk = torch.randint(0, total, (int(batch_size),), generator=generator)
        chunk = chunk.to(episode_index.device)
        batch = cache.gather_windows(episode_index[chunk], start[chunk], length)
        blind = batch["action"].clone()
        blind[:, :, 2] = 0.0
        loss, mean_squared = jepa_loss(model, {**batch, "action": blind}, spec=spec)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        schedule.step()
        if step % max(1, int(steps) // 20) == 0 or step == int(steps) - 1:
            history.append(
                {
                    "step": step,
                    "jepa_loss": float(loss.detach()),
                    "mean_squared": float(mean_squared.detach()),
                }
            )
            print(f"  blind-pretrain step {step:>6} L={float(loss.detach()):.4e}", flush=True)
    model.eval()
    return history


def safe_box_from_scene(scene: dict) -> tuple[float, float, float, float]:
    corner = 1.0 if scene.get("shape", "ball") == "ball" else math.sqrt(2.0)
    reach = 0.5 * float(scene["view_height"]) - float(scene["half_size"]) * corner - float(
        scene["margin"]
    )
    centre = float(scene["z_center"])
    return (-reach, reach, centre - reach, centre + reach)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument(
        "--latent-noise",
        type=float,
        default=0.02,
        help="latent noise, which sets the floor on the probe accuracy delta",
    )
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--blind-steps", type=int, default=4000)
    parser.add_argument("--blind-lr", type=float, default=1.0e-3)
    parser.add_argument("--post-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--gate", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=1.0e4)
    parser.add_argument("--gravity-low", type=float, default=1.0)
    parser.add_argument("--gravity-high", type=float, default=9.0)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument(
        "--epsilon-source",
        default="real_latents",
        choices=("real_latents", "factual_rollout"),
    )
    parser.add_argument(
        "--lr-sweep",
        nargs="+",
        type=float,
        default=None,
        help="before the variants, sweep the OPTIMISE step size on the jepa ablation",
    )
    parser.add_argument("--penalty-lambda", type=float, default=1.0)
    parser.add_argument("--probe-episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["pretrained", "jepa", "factual", "penalty", "counterfactual"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    spec = ConstraintSpec(horizon=args.horizon, stride=args.stride, history=args.history)
    cache = synthetic_cache(
        args.data,
        latent_dim=args.latent_dim,
        noise=args.latent_noise,
        seed=args.seed,
        device=device,
    )
    scene = cache.metadata["scene"]
    safe_box = safe_box_from_scene(scene)

    development = cache.select(cache.split == 0)
    heldout = cache.select(cache.split == 1)
    train_mean = float((cache.metadata.get("train_gravity") or {}).get("mean", 4.0))
    train_std = float((cache.metadata.get("train_gravity") or {}).get("std", 0.5))
    source_range = cache.metadata.get("heldout_train_gravity_source_range")
    if source_range:
        low, high = int(source_range[0]), int(source_range[1])
        near_training = heldout.select(
            (heldout.source_id >= low) & (heldout.source_id <= high)
        )
    else:
        near_training = heldout.select((heldout.gravity - train_mean).abs() <= 4.0 * train_std)
    if near_training.episodes == 0:
        raise ValueError(
            "the dataset has no held-out episodes at the training gravity; the "
            "counterfactual protocol needs them as its real windows"
        )
    print(
        f"cache: {cache.episodes} episodes, {cache.frames} frames, dim {cache.latent_dim}; "
        f"development {development.episodes}, held-out {heldout.episodes} "
        f"(near training gravity {near_training.episodes})",
        flush=True,
    )

    probe_episodes = min(args.probe_episodes, development.episodes)
    features = development.latents[:probe_episodes].reshape(-1, cache.latent_dim)
    positions = development.state[:probe_episodes].reshape(-1, 8)[:, :2]
    episode_ids = np.repeat(np.arange(probe_episodes), cache.frames)
    probe_train, probe_eval, probe_summary = train_position_probes(
        features,
        positions,
        episode_ids,
        config=ProbeConfig(
            input_norm=False, max_epochs=200, patience=20, batch_size=1024, seed=1234
        ),
        device=device,
    )
    delta = {
        "train": measure_probe_delta(probe_train, heldout),
        "eval": measure_probe_delta(probe_eval, heldout),
    }
    print(
        f"probe delta: train {delta['train']['delta_world']:.5f} world "
        f"({delta['train']['delta_pixels']:.3f} px), eval {delta['eval']['delta_world']:.5f} "
        f"({delta['eval']['delta_pixels']:.3f} px)",
        flush=True,
    )

    model = build_predictor_only_model(
        latent_dim=cache.latent_dim, history=args.history, device=device, seed=args.seed
    )
    blind_history = pretrain_gravity_blind(
        model,
        development,
        spec=spec,
        steps=args.blind_steps,
        batch_size=args.batch_size,
        lr=args.blind_lr,
        seed=args.seed + 1,
    )
    baseline_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    gravity = torch.full((1,), train_mean, device=device)
    c_real = calibrate_tolerance(
        probe_train, heldout, spec=spec, windows=8192, seed=args.seed
    )
    epsilon = (1.0 + args.rho) * c_real["c_real_mean"]
    blind_violation = float(
        gravity_blind_violation(spec, gravity_vector(gravity), gravity_vector(gravity + 5.0))
    )
    reference = {
        "c_real": c_real,
        "epsilon": epsilon,
        "gravity_blind_violation_shift_5": blind_violation,
        "margin": blind_violation / epsilon if epsilon > 0 else float("inf"),
    }
    print(
        f"eps={epsilon:.4e}  c_blind(shift 5)={blind_violation:.3f}  "
        f"margin={reference['margin']:.1f}",
        flush=True,
    )

    if args.lr_sweep:
        # Plain SGD on L is the `jepa` ablation.  It is only a fair baseline if
        # its step size decreases L; too large a step makes the "continued JEPA
        # training" comparison a straw man.  Pick the largest that descends.
        print("\n===== step-size sweep on the jepa ablation =====", flush=True)
        print(f"  {'lr':>10} {'L before':>12} {'L after':>12} {'ratio':>8}", flush=True)
        descending = []
        for candidate in args.lr_sweep:
            model.load_state_dict(baseline_state)
            set_posttraining_modes(model, ("predictor", "pred_proj", "action_encoder"))
            before = evaluate_latent_error(
                model, near_training, spec=spec, batch_size=512, max_windows=2048, seed=args.seed
            )["jepa_loss_sum_form"]
            probe_config = PostTrainConfig(
                variant="jepa",
                steps=min(args.post_steps, 600),
                batch_size=args.batch_size,
                constraint_batch_size=2 * args.batch_size,
                gravity_low=args.gravity_low,
                gravity_high=args.gravity_high,
                rho=args.rho,
                epsilon_source=args.epsilon_source,
                eval_every=10**9,
                seed=args.seed,
                switching=SwitchingConfig(lr=candidate, gate=args.gate, grad_clip=args.grad_clip),
                constraint=spec,
            )
            run_post_training(
                model, probe_train, probe_eval, development, heldout,
                config=probe_config, safe_box=safe_box, device=device,
            )
            after = evaluate_latent_error(
                model, near_training, spec=spec, batch_size=512, max_windows=2048, seed=args.seed
            )["jepa_loss_sum_form"]
            ratio = after / max(before, 1e-30)
            print(f"  {candidate:>10.1e} {before:>12.4e} {after:>12.4e} {ratio:>8.3f}", flush=True)
            if ratio < 0.98:
                descending.append((ratio, candidate))
        if not descending:
            raise RuntimeError(
                "no step size in the sweep decreased the JEPA loss; widen --lr-sweep downwards"
            )
        chosen = min(descending)[1]
        print(f"  -> using lr={chosen:.1e} for the variants", flush=True)
        args.lr = chosen

    gravities = [float(value) for value in np.arange(args.gravity_low, args.gravity_high + 0.01, 0.5)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    headline: dict[str, dict] = {}

    for variant in args.variants:
        print(f"\n===== {variant} =====", flush=True)
        model.load_state_dict(baseline_state)
        set_posttraining_modes(model, ("predictor", "pred_proj", "action_encoder"))
        config = PostTrainConfig(
            variant=variant,
            steps=0 if variant == "pretrained" else args.post_steps,
            batch_size=args.batch_size,
            constraint_batch_size=2 * args.batch_size,
            gravity_low=args.gravity_low,
            gravity_high=args.gravity_high,
            rho=args.rho,
            epsilon_source=args.epsilon_source,
            penalty_lambda=args.penalty_lambda,
            eval_every=max(1, args.post_steps // 20),
            seed=args.seed,
            switching=SwitchingConfig(lr=args.lr, gate=args.gate, grad_clip=args.grad_clip),
            constraint=spec,
        )
        started = time.perf_counter()
        outcome = run_post_training(
            model,
            probe_train,
            probe_eval,
            development,
            near_training,   # calibrate on the same contexts the constraint uses
            config=config,
            safe_box=safe_box,
            device=device,
            monitor=lambda step, row: print(
                f"  step {step:>5} {row['mode']:<8} L={row['jepa_loss']:.4e} "
                f"C={row['constraint']:.4e} Chat={row['constraint_ema']:.4e} "
                f"|d|={row['direction_norm']:.3e} proj={int(row['projected'])}",
                flush=True,
            ),
        )
        elapsed = time.perf_counter() - started

        model.eval()
        model.requires_grad_(False)
        sweep = evaluate_counterfactual_sweep(
            model,
            probe_train,
            probe_eval,
            near_training,
            gravities,
            spec=spec,
            safe_box=safe_box,
            batch_size=512,
            max_windows=6144,
            seed=args.seed,
        )
        latent = evaluate_latent_error(
            model, near_training, spec=spec, batch_size=512, max_windows=6144, seed=args.seed
        )
        bounds = []
        for row in sweep:
            per_step = [
                corollary_two_bound(
                    delta=delta["eval"]["delta_world"],
                    constraint=row.violation_unit_stride,
                    dt=cache.dt,
                    horizon=spec.horizon,
                    step=step,
                )
                for step in range(1, spec.horizon + 1)
            ]
            bounds.append(
                {
                    "gravity": row.gravity,
                    "bound_by_step": per_step,
                    "measured_by_step": row.position_rmse_by_step,
                    "holds": all(
                        measured <= bound + 1e-9
                        for bound, measured in zip(per_step, row.position_rmse_by_step, strict=True)
                    ),
                }
            )

        zeta_values = [
            row["zeta_C_estimate"] for row in outcome.history if "zeta_C_estimate" in row
        ]
        grad_l = [row["grad_l_norm"] for row in outcome.history if np.isfinite(row.get("grad_l_norm", np.nan))]
        grad_c = [row["grad_c_norm"] for row in outcome.history if np.isfinite(row.get("grad_c_norm", np.nan))]
        kkt = evaluate_kkt_residuals(
            outcome.history,
            epsilon=outcome.epsilon,
            gate=args.gate,
            constraint_at_output=float(np.mean([row.violation_stride for row in sweep])),
            zeta_c=float(np.median(zeta_values)) if zeta_values else float("nan"),
            grad_l_bound=float(np.max(grad_l)) if grad_l else float("nan"),
            mu=(float(np.min(grad_c)) / outcome.epsilon) if grad_c else float("nan"),
        )

        payload = {
            "variant": variant,
            "label": variant,
            "elapsed_seconds": elapsed,
            "posttrain": outcome.to_dict(),
            "counterfactual_sweep": [row.to_dict() for row in sweep],
            "latent_error": latent,
            "corollary_two": bounds,
            "probe_delta": delta,
            "train_gravity": {"mean": train_mean, "std": train_std},
            "constraint": {
                "history": spec.history,
                "horizon": spec.horizon,
                "stride": spec.stride,
            },
            "dt": cache.dt,
            "pixels_per_unit": cache.pixels_per_unit,
            "safe_box": list(safe_box),
            "kkt": kkt,
        }
        (args.output_dir / f"evaluation_{variant}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        )

        shift = np.array([abs(row.gravity - train_mean) for row in sweep])
        error = np.array([row.position_rmse_final for row in sweep])
        slope = float(np.polyfit(shift, error, 1)[0]) if shift.std() > 0 else float("nan")
        headline[variant] = {
            "steps": config.steps,
            "accel_rmse": float(np.mean([row.acceleration_rmse for row in sweep])),
            "position_rmse_final_world": float(error.mean()),
            "position_rmse_final_pixels": float(error.mean() * cache.pixels_per_unit),
            "slope_vs_gravity_shift": slope,
            "jepa_loss": latent["jepa_loss_sum_form"],
            "constraint_at_output": kkt.get("constraint_at_output"),
            "feasible": kkt.get("feasible"),
            "corollary_two_holds": all(entry["holds"] for entry in bounds),
            "elapsed_seconds": elapsed,
        }
        row = headline[variant]
        print(
            f"  -> accel RMSE {row['accel_rmse']:.4f}  final pos RMSE "
            f"{row['position_rmse_final_world']:.4f} world "
            f"({row['position_rmse_final_pixels']:.2f} px)  slope {row['slope_vs_gravity_shift']:+.5f}  "
            f"L={row['jepa_loss']:.4e}  C={row['constraint_at_output']}",
            flush=True,
        )

    (args.output_dir / "control_summary.json").write_text(
        json.dumps(
            {
                "reference": reference,
                "headline": headline,
                "probe_summary": probe_summary,
                "blind_pretrain_history": blind_history,
                "settings": {key: str(value) for key, value in vars(args).items()},
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )

    print(f"\n{'variant':<16} {'accel RMSE':>11} {'pos RMSE (px)':>14} {'slope':>10} {'JEPA L':>11} {'feasible':>9}")
    for variant, row in headline.items():
        feasible = "-" if row["feasible"] is None else str(row["feasible"])
        print(
            f"{variant:<16} {row['accel_rmse']:>11.4f} {row['position_rmse_final_pixels']:>14.3f} "
            f"{row['slope_vs_gravity_shift']:>+10.5f} {row['jepa_loss']:>11.4e} {feasible:>9}"
        )
    print("\nwrote", args.output_dir)


if __name__ == "__main__":
    main()
