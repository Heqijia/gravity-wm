"""Track whether a pretraining run is converging to a usable constraint.

The quantity that decides whether the whole study can run on real pixels is
not the JEPA loss: it is the position probe's accuracy relative to the gravity
displacement the constraint has to resolve.  This script evaluates that for the
newest checkpoint of each run, so the trend is visible while training is still
going and a run that will not get there can be stopped early.

For each checkpoint it reports

* ``delta``  -- the probe's held-out RMS position error, in world units and
  pixels.  The probe is the study's own MLP, fitted here on development
  episodes and evaluated on episodes it has never seen.
* ``c_real`` -- the *measured* tolerance scale at each candidate stride, not
  Lemma 2's bound, which is one to three orders of magnitude loose because a
  trained probe's error is a smooth function of the state and so largely
  cancels in a second difference.
* ``margin`` -- that tolerance against the closed-form violation of a predictor
  that ignores the conditioning gravity.  This is the signal the constraint has
  to work with; below about 3 it cannot teach the predictor anything, and
  Theorem 2's separation has nothing to separate.

Appends one row per checkpoint to a TSV so the trend can be read off.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sg_jepa.data import open_trajectory_store

from gwm.constraint import ConstraintSpec, gravity_blind_violation, gravity_vector
from gwm.latents import LatentCache, load_world_model
from gwm.posttrain import calibrate_tolerance
from gwm.probe import ProbeConfig, build_position_probe

FIELDS = (
    "timestamp",
    "run",
    "readout",
    "checkpoint",
    "epochs",
    "step",
    "val_loss",
    "val_rollout",
    "delta_world",
    "delta_pixels",
    "baseline_pixels",
    "variance_explained",
    "best_stride",
    "best_margin",
    "epsilon_at_best",
    "c_real_at_best",
)


def encode_episodes(model, store, indices, device, *, statistics):
    image_mean = torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
    image_std = torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
    action_mean = torch.tensor(statistics.mean)
    action_std = torch.tensor(statistics.std)
    latents, states, actions, gravities = [], [], [], []
    with torch.no_grad():
        for begin in range(0, len(indices), 24):
            pixels, state, action = [], [], []
            for index in indices[begin : begin + 24]:
                episode = store.episode(int(index))
                frames = torch.from_numpy(
                    np.ascontiguousarray(np.moveaxis(episode.pixels, -1, 1))
                ).float().div_(255.0)
                pixels.append((frames - image_mean) / image_std)
                state.append(torch.from_numpy(episode.state.astype(np.float32)))
                action.append(
                    (torch.from_numpy(episode.action.astype(np.float32)) - action_mean) / action_std
                )
                gravities.append(float(episode.gravity[0]))
            encoded = model.encode({"pixels": torch.stack(pixels).to(device)})["emb"].float()
            latents.append(encoded)
            states.append(torch.stack(state).to(device))
            actions.append(torch.stack(action).to(device))
    return (
        torch.cat(latents),
        torch.cat(states),
        torch.cat(actions),
        torch.tensor(gravities, device=device),
    )


def fit_probe(features, positions, *, device, epochs, seed=0):
    """Fit the study's probe and return it; selection is on a held-out slice."""

    cut = int(0.85 * features.shape[0])
    mean, std = positions[:cut].mean(0), positions[:cut].std(0).clamp_min(1e-6)
    torch.manual_seed(int(seed))
    network = build_position_probe(features.shape[1], ProbeConfig(input_norm=False)).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1.0e-3, weight_decay=1.0e-6)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))
    best, best_state = float("inf"), None
    for _ in range(int(epochs)):
        network.train()
        order = torch.randperm(cut, device=device)
        for begin in range(0, cut, 4096):
            batch = order[begin : begin + 4096]
            loss = F.mse_loss(network(features[batch]), (positions[batch] - mean) / std)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        schedule.step()
        network.eval()
        with torch.no_grad():
            value = float(
                (network(features[cut:]) * std + mean - positions[cut:]).pow(2).sum(-1).mean()
            )
        if value < best:
            best = value
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
    network.load_state_dict(best_state)
    network.eval().requires_grad_(False)

    class Wrapped(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = network
            self.register_buffer("m", mean.reshape(1, 2))
            self.register_buffer("s", std.reshape(1, 2))

        def forward(self, x):
            return self.net(x) * self.s + self.m

    return Wrapped().to(device).eval().requires_grad_(False)


def evaluate_checkpoint(path: Path, data: Path, args, device) -> dict:
    model, model_config, statistics, config = load_world_model(path, device=device)
    model.eval()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    history = [row for row in payload.get("history", []) if "split" in row]
    store = open_trajectory_store(data)
    metadata = store.metadata
    scene = metadata["scene"]
    pixels_per_unit = int(metadata["pixels_shape"][0]) / float(scene["view_height"])

    development = np.flatnonzero(store.split_id == 0)
    fit_indices = development[: args.fit_episodes]
    heldout = np.flatnonzero(store.split_id == 1)[: args.test_episodes]

    Zf, Sf, _Af, _Gf = encode_episodes(model, store, fit_indices, device, statistics=statistics)
    Zt, St, At, Gt = encode_episodes(model, store, heldout, device, statistics=statistics)
    features = Zf.reshape(-1, Zf.shape[-1])
    positions = Sf.reshape(-1, Sf.shape[-1])[:, :2]
    probe = fit_probe(features, positions, device=device, epochs=args.probe_epochs)

    target = St.reshape(-1, St.shape[-1])[:, :2]
    with torch.no_grad():
        predicted = probe(Zt.reshape(-1, Zt.shape[-1]))
    delta = float((predicted - target).pow(2).sum(-1).mean().sqrt())
    baseline = float((target - positions.mean(0)).pow(2).sum(-1).mean().sqrt())

    cache = LatentCache(
        latents=Zt,
        state=St,
        action=At,
        gravity=Gt,
        split=torch.ones(Zt.shape[0], dtype=torch.long, device=device),
        source_id=torch.arange(Zt.shape[0], device=device),
        # The contact-free generator's scenes cannot produce one.
        contact=torch.zeros(Zt.shape[:2], dtype=torch.bool, device=device),
        dt=float(metadata["dt"]),
        pixels_per_unit=pixels_per_unit,
        action_mean=torch.tensor(statistics.mean, device=device),
        action_std=torch.tensor(statistics.std, device=device),
        metadata={},
    )
    frames = Zt.shape[1]
    train_gravity = float((metadata.get("train_gravity") or {}).get("mean", 4.0))
    rows = []
    for stride in args.strides:
        history_len = max(args.history, 2 * stride)
        horizon = frames - history_len
        if horizon < stride or horizon < 1:
            continue
        spec = ConstraintSpec(horizon=horizon, stride=stride, history=history_len)
        measured = calibrate_tolerance(
            probe, cache, spec=spec, windows=args.calibration_windows, seed=0
        )
        epsilon = (1.0 + args.rho) * measured["c_real_mean"]
        gravity = torch.full((1,), train_gravity, device=device)
        blind = float(
            gravity_blind_violation(
                spec, gravity_vector(gravity), gravity_vector(gravity + args.shift)
            )
        )
        rows.append(
            {
                "stride": stride,
                "history": history_len,
                "horizon": horizon,
                "gravity_step_pixels": train_gravity
                * (cache.dt * stride) ** 2
                * pixels_per_unit,
                "c_real": measured["c_real_mean"],
                "epsilon": epsilon,
                "gravity_blind_violation": blind,
                "margin": blind / epsilon if epsilon > 0 else float("inf"),
            }
        )
    best = max(rows, key=lambda row: row["margin"]) if rows else None
    return {
        "readout": config.get("readout", "cls"),
        "epochs": int(payload["cursor"]["completed_epochs"]),
        "step": int(payload["step"]),
        "val_loss": history[-1]["validation_loss"] if history else float("nan"),
        "val_rollout": history[-1]["validation_rollout_loss"] if history else float("nan"),
        "delta_world": delta,
        "delta_pixels": delta * pixels_per_unit,
        "baseline_pixels": baseline * pixels_per_unit,
        "variance_explained": 1.0 - (delta / baseline) ** 2,
        "strides": rows,
        "best": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-episodes", type=int, default=1200)
    parser.add_argument("--test-episodes", type=int, default=400)
    parser.add_argument("--probe-epochs", type=int, default=250)
    parser.add_argument("--calibration-windows", type=int, default=4096)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--strides", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--shift", type=float, default=5.0, help="|g' - g| for the margin")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--include-best", action="store_true",
                        help="also score best.pt, which is fresher than the periodic checkpoints")
    args = parser.parse_args()

    if len(args.data) not in (1, len(args.runs)):
        parser.error("--data must be one path or one per run")
    datasets = args.data if len(args.data) == len(args.runs) else list(args.data) * len(args.runs)
    device = torch.device(args.device)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if args.log.is_file():
        for line in args.log.read_text().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) > 3:
                seen.add(parts[3])
    else:
        args.log.write_text("\t".join(FIELDS) + "\n")

    for run, data in zip(args.runs, datasets, strict=True):
        candidates = sorted(run.glob("checkpoint_epoch_*.pt"))
        if (run / "checkpoint.pt").is_file():
            candidates.append(run / "checkpoint.pt")
        if args.include_best and (run / "best.pt").is_file():
            # best.pt is rewritten whenever validation improves, so it is the
            # freshest view of a run between periodic checkpoints.  It is keyed
            # by epoch so a newer one is scored rather than skipped.
            candidates.append(run / "best.pt")
        if args.latest_only:
            candidates = candidates[-1:]
        for checkpoint in candidates:
            key = str(checkpoint)
            if checkpoint.name == "best.pt":
                key = f"{checkpoint}@{int(checkpoint.stat().st_mtime)}"
            if key in seen:
                continue
            try:
                result = evaluate_checkpoint(checkpoint, data, args, device)
            except Exception as error:  # a half-written checkpoint, or a run just starting
                print(f"  {run.name}/{checkpoint.name}: {type(error).__name__}: {error}", flush=True)
                continue
            best = result["best"] or {}
            row = [
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                run.name,
                result["readout"],
                key,
                result["epochs"],
                result["step"],
                f"{result['val_loss']:.6g}",
                f"{result['val_rollout']:.6g}",
                f"{result['delta_world']:.6g}",
                f"{result['delta_pixels']:.6g}",
                f"{result['baseline_pixels']:.6g}",
                f"{result['variance_explained']:.6g}",
                best.get("stride", ""),
                f"{best.get('margin', float('nan')):.6g}",
                f"{best.get('epsilon', float('nan')):.6g}",
                f"{best.get('c_real', float('nan')):.6g}",
            ]
            with args.log.open("a") as handle:
                handle.write("\t".join(str(value) for value in row) + "\n")
            print(
                f"  {run.name:<22} {result['readout']:<12} ep {result['epochs']:>3}  "
                f"val {result['val_loss']:>9.4f}  delta {result['delta_pixels']:>6.2f} px "
                f"(baseline {result['baseline_pixels']:.1f}, var_expl {result['variance_explained']:+.4f})  "
                f"best margin {best.get('margin', float('nan')):>7.2f} at stride {best.get('stride', '-')}",
                flush=True,
            )
            seen.add(key)
            (run / "supervise_latest.json").write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
            )


if __name__ == "__main__":
    main()
