"""Measure whether a pretrained encoder supports the constraint at all.

Before any post-training it is worth knowing two things about a checkpoint.

*Is position decodable from its latents?*  A closed-form ridge regression from
cached latents to true positions upper-bounds what any probe of that input can
achieve, so a poor result here is the encoder's and not the probe's.

*Is the constraint discriminative at the chosen stride?*  The tolerance is
``eps = (1 + rho) E[c_real]``, which Lemma 2 ties to the probe accuracy as
``16 delta^2 / (m dt)^4``, while a predictor that ignores the conditioning
gravity incurs the closed-form violation of
:func:`gwm.constraint.gravity_blind_violation`.  The ratio of the two is the
margin the constraint has to work with, and it grows as ``m^4``.  This script
tabulates it so the stride is chosen from measurement rather than by guess.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gwm.constraint import ConstraintSpec, gravity_blind_violation, gravity_vector
from gwm.latents import load_latent_cache


def ridge_decodability(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    lambdas: list[float],
    episode_ids: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Held-out ridge RMSE from ``features`` to ``targets``, in target units.

    The split is by **episode** when ``episode_ids`` is given.  Splitting by
    frame would leak: consecutive frames of one episode are nearly the same
    image, so a frame-level split lets the regression interpolate between
    neighbours of a test frame and reports an error far below what it achieves
    on an unseen episode.

    This is a *linear* read-out, so it is a lower bound on what a nonlinear
    probe can decode, not an upper bound: for a scene whose image is a function
    of position, the latent is necessarily a function of position, and the only
    question is how nonlinearly it is encoded.
    """

    count = features.shape[0]
    if episode_ids is None:
        order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
        cut = int(0.8 * count)
        train, test = order[:cut], order[cut:]
    else:
        episode_ids = np.asarray(episode_ids)
        if episode_ids.shape[0] != count:
            raise ValueError("episode_ids must have one entry per row of features")
        unique = np.unique(episode_ids)
        shuffled = unique.copy()
        np.random.default_rng(int(seed)).shuffle(shuffled)
        cut = max(1, int(0.8 * len(shuffled)))
        train_mask = np.isin(episode_ids, shuffled[:cut])
        train = torch.from_numpy(np.flatnonzero(train_mask))
        test = torch.from_numpy(np.flatnonzero(~train_mask))
        if len(test) == 0:
            raise ValueError("episode split left no held-out frames")
    # Choose the ridge penalty on a validation split, never on the split the
    # number is reported from.  Selecting it on the test fold inflates the
    # variance explained -- here by an order of magnitude, enough to look like
    # signal where there is none.
    fit = train[: int(0.75 * len(train))]
    validate = train[int(0.75 * len(train)) :]
    if len(validate) == 0:
        fit, validate = train, train
    design = torch.cat([features, torch.ones(count, 1, dtype=features.dtype)], dim=1)
    identity = torch.eye(design.shape[1], dtype=features.dtype)
    gram = design[fit].T @ design[fit]
    rhs = design[fit].T @ targets[fit]
    chosen, best_validation = lambdas[0], None
    for value in lambdas:
        weights = torch.linalg.solve(gram + value * identity, rhs)
        rmse = float((design[validate] @ weights - targets[validate]).pow(2).sum(-1).mean().sqrt())
        if best_validation is None or rmse < best_validation:
            best_validation, chosen = rmse, value
    # Refit on all of the training rows at the chosen penalty.
    weights = torch.linalg.solve(
        design[train].T @ design[train] + chosen * identity, design[train].T @ targets[train]
    )
    error = design[test] @ weights - targets[test]
    best = (float(error.pow(2).sum(-1).mean().sqrt()), chosen)
    baseline = float((targets[test] - targets[train].mean(0)).pow(2).sum(-1).mean().sqrt())
    return {
        "ridge_rmse_world": best[0],
        "best_lambda": best[1],
        "predict_mean_rmse_world": baseline,
        "variance_explained": 1.0 - (best[0] / baseline) ** 2,
        "frames": count,
        "split": "episode" if episode_ids is not None else "frame",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=3000)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--strides", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--shift", type=float, default=5.0, help="|g' - g| for the margin")
    parser.add_argument("--train-gravity", type=float, default=4.0)
    args = parser.parse_args()

    cache = load_latent_cache(args.cache)
    development = cache.select(cache.split == 0)
    episodes = min(args.max_episodes, development.episodes)
    latents = development.latents[:episodes].reshape(-1, development.latent_dim).double()
    positions = development.state[:episodes].reshape(-1, development.state.shape[-1])[:, :2].double()

    lambdas = [1e-4, 1e-2, 1.0, 1e1, 1e2, 1e3, 1e4]
    variants = {
        "raw": latents,
        "layer_normed": torch.nn.functional.layer_norm(
            latents.float(), (latents.shape[1],)
        ).double(),
    }
    episode_ids = np.repeat(np.arange(episodes), development.frames)
    decodability = {
        name: ridge_decodability(value, positions, lambdas=lambdas, episode_ids=episode_ids)
        for name, value in variants.items()
    }
    best_name = min(decodability, key=lambda name: decodability[name]["ridge_rmse_world"])
    delta = decodability[best_name]["ridge_rmse_world"]

    gravity = torch.tensor([float(args.train_gravity)])
    rows = []
    for stride in args.strides:
        if args.history < 2 * stride or args.horizon < stride:
            rows.append({"stride": stride, "status": "incompatible with history/horizon"})
            continue
        spec = ConstraintSpec(horizon=args.horizon, stride=stride, history=args.history)
        dt_effective = cache.dt * stride
        epsilon = (1.0 + args.rho) * 16.0 * delta**2 / dt_effective**4
        blind = float(
            gravity_blind_violation(
                spec, gravity_vector(gravity), gravity_vector(gravity + args.shift)
            )
        )
        rows.append(
            {
                "stride": stride,
                "dt_effective": dt_effective,
                "gravity_step_world": args.train_gravity * dt_effective**2,
                "gravity_step_pixels": args.train_gravity * dt_effective**2 * cache.pixels_per_unit,
                "epsilon_lemma2_upper_bound": epsilon,
                "gravity_blind_violation": blind,
                "margin": blind / epsilon if epsilon > 0 else float("inf"),
                "offsets": list(spec.offsets),
            }
        )

    payload = {
        "cache": str(args.cache.resolve()),
        "episodes_used": episodes,
        "pixels_per_unit": cache.pixels_per_unit,
        "dt": cache.dt,
        "position_scale_world": float(positions.pow(2).sum(-1).mean().sqrt()),
        "decodability": decodability,
        "best_feature_variant": best_name,
        "delta_upper_bound_world": delta,
        "delta_upper_bound_pixels": delta * cache.pixels_per_unit,
        "stride_table": rows,
        "note": (
            "delta here is a LINEAR read-out, split by episode; a nonlinear probe can "
            "do better, so these margins are pessimistic."
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"position scale (rms)        {payload['position_scale_world']:.4f} world units")
    for name, value in decodability.items():
        print(
            f"  ridge {name:<13} RMSE {value['ridge_rmse_world']:.4f} world "
            f"= {value['ridge_rmse_world'] * cache.pixels_per_unit:6.2f} px   "
            f"var explained {value['variance_explained']:+.4f}"
        )
    print()
    print(f"{'m':>2} {'dt_eff':>7} {'g*dt_eff^2 (px)':>16} {'eps bound':>12} {'c_blind':>10} {'margin':>8}")
    for row in rows:
        if "status" in row:
            print(f"{row['stride']:>2}  {row['status']}")
            continue
        print(
            f"{row['stride']:>2} {row['dt_effective']:>7.3f} {row['gravity_step_pixels']:>16.2f} "
            f"{row['epsilon_lemma2_upper_bound']:>12.3e} {row['gravity_blind_violation']:>10.3f} "
            f"{row['margin']:>8.2f}"
        )


if __name__ == "__main__":
    main()
