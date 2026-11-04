"""The comparison table, from whatever evaluations exist.

Reads the per-variant evaluation JSON that ``evaluate.py`` and
``control_experiment.py`` write and prints the table the two notes ask for:
counterfactual accuracy inside and outside the training gravity range, how that
error depends on the gravity shift, whether the world model was degraded, and
whether the returned iterate carries the feasibility certificate Theorem 1
promises.

The slope column is the one Theorem 2 is about.  The theorem bounds the full
method's counterfactual error by a quantity that does *not* depend on the
gravity shift, while a gravity-blind predictor's error is bounded *below* by
one that grows with it.  A flat slope for the full method and a rising slope
for the ablations is the separation; the absolute errors alone do not show it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ORDER = ["pretrained", "jepa", "factual", "penalty", "counterfactual"]
LABELS = {
    "pretrained": "pretrained (theta_0)",
    "jepa": "continued JEPA training",
    "factual": "constraint, g'=g",
    "penalty": "fixed penalty L + lambda C",
    "counterfactual": "full method",
}


def summarise(payload: dict) -> dict:
    sweep = payload["counterfactual_sweep"]
    mean = float(payload["train_gravity"]["mean"])
    std = float(payload["train_gravity"]["std"])
    scale = float(payload["pixels_per_unit"])
    gravity = np.array([row["gravity"] for row in sweep])
    shift = np.abs(gravity - mean)
    error = np.array([row["position_rmse_final"] for row in sweep]) * scale
    accel = np.array([row["acceleration_rmse"] for row in sweep])
    inside = shift <= 2.0 * std
    outside = shift > 4.0 * std
    # The model's gravity response saturates only for *large upward* shifts:
    # measured on the pretrained predictor the ratio is near one below the
    # training gravity and 0.47 at the top of the grid.  Averaging over a
    # symmetric band mixes the half it handles with the half it does not and
    # hides what the constraint repairs, so report the far band separately.
    far = gravity >= mean + 3.0
    response = np.array([row.get("response_ratio", float("nan")) for row in sweep])
    kkt = payload.get("kkt", {})
    bounds = payload.get("corollary_two", [])
    return {
        "variant": payload["variant"],
        "accel_in": float(accel[inside].mean()) if inside.any() else float("nan"),
        "accel_out": float(accel[outside].mean()) if outside.any() else float("nan"),
        "pos_in": float(error[inside].mean()) if inside.any() else float("nan"),
        "pos_out": float(error[outside].mean()) if outside.any() else float("nan"),
        "pos_far": float(error[far].mean()) if far.any() else float("nan"),
        "response_far": float(np.nanmean(response[far])) if far.any() else float("nan"),
        "accel_far": float(accel[far].mean()) if far.any() else float("nan"),
        "slope": float(np.polyfit(shift, error, 1)[0]) if shift.std() > 0 else float("nan"),
        "jepa_loss": float(payload["latent_error"]["jepa_loss_sum_form"]),
        "feasible": kkt.get("feasible"),
        "constraint": kkt.get("constraint_at_output"),
        "stationarity": kkt.get("stationarity_certificate"),
        "bound_holds": all(row["holds"] for row in bounds) if bounds else None,
        "probe_disagreement": float(
            np.mean([row["probe_disagreement_by_step"][-1] for row in sweep]) * scale
        ),
    }


def configuration_of(payload: dict, path: Path) -> str:
    """The configuration a run belongs to, from its ``label``.

    Labels are ``<model>__<tag>`` and the tag names the configuration -- the
    variant plus whatever was swept, e.g. ``fr_eps0.3``.  Averaging must happen
    across *seeds* of one configuration, never across configurations: a
    tolerance sweep and a penalty-weight sweep are different experiments, and
    averaging them produced a "full method" row that mixed five tolerances,
    one of them infeasible.
    """

    label = payload.get("label") or path.parent.name
    return label.split("__", 1)[1] if "__" in label else label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--group",
        default=None,
        help="only report configurations whose tag starts with this prefix",
    )
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = {}
    labels: dict[str, str] = {}
    for path in args.evaluations:
        payload = json.loads(path.read_text())
        variant = payload.get("variant") or path.parent.name.replace("post_", "")
        payload["variant"] = variant
        tag = configuration_of(payload, path)
        if args.group and not tag.startswith(args.group):
            continue
        grouped.setdefault(tag, []).append(summarise(payload))
        labels[tag] = variant
    if not grouped:
        print("no evaluations matched")
        return

    def sort_key(tag: str) -> tuple[int, str]:
        variant = labels[tag]
        return (ORDER.index(variant) if variant in ORDER else len(ORDER), tag)

    header = (
        f"{'configuration':<28} {'n':>2} {'pos in':>7} {'pos OOD':>8} {'pos far':>8} "
        f"{'resp far':>9} {'accel far':>10} {'slope':>9} {'JEPA L':>9} {'feas':>5}"
    )
    print(header)
    print("-" * len(header))
    rows: dict[str, dict] = {}
    for tag in sorted(grouped, key=sort_key):
        runs = grouped[tag]

        def average(key: str, runs: list[dict] = runs) -> float:
            values = [row[key] for row in runs if row[key] is not None]
            return float(np.mean(values)) if values else float("nan")

        feasible = {row["feasible"] for row in runs}
        mark = "-" if feasible == {None} else ("yes" if feasible == {True} else "no")
        row = {
            key: average(key)
            for key in (
                "accel_in", "accel_out", "accel_far", "pos_in", "pos_out", "pos_far",
                "response_far", "slope", "jepa_loss", "constraint", "stationarity",
                "probe_disagreement",
            )
        }
        row.update(
            variant=labels[tag],
            feasible=mark,
            seeds=len(runs),
            bound_holds=(
                all(r["bound_holds"] for r in runs if r["bound_holds"] is not None)
                if any(r["bound_holds"] is not None for r in runs)
                else None
            ),
        )
        rows[tag] = row
        name = LABELS.get(labels[tag], labels[tag])
        shown = tag if tag.replace(labels[tag], "").strip("_") else name
        print(
            f"{shown:<28} {len(runs):>2} {row['pos_in']:>7.3f} {row['pos_out']:>8.3f} "
            f"{row['pos_far']:>8.3f} {row['response_far']:>9.3f} {row['accel_far']:>10.4f} "
            f"{row['slope']:>+9.5f} {row['jepa_loss']:>9.3f} {mark:>5}"
        )
    print()
    print("n = seeds averaged.  pos columns are px.  'far' is g' >= mean + 3, where the")
    print("pretrained response saturates; 'resp far' is the realised gravity response as a")
    print("fraction of the exact one (1 = correct, 0 = gravity-blind).  slope is px per")
    print("unit of |g' - g|, the quantity Theorem 2 separates.")
    baseline = next((rows[t] for t in rows if labels[t] == "pretrained"), None)
    best = min(
        (rows[t] for t in rows if labels[t] == "counterfactual"),
        key=lambda r: r["pos_far"],
        default=None,
    )
    if baseline and best:
        print(
            f"best counterfactual vs pretrained, far band: position error "
            f"{baseline['pos_far'] / max(best['pos_far'], 1e-9):.2f}x lower, "
            f"response {baseline['response_far']:.3f} -> {best['response_far']:.3f}, "
            f"slope {baseline['slope']:+.5f} -> {best['slope']:+.5f}"
        )
    for tag, row in rows.items():
        if row["bound_holds"] is False:
            print(f"  !! Corollary 2 violated somewhere for {tag}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
