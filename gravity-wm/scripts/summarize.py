"""Collect the per-variant evaluations into the comparison the notes ask for."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["pretrained", "jepa", "factual", "penalty", "counterfactual"]
LABELS = {
    "pretrained": "pretrained SG-JEPA (theta_0)",
    "jepa": "continued JEPA training",
    "factual": "constraint, g'=g (no counterfactuals)",
    "penalty": "fixed penalty L + lambda C",
    "counterfactual": "full method",
}


def load(paths: list[Path]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text())
        variant = payload.get("variant") or payload["label"].split("__")[0]
        payload["variant"] = variant
        grouped[variant].append(payload)
    return grouped


def band(rows: list[dict], centre: float, width: float, inside: bool) -> list[dict]:
    out = []
    for row in rows:
        distance = abs(row["gravity"] - centre)
        if (distance <= width) == inside:
            out.append(row)
    return out


def aggregate(payloads: list[dict]) -> dict:
    """Average each metric over seeds, reporting the spread."""

    mean = payloads[0]["train_gravity"]["mean"]
    std = payloads[0]["train_gravity"]["std"]
    width = 2.0 * std

    def collect(fn) -> tuple[float, float]:
        values = [fn(payload) for payload in payloads]
        return float(np.mean(values)), float(np.std(values))

    def accel(payload: dict, inside: bool) -> float:
        rows = band(payload["counterfactual_sweep"], mean, width, inside)
        return float(np.mean([row["acceleration_rmse"] for row in rows]))

    def position(payload: dict, inside: bool) -> float:
        rows = band(payload["counterfactual_sweep"], mean, width, inside)
        return float(np.mean([row["position_rmse_final"] for row in rows]))

    def disagreement(payload: dict) -> float:
        rows = payload["counterfactual_sweep"]
        return float(np.mean([row["probe_disagreement_by_step"][-1] for row in rows]))

    return {
        "seeds": len(payloads),
        "accel_rmse_in_range": collect(lambda p: accel(p, True)),
        "accel_rmse_ood": collect(lambda p: accel(p, False)),
        "position_rmse_in_range": collect(lambda p: position(p, True)),
        "position_rmse_ood": collect(lambda p: position(p, False)),
        "jepa_loss": collect(lambda p: p["latent_error"]["jepa_loss_sum_form"]),
        "probe_disagreement": collect(disagreement),
        "corollary_two_holds": all(
            row["holds"] for payload in payloads for row in payload["corollary_two"]
        ),
        "probe_delta_world": collect(lambda p: p["probe_delta"]["eval"]["delta_world"]),
        "probe_delta_pixels": collect(lambda p: p["probe_delta"]["eval"]["delta_pixels"]),
    }


def table(summary: dict[str, dict], pixels_per_unit: float) -> str:
    header = (
        f"{'variant':<40} {'accel RMSE in-range':>19} {'accel RMSE OOD':>16} "
        f"{'pos RMSE OOD (px)':>18} {'JEPA loss':>12}"
    )
    lines = [header, "-" * len(header)]
    for variant in ORDER:
        if variant not in summary:
            continue
        row = summary[variant]
        lines.append(
            f"{LABELS[variant]:<40} "
            f"{row['accel_rmse_in_range'][0]:>13.4f}+-{row['accel_rmse_in_range'][1]:<5.3f} "
            f"{row['accel_rmse_ood'][0]:>10.4f}+-{row['accel_rmse_ood'][1]:<5.3f} "
            f"{row['position_rmse_ood'][0] * pixels_per_unit:>12.3f}+-"
            f"{row['position_rmse_ood'][1] * pixels_per_unit:<5.3f} "
            f"{row['jepa_loss'][0]:>12.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped = load(args.evaluations)
    summary = {variant: aggregate(payloads) for variant, payloads in grouped.items()}
    first = next(iter(grouped.values()))[0]
    pixels_per_unit = float(first["pixels_per_unit"])

    # The separation of Theorem 2: the full method's counterfactual error should
    # not grow with the gravity shift, while a gravity-blind predictor's should.
    separation = {}
    for variant, payloads in grouped.items():
        rows = payloads[0]["counterfactual_sweep"]
        shift = np.array([abs(row["gravity"] - first["train_gravity"]["mean"]) for row in rows])
        error = np.array([row["position_rmse_final"] for row in rows])
        if len(shift) > 2 and shift.std() > 0:
            slope = float(np.polyfit(shift, error, 1)[0])
            correlation = float(np.corrcoef(shift, error)[0, 1])
        else:
            slope, correlation = float("nan"), float("nan")
        separation[variant] = {
            "position_error_vs_shift_slope": slope,
            "position_error_vs_shift_correlation": correlation,
            "error_at_smallest_shift": float(error[np.argmin(shift)]),
            "error_at_largest_shift": float(error[np.argmax(shift)]),
        }

    payload = {
        "summary": summary,
        "separation_vs_gravity_shift": separation,
        "pixels_per_unit": pixels_per_unit,
        "train_gravity": first["train_gravity"],
        "constraint": first["constraint"],
        "evaluations": [str(path) for path in args.evaluations],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(table(summary, pixels_per_unit))
    print()
    print("Counterfactual position error vs |g' - g| (slope, world units per unit of g):")
    for variant in ORDER:
        if variant in separation:
            row = separation[variant]
            print(
                f"  {LABELS[variant]:<40} slope={row['position_error_vs_shift_slope']:+.5f}  "
                f"r={row['position_error_vs_shift_correlation']:+.3f}  "
                f"near={row['error_at_smallest_shift']:.4f} far={row['error_at_largest_shift']:.4f}"
            )


if __name__ == "__main__":
    main()
