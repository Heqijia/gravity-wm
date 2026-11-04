"""Figures for the gravity-constrained post-training study.

Four panels, each tied to a specific claim:

1. *Counterfactual error against the gravity shift.*  Theorem 2's separation:
   the full method's bound does not depend on the shift, while a gravity-blind
   predictor's error grows linearly in it.  A flat curve for the full method and
   a rising one for the ablations is the result.
2. *Corollary 2's bound against what was measured*, per rollout step.  The bound
   has to sit above the measurement; how much slack there is says how tight it
   is in practice.
3. *The switching trace*: the violation, its moving average, the tolerance and
   which mode each step took.  This is where RESTORE / OPTIMISE and the
   hysteresis become visible, and where a run that never reaches feasibility
   shows itself.
4. *Probe accuracy and constraint margin against epoch*, from the supervision
   log.  This is the plot that decides whether a pretrained model can support
   the constraint at all.

Reads the JSON the evaluation and post-training stages write, so it needs no
state of its own.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["pretrained", "jepa", "factual", "penalty", "counterfactual"]
LABELS = {
    "pretrained": "pretrained (gravity-blind)",
    "jepa": "continued JEPA training",
    "factual": "constraint, $g'=g$",
    "penalty": r"fixed penalty $L+\lambda C$",
    "counterfactual": "full method",
}
COLOURS = {
    "pretrained": "#888888",
    "jepa": "#1f77b4",
    "factual": "#2ca02c",
    "penalty": "#d62728",
    "counterfactual": "#000000",
}


def load_evaluations(paths: list[Path]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        variant = payload.get("variant") or payload.get("label", path.stem).split("__")[0]
        grouped.setdefault(variant, []).append(payload)
    return grouped


def panel_separation(axis, grouped: dict[str, list[dict]]) -> None:
    mean = next(iter(grouped.values()))[0]["train_gravity"]["mean"]
    scale = next(iter(grouped.values()))[0]["pixels_per_unit"]
    for variant in ORDER:
        if variant not in grouped:
            continue
        runs = grouped[variant]
        gravities = np.array([row["gravity"] for row in runs[0]["counterfactual_sweep"]])
        errors = np.stack(
            [
                [row["position_rmse_final"] for row in payload["counterfactual_sweep"]]
                for payload in runs
            ]
        ) * scale
        shift = gravities - mean
        order = np.argsort(shift)
        middle = errors.mean(axis=0)[order]
        axis.plot(
            shift[order], middle, label=LABELS[variant], color=COLOURS[variant],
            linewidth=2.0 if variant == "counterfactual" else 1.4,
        )
        if errors.shape[0] > 1:
            spread = errors.std(axis=0)[order]
            axis.fill_between(
                shift[order], middle - spread, middle + spread,
                color=COLOURS[variant], alpha=0.15, linewidth=0,
            )
    axis.axvline(0.0, color="#cccccc", linewidth=0.8, zorder=0)
    axis.set_xlabel(r"gravity shift  $g' - g$")
    axis.set_ylabel("counterfactual position error (px)")
    axis.set_title("Theorem 2: error against the gravity shift")
    axis.legend(fontsize=8, frameon=False)


def panel_bound(axis, grouped: dict[str, list[dict]]) -> None:
    if "counterfactual" not in grouped:
        axis.set_visible(False)
        return
    payload = grouped["counterfactual"][0]
    scale = payload["pixels_per_unit"]
    rows = payload["corollary_two"]
    mean = payload["train_gravity"]["mean"]
    far = max(rows, key=lambda row: abs(row["gravity"] - mean))
    steps = np.arange(1, len(far["bound_by_step"]) + 1)
    axis.plot(
        steps, np.array(far["bound_by_step"]) * scale,
        marker="o", color="#d62728", label="Corollary 2 bound",
    )
    axis.plot(
        steps, np.array(far["measured_by_step"]) * scale,
        marker="s", color="#000000", label="measured",
    )
    axis.set_xlabel("rollout step $s$")
    axis.set_ylabel("position error (px)")
    axis.set_title(f"Corollary 2 at $g'={far['gravity']:.1f}$ (holds: {far['holds']})")
    axis.legend(fontsize=8, frameon=False)


def panel_switching(axis, report: Path | None) -> None:
    if report is None or not report.is_file():
        axis.set_visible(False)
        return
    payload = json.loads(report.read_text())
    history = payload.get("history", [])
    if not history:
        axis.set_visible(False)
        return
    steps = [row["step"] for row in history]
    epsilon = float(payload["epsilon"])
    axis.plot(steps, [row["constraint"] for row in history], color="#aaaaaa",
              linewidth=0.9, label=r"$C_{B}$")
    axis.plot(steps, [row["constraint_ema"] for row in history], color="#1f77b4",
              linewidth=1.6, label=r"$\hat{C}$")
    axis.axhline(epsilon, color="#d62728", linestyle="--", linewidth=1.2, label=r"$\varepsilon$")
    restore = [row["step"] for row in history if row["mode"] == "restore"]
    if restore:
        axis.scatter(
            restore, [epsilon * 1.6] * len(restore), marker="|", s=40,
            color="#d62728", label="RESTORE",
        )
    axis.set_yscale("log")
    axis.set_xlabel("post-training step")
    axis.set_ylabel("violation")
    axis.set_title("switching trace")
    axis.legend(fontsize=8, frameon=False)


def panel_supervision(axis, log: Path | None) -> None:
    if log is None or not log.is_file():
        axis.set_visible(False)
        return
    rows = list(csv.DictReader(log.open(), delimiter="\t"))
    if not rows:
        axis.set_visible(False)
        return
    twin = axis.twinx()
    colours = {"cls": "#d62728", "attn_pool": "#ff7f0e", "spatial_map": "#000000"}
    names = {"cls": "[CLS] (published)", "attn_pool": "attention pool", "spatial_map": "spatial map"}
    markers = ["o", "s", "^", "D"]
    # One line per *run*, not per read-out: two seeds of the same read-out are
    # separate curves, and concatenating them would interleave their epochs.
    runs: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        runs.setdefault((row["readout"], row["run"]), []).append(row)
    for index, ((readout, run), subset) in enumerate(sorted(runs.items())):
        subset = sorted(subset, key=lambda row: int(row["epochs"]))
        epochs = [int(row["epochs"]) for row in subset]
        colour = colours.get(readout, "#555555")
        marker = markers[index % len(markers)]
        axis.plot(
            epochs, [float(row["delta_pixels"]) for row in subset],
            color=colour, marker=marker, markersize=4, linewidth=1.5,
            label=f"{names.get(readout, readout)} · {run.split('_')[-1]}",
        )
        twin.plot(
            epochs, [float(row["best_margin"]) for row in subset],
            color=colour, marker=marker, markersize=3, linestyle=":",
            linewidth=1.1, alpha=0.65,
        )
    twin.axhline(3.0, color="#2ca02c", linestyle="--", linewidth=1.0)
    twin.annotate(
        "margin 3: usable",
        xy=(1.0, 3.0), xycoords=("axes fraction", "data"),
        xytext=(-4, 4), textcoords="offset points",
        ha="right", va="bottom", fontsize=7, color="#2ca02c",
    )
    axis.set_yscale("log")
    twin.set_yscale("log")
    axis.set_xlabel("pretraining epoch")
    axis.set_ylabel("probe error (px) — solid")
    twin.set_ylabel("constraint margin — dotted")
    axis.set_title("does the latent support the constraint?")
    axis.legend(fontsize=7, frameon=False, loc="center left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="*", type=Path, default=[])
    parser.add_argument("--posttrain-report", type=Path, default=None)
    parser.add_argument("--supervise-log", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped = load_evaluations(args.evaluations) if args.evaluations else {}
    panels = []
    if grouped:
        panels.append(("separation", lambda ax: panel_separation(ax, grouped)))
        panels.append(("bound", lambda ax: panel_bound(ax, grouped)))
    if args.posttrain_report is not None and args.posttrain_report.is_file():
        panels.append(("switching", lambda ax: panel_switching(ax, args.posttrain_report)))
    if args.supervise_log is not None and args.supervise_log.is_file():
        panels.append(("supervision", lambda ax: panel_supervision(ax, args.supervise_log)))
    if not panels:
        raise SystemExit("nothing to plot: pass --evaluations, --posttrain-report or --supervise-log")
    # Lay out only the panels there is data for, so a partial run does not
    # produce a figure that is mostly blank.
    columns = min(2, len(panels))
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5.8 * columns, 4.3 * rows), squeeze=False)
    flat = [axis for row in axes for axis in row]
    for axis, (_name, draw) in zip(flat, panels, strict=False):
        draw(axis)
    for axis in flat[len(panels) :]:
        axis.set_visible(False)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
