"""Render a contact sheet of one dataset: what the model is trained on, and what it is asked for.

Every dataset in this study differs from `ball_v1` in exactly one respect -- the
spread of the training gravity, the frame interval, the object's shape, or the
presence of a fourth action coordinate -- and the point each one makes is
visual.  A sheet puts the training rows above the counterfactual rows so the
gap the constraint has to cross is the thing you see first.

    python scripts/plot_samples.py --dataset DATA/ball_single --out docs/figures/ball_single.png
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import lance
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def episode_table(dataset: Path):
    table = lance.dataset(str(dataset / "data.lance")).to_table(
        columns=["episode_idx", "step_idx", "pixels", "state", "action", "gravity", "split_id"]
    ).to_pydict()
    episode = np.asarray(table["episode_idx"])
    first = np.unique(episode, return_index=True)[1]
    return table, episode, first


def choose_rows(table, episode, first, *, counterfactuals):
    """Training rows spanning the training spread, then the counterfactual rows.

    When the training gravity is a single value the three training rows are
    three different launches at that one gravity, which is the point: nothing
    in the training data varies the quantity the constraint is about.
    """

    gravity = np.asarray(table["gravity"])[first]
    split = np.asarray(table["split_id"])[first]
    ids = np.asarray(table["episode_idx"])[first]
    train, held = split == 0, split == 1
    spread = float(gravity[train].std()) if train.any() else 0.0

    rows = []
    if spread > 1e-6:
        for target, tag in ((gravity[train].mean() - 3.8 * spread, "-3.8 sd"),
                            (gravity[train].mean(), "mean"),
                            (gravity[train].mean() + 3.8 * spread, "+3.8 sd")):
            k = int(ids[train][np.argmin(np.abs(gravity[train] - target))])
            rows.append((k, f"train  g={gravity[ids == k][0]:.2f}  ({tag})"))
    else:
        for n, k in enumerate(ids[train][:3]):
            rows.append((int(k), f"train  g={gravity[train][0]:.2f}  (launch {n + 1})"))

    for target in counterfactuals:
        if not held.any():
            continue
        k = int(ids[held][np.argmin(np.abs(gravity[held] - target))])
        rows.append((k, f"held-out  g={gravity[ids == k][0]:.2f}  COUNTERFACTUAL"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--counterfactuals", type=float, nargs="*", default=[1.0, 9.0])
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "generation_manifest.json").read_text())
    table, episode, first = episode_table(args.dataset)
    rows = choose_rows(table, episode, first, counterfactuals=args.counterfactuals)
    step = np.asarray(table["step_idx"])

    figure, axes = plt.subplots(len(rows), args.columns,
                                figsize=(1.9 * args.columns, 1.9 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (eid, label) in enumerate(rows):
        where = np.where(episode == eid)[0]
        where = where[np.argsort(step[where])]
        stride = max(1, len(where) // args.columns)
        for c, k in enumerate(where[::stride][: args.columns]):
            axes[r, c].imshow(Image.open(io.BytesIO(table["pixels"][k])))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"t={step[k]}", fontsize=9)
        axes[r, 0].set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center")

    actions = manifest.get("action_names", [])
    title = args.title or (
        f"{args.dataset.name}: {manifest.get('task')}, {manifest.get('image_size')}px, "
        f"dt={manifest.get('dt')}, actions={actions}\n"
        "training rows above, counterfactual rows below -- the counterfactuals have no video in training"
    )
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"{args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
