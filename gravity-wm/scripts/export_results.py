"""Copy the experiment results into the repository, small enough to version.

The raw run tree is 356 MB, 310 MB of which is ``train_report.json``: one
record per optimiser step of pretraining.  That detail belongs in scratch.
What a reader needs to check a claim is the per-run summary, the calibration
that produced ``eps``, the evaluation sweep and the dataset manifests -- about
23 MB, which fits in git and makes every table in RESULTS.md traceable to the
JSON it came from.

    python scripts/export_results.py --runs RUNS --data DATA --out results
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Kept whole: these are what the tables are computed from.
WHOLE = ("posttrain_report.json", "evaluation.json", "evaluation_ax.json",
         "probe_report.json", "generation_manifest.json")
# Kept with the per-step history dropped.
TRIMMED = ("train_report.json",)


def trim_history(payload: dict) -> dict:
    """Drop per-step records, keep the per-epoch validation rows.

    The validation rows are what §6.15's collapse diagnosis reads, and they are
    a few dozen entries against tens of thousands of step records.
    """

    history = payload.get("history")
    if isinstance(history, list):
        payload = dict(payload)
        payload["history"] = [row for row in history if row.get("split") == "validation"]
        payload["history_note"] = (
            "per-step records dropped on export; validation rows kept. "
            "The full history stays in the run directory."
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    kept = dropped = 0
    total = 0
    for root, names in ((args.runs, WHOLE + TRIMMED), (args.data, ("generation_manifest.json",))):
        if not root.is_dir():
            print(f"missing: {root}")
            continue
        for name in names:
            for source in sorted(root.rglob(name)):
                if ".stale" in source.name:
                    dropped += 1
                    continue
                target = args.out / source.relative_to(root.parent)
                target.parent.mkdir(parents=True, exist_ok=True)
                if name in TRIMMED:
                    payload = trim_history(json.loads(source.read_text()))
                    target.write_text(json.dumps(payload, indent=1, sort_keys=True))
                else:
                    shutil.copy2(source, target)
                kept += 1
                total += target.stat().st_size

    print(f"exported {kept} files, skipped {dropped}, {total / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
