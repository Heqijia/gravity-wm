"""Emit the result tables of ``docs/RESULTS.md`` as markdown, from the JSON.

The experiment log is hand-written prose around machine-generated tables, so
the tables have to be regenerable or they drift from the data.  This prints
them in the exact form the document carries; updating a section is a
copy-paste, and checking one is a diff.

    python scripts/results_tables.py --suite RUNS/suite
    python scripts/results_tables.py --suite RUNS/suite --section headline
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "gwm_report", Path(__file__).resolve().parent / "report.py"
)
_REPORT = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_REPORT)

VARIANT_LABEL = {
    "pretrained": "pretrained `theta_0`",
    "jepa": "continued JEPA training",
    "factual": "constraint at `g' = g`",
    "penalty": "penalty `lambda = 1`",
    "counterfactual": "**full method**",
}


def load(suite: Path, model: str, tag: str) -> dict | None:
    path = suite / model / f"post_{tag}" / "evaluation.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    payload.setdefault("variant", tag.split("_", 1)[-1])
    summary = _REPORT.summarise(payload)
    summary["kkt"] = payload.get("kkt", {})
    return summary


def average(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows if row is not None and row[key] is not None]
    return float(np.mean(values)) if values else float("nan")


def ratio(row: dict | None) -> str:
    kkt = (row or {}).get("kkt") or {}
    if not kkt.get("epsilon") or kkt.get("constraint_at_output") is None:
        return "no certified iterate"
    return f"{kkt['constraint_at_output'] / kkt['epsilon']:.2f}"


def headline(suite: Path, models: list[str], prefix: str) -> None:
    print(f"`{'` + `'.join(models)}`, `n = {len(models)}`.\n")
    print("| configuration | pos in | pos OOD | **pos far** | **resp far** | "
          "**slope** | JEPA L | feas |")
    print("|---|---|---|---|---|---|---|---|")
    for variant in _REPORT.ORDER:
        rows = [load(suite, model, f"{prefix}_{variant}") for model in models]
        rows = [row for row in rows if row is not None]
        if not rows:
            continue
        feasible = {row["feasible"] for row in rows}
        mark = "–" if feasible == {None} else ("yes" if feasible == {True} else "no")
        print(
            f"| {VARIANT_LABEL.get(variant, variant)} | {average(rows, 'pos_in'):.3f} | "
            f"{average(rows, 'pos_out'):.3f} | {average(rows, 'pos_far'):.3f} | "
            f"{average(rows, 'response_far'):.3f} | {average(rows, 'slope'):+.3f} | "
            f"{average(rows, 'jepa_loss'):.3f} | {mark} |"
        )


def certificate(suite: Path, rows: list[tuple[str, str, str]]) -> None:
    print("| configuration | `C/eps` s42 | `C/eps` s43 | certified | pos far | JEPA L |")
    print("|---|---|---|---|---|---|")
    for label, tag, tag43 in rows:
        row = load(suite, "spatial_s42", tag)
        if row is None:
            continue
        other = load(suite, "spatial_s43", tag43) if tag43 else None
        mark = {True: "yes", False: "no", None: "no"}[row["feasible"]]
        print(
            f"| {label} | {ratio(row)} | {ratio(other) if other else '–'} | {mark} | "
            f"{row['pos_far']:.3f} | {row['jepa_loss']:.3f} |"
        )


def ablations(suite: Path, rows: list[tuple[str, str]]) -> None:
    print("| ablation | pos far | resp far | slope | JEPA L | feas |")
    print("|---|---|---|---|---|---|")
    for label, tag in rows:
        row = load(suite, "spatial_s42", tag)
        if row is None:
            continue
        mark = {True: "yes", False: "no", None: "no"}[row["feasible"]]
        print(
            f"| {label} | {row['pos_far']:.3f} | {row['response_far']:.3f} | "
            f"{row['slope']:+.3f} | {row['jepa_loss']:.3f} | {mark} |"
        )


def horizon(suite: Path, model: str, filename: str, band: str) -> None:
    """Position error at each rollout step, averaged over a gravity band.

    The evaluation is multi-step -- an autoregressive rollout from the context
    -- and the headline tables report the error at the *final* step.  Every
    sweep row also records `position_rmse_by_step`, which is where this comes
    from: no extra run is needed for `h = 1..H`.  The interesting quantity is
    the growth factor: a constraint that controls error *accumulation* flattens
    the curve, which is what Corollary 2's `(2s+1) delta + dt^2 sqrt(...) sqrt(C)`
    predicts and what a single final-step number hides.
    """

    rows = []
    for variant in _REPORT.ORDER:
        path = suite / model / f"post_mp_{variant}" / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        sweep = payload["counterfactual_sweep"]
        mean = float(payload["train_gravity"]["mean"])
        scale = float(payload["pixels_per_unit"])
        gravity = np.array([r["gravity"] for r in sweep])
        keep = gravity >= mean + 3.0 if band == "far" else np.abs(gravity - mean) <= 1.0
        curve = np.array([r["position_rmse_by_step"] for r in sweep])[keep].mean(0) * scale
        rows.append((VARIANT_LABEL.get(variant, variant), curve))
    if not rows:
        print(f"no evaluations for {model}/{filename}")
        return
    steps = len(rows[0][1])
    header = "| variant | " + " | ".join(f"h={i}" for i in range(1, steps + 1)) + " | h/h1 |"
    described = "far band `g' >= g+3`" if band == "far" else "near band `|g'-g| <= 1`"
    print(f"`{model}`, {described}, position RMSE in px.\n")
    print(header)
    print("|---" * (steps + 2) + "|")
    for label, curve in rows:
        cells = " | ".join(f"{value:.2f}" for value in curve)
        print(f"| {label} | {cells} | {curve[-1] / curve[0]:.1f}x |")


REGIMES = [
    ("ball_v1, g ~ N(4, 0.5^2)", ["spatial_s42", "spatial_s43"], 2.268),
    ("ball_narrow, g ~ N(4, 0.05^2)", ["narrowfx_s42", "narrowfx_s43"], 0.227),
    ("ball_single, g == 4 exactly", ["single_s42", "single_s43"], 0.0),
]


def regimes(suite: Path) -> None:
    """How the separation depends on how much gravity information the data has.

    The three datasets differ in one thing: the spread of the training gravity,
    and therefore how much the object's motion varies with it.  Measured as
    pixels of vertical displacement at the end of a six-step rollout, one
    standard deviation of the training gravity moves the object 2.27 px on
    ball_v1, 0.23 px on ball_narrow -- against a probe accuracy of 0.17 px --
    and exactly nothing on ball_single.

    This is the table Remark 11 is about.  Where the pretrained predictor can
    already read gravity off the data it extrapolates by itself and there is
    little for a constraint to add; where it cannot, the constraint is the only
    signal and the separation is an order of magnitude.
    """

    print("| regime | 1 sd of train g, in px at h=6 | pretrained resp | "
          "continued JEPA | full method | gain | slope |")
    print("|---|---|---|---|---|---|---|")
    for label, models, pixels in REGIMES:
        base = [load(suite, m, "mp_pretrained") for m in models]
        jepa = [load(suite, m, "mp_jepa") for m in models]
        full = [load(suite, m, "mp_counterfactual") for m in models]
        if not all(base) or not all(jepa) or not all(full):
            print(f"| {label} | {pixels:.2f} | *incomplete* | | | | |")
            continue
        j, f = average(jepa, "pos_far"), average(full, "pos_far")
        print(
            f"| {label} | {pixels:.2f} | {average(base, 'response_far'):+.3f} | "
            f"{j:.3f} px | **{f:.3f} px** | **{j / max(f, 1e-9):.1f}x** | "
            f"{average(jepa, 'slope'):+.2f} -> {average(full, 'slope'):+.2f} |"
        )


BUDGET_MODELS = [
    "spatial_s42", "narrow_s42", "narrowfx_s42", "single_s42",
    "sq_single_s42", "ssm_s42", "ax_s42",
]


def budget(suite: Path, tag: str = "mp_counterfactual") -> None:
    """Why some runs keep the world model and others destroy it.

    ``eps`` is calibrated against the predictor's *own* factual rollout error,
    while ``C_0`` is set by how wrong its counterfactual is.  A predictor that
    is better factually therefore gets a tighter tolerance and starts
    relatively further outside it, and a large ``C_0/eps`` consumes the whole
    step budget in RESTORE.  A run that never reaches OPTIMISE has no
    certificate -- Theorem 1(c) reads it off an OPTIMISE iterate -- and nothing
    holding the JEPA loss down.

    The point of the table is that ``C_0/eps`` is computable before training,
    so the budget a model needs is predictable rather than discovered by
    wasting a run.
    """

    print("| model | `eps` | `C_0` | `C_0/eps` | RESTORE | OPTIMISE | `L` start | `L` end | ratio |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for model in BUDGET_MODELS:
        path = suite / model / f"post_{tag}" / "posttrain_report.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        history = payload.get("history") or []
        summary = payload.get("summary") or {}
        losses = [row["jepa_loss"] for row in history if "jepa_loss" in row]
        # Derived from the history rather than read from the summary, because
        # the summary field postdates most of these runs.
        constraints = [
            row["constraint"]
            for row in history
            if isinstance(row.get("constraint"), float) and row["constraint"] == row["constraint"]
        ]
        if not losses or not constraints:
            continue
        start = constraints[0] / payload["epsilon"]
        rows.append((model, payload["epsilon"], start * payload["epsilon"], start,
                     summary.get("restore_steps"), summary.get("optimise_steps"),
                     losses[0], losses[-1]))
    for model, eps, c0, ratio, restore, optimise, first, last in sorted(rows, key=lambda r: -r[1]):
        mark = "**" if optimise == 0 else ""
        print(
            f"| `{model}` | {eps:.4f} | {c0:.3f} | {ratio:.1f} | {mark}{restore}{mark} | "
            f"{mark}{optimise}{mark} | {first:.2f} | {mark}{last:.2f}{mark} | "
            f"{mark}{last / first:.2f}x{mark} |"
        )
    if len(rows) > 2:
        eps = np.array([r[1] for r in rows])
        quotient = np.array([r[3] for r in rows])
        print(
            f"\nCorrelation of `log eps` with `log(C_0/eps)`: "
            f"**{np.corrcoef(np.log(eps), np.log(quotient))[0, 1]:+.3f}**  "
            f"(n = {len(rows)})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--horizon-model",
        default="narrow_s42",
        help="which base model the horizon curve is drawn for",
    )
    parser.add_argument(
        "--horizon-file",
        default="evaluation.json",
        help="evaluation_h8.json for the rollout that goes past the trained horizon",
    )
    parser.add_argument(
        "--section",
        default="all",
        choices=(
            "all", "headline", "forms", "certificate", "ablations", "models",
            "horizon", "regimes", "budget",
        ),
    )
    args = parser.parse_args()
    suite = args.suite
    both = ["spatial_s42", "spatial_s43"]

    if args.section in ("all", "headline"):
        print("### 6.1 Headline: the five variants, position form, 2 seeds\n")
        headline(suite, both, "mp")
        print()

    if args.section in ("all", "forms"):
        print("### 6.2 The constraint form is the decisive design choice\n")
        print("| form | pos far | resp far | slope | JEPA L |")
        print("|---|---|---|---|---|")
        for label, prefix in [
            ("Definition 1 (acceleration)", "s1"),
            ("**Proposition 1 (position)**", "mp"),
        ]:
            rows = [load(suite, m, f"{prefix}_counterfactual") for m in both]
            rows = [r for r in rows if r is not None]
            print(
                f"| {label} | {average(rows, 'pos_far'):.3f} | "
                f"{average(rows, 'response_far'):.3f} | {average(rows, 'slope'):+.3f} | "
                f"{average(rows, 'jepa_loss'):.3f} |"
            )
        rows = [load(suite, m, "mp_jepa") for m in both]
        rows = [r for r in rows if r is not None]
        print(
            f"| *(continued JEPA, for reference)* | *{average(rows, 'pos_far'):.3f}* | "
            f"*{average(rows, 'response_far'):.3f}* | *{average(rows, 'slope'):+.3f}* | "
            f"*{average(rows, 'jepa_loss'):.3f}* |"
        )
        print()

    if args.section in ("all", "certificate"):
        print("### 6.3 The certificate\n")
        certificate(
            suite,
            [
                ("continued JEPA training", "mp_jepa", "mp_jepa"),
                ("constraint at `g' = g`", "mp_factual", "mp_factual"),
                ("penalty `lambda = 1`", "pq_lam1.0", "pq_lam1.0"),
                ("penalty `lambda = 10`", "pq_lam10.0", "pq_lam10.0"),
                ("penalty `lambda = 100`", "pq_lam100.0", "pq_lam100.0"),
                ("penalty `lambda = 300`", "pq_lam300.0", ""),
                ("penalty `lambda = 1000`", "pq_lam1000.0", "pq_lam1000.0"),
                ("penalty `lambda = 3000`", "pq_lam3000.0", ""),
                ("**full method, `eps`**", "mp_counterfactual", "mp_counterfactual"),
                ("full method, `eps x 0.8`", "pq_eps0.8", "pq_eps0.8"),
                ("full method, `eps x 0.6`", "pq_eps0.6", "pq_eps0.6"),
                ("full method, `eps x 0.45`", "pq_eps0.45", ""),
                ("full method, `eps x 0.3`", "pq_eps0.3", ""),
            ],
        )
        print()

    if args.section in ("all", "ablations"):
        print("### 6.4 Ablations\n")
        ablations(
            suite,
            [
                ("default", "mp_counterfactual"),
                ("`gamma = 0.5` (the PDF's value)", "ab_gate0.5"),
                ("no hysteresis", "ab_nohyst"),
                ("no projection", "ab_noproject"),
                ("no in-frame filter", "ab_noframe"),
                ("RESTORE normalised", "ab_restore_nrm"),
                ("**RESTORE = `grad C / eps`** (the PDF's)", "ab_restore_eps"),
                ("**`eps` from `real_latents`**", "ab_eps_real"),
            ],
        )
        print()

    if args.section in ("all", "regimes"):
        print("### The separation against how much gravity the data shows\n")
        regimes(suite)
        print()

    if args.section in ("all", "budget"):
        print("### The step budget a model needs is predictable\n")
        budget(suite)
        print()

    if args.section in ("all", "horizon"):
        print("### Error against rollout horizon\n")
        for band in ("far", "near"):
            horizon(suite, args.horizon_model, args.horizon_file, band)
            print()

    if args.section in ("all", "models"):
        print("### 6.5 Base models, position form\n")
        print("| base model | pretrained pos far | best method pos far | resp far | "
              "JEPA L | feas |")
        print("|---|---|---|---|---|---|")
        for model in (
            "spatial_s42", "spatial_s43", "lewm1step_s42", "ssm_s42",
            "narrow_s42", "v2_s42", "cls_s42",
        ):
            base = load(suite, model, "mp_pretrained")
            best = load(suite, model, "mp_counterfactual")
            if base is None or best is None:
                print(f"| `{model}` | – | *not finished* | – | – | – |")
                continue
            mark = {True: "yes", False: "no", None: "no"}[best["feasible"]]
            print(
                f"| `{model}` | {base['pos_far']:.3f} | {best['pos_far']:.3f} | "
                f"{best['response_far']:.3f} | {best['jepa_loss']:.3f} | {mark} |"
            )
        print()


if __name__ == "__main__":
    main()
