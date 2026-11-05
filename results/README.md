# Experiment results

Every number in [`gravity-wm/docs/RESULTS.md`](../gravity-wm/docs/RESULTS.md) is computed
from the JSON in here, so a claim in the prose can be traced to the run that produced it.

Regenerate with:

    python gravity-wm/scripts/export_results.py \
      --runs  /path/to/gwm/runs --data /path/to/gwm/data --out results

| file | what it holds |
|---|---|
| `runs/*/train_report.json` | pretraining, per-epoch validation rows (per-step records dropped on export) |
| `runs/*/probes/probe_report.json` | position-probe accuracy, SNR, the discriminative margin, `eps` |
| `runs/*/post_*/posttrain_report.json` | calibration, the switching summary, the per-step constraint and loss history |
| `runs/*/post_*/evaluation.json` | the counterfactual sweep, latent error, the KKT block |
| `data/*/generation_manifest.json` | what each dataset is: scene, gravity law, residuals, and for the bouncing scene the measured restitution and usable-window share |

Not in here, because they belong in scratch storage:

- **checkpoints** (`*.pt`) — regenerate with `scripts/pretrain.py` and `scripts/posttrain.py`
- **datasets** (`*.lance`, 335 MB each) — regenerate with `scripts/gen_data.py` and `scripts/generate_bounce.py`;
  the manifests here record the seed and every scene parameter needed to reproduce them bit for bit
- **per-step pretraining history** — 310 MB of optimiser-step records
