# Phys-WM

Gravity-constrained post-training of JEPA world models.

- [`gravity-wm/`](gravity-wm/) — the study: generators, constraint, post-training, evaluation, tests.
  Start at [`gravity-wm/docs/RESULTS.md`](gravity-wm/docs/RESULTS.md) (中文版：[`RESULTS.zh.md`](gravity-wm/docs/RESULTS.zh.md)).
- [`work/`](work/) — the queue scripts that drive the experiment suite on a Slurm allocation.

## Not in this repository

- **`sg-jepa/`** — the pretrained world model this builds on, cloned separately from
  <https://github.com/sg-jepa/sg-jepa>.
- **Datasets and checkpoints** — generated into scratch storage, not version-controlled.
  Regenerate with `gravity-wm/scripts/generate_freefall.py` and `generate_bounce.py`.
- **`.venv-wm/`** — the virtual environment.
