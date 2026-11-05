#!/usr/bin/env bash
# The three numbers that decide whether the bouncing experiment is worth running,
# and at what budget.  All of them are measurable before any post-training step.
#
#   probe accuracy     the constraint is read through a frozen position probe;
#                      at 33.4 px (the published [CLS] head) there is no margin
#                      and nothing downstream means anything
#   discriminative     c_blind / eps -- how far outside the tolerance a
#     margin           gravity-blind predictor sits.  Needs to be >= 3 or so
#   C_0 / eps          sets the step budget (RESULTS.md section 6.7).  Above
#                      ~90 a 4000-step run spends every step in RESTORE,
#                      certifies nothing and destroys the world model
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
S=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
D=/fs/scratch/PAS3151/qijia387/gwm/data/bounce_single
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
srun_one() { srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 "$@"; }

dir=$S/bounce_s42
say "=== bounce 闸门指标 ==="
[ -f "$dir/latents.pt" ] || { say "  编码 latent cache"; \
  srun_one "$PY" "$G/scripts/build_cache.py" --checkpoint "$dir/checkpoint.pt" \
    --data "$D" --output "$dir/latents.pt" --device cuda \
    > $ROOT/work/logs/cache_bounce_s42.log 2>&1; say "  cache exit=$?"; }
[ -f "$dir/probes/probe_eval.pt" ] || { say "  训练两个探针"; \
  srun_one "$PY" "$G/scripts/train_probes.py" --cache "$dir/latents.pt" \
    --output-dir "$dir/probes" --device cuda --history 8 --horizon 6 --stride 1 \
    --max-episodes 6000 > $ROOT/work/logs/probes_bounce_s42.log 2>&1; say "  probes exit=$?"; }
: > $ROOT/work/.done-bounce_gates
say "=== bounce_gates done ==="
