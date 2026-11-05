#!/usr/bin/env bash
# Option A, the full five-variant comparison on the bouncing scene.
#
# Budget from C_0/eps = 36.2, measured before committing (RESULTS.md 6.7).
# That sits between narrowfx (26.9, which reached OPTIMISE at 4000 steps with
# 760 of them) and single (55.0, which needed 12000 to bring C/eps to 0.73).
# 12000 steps is the value that worked at the harder end, so it is the safe
# choice rather than discovering a starved run after the fact.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
M=$R/bounce_r08_s42
FLOOR=1.60
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
one() { srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 "$@"; }

say "=== bounce 五变体后训练, 12000 步 (C_0/eps = 36.2) ==="
for v in pretrained jepa factual penalty counterfactual; do
  out=$M/post_mp_$v
  [ -f "$out/posttrain_report.json" ] && { say "  $v 已有"; continue; }
  steps=12000; [ "$v" = pretrained ] && steps=0
  one "$PY" "$G/scripts/posttrain.py" --checkpoint "$M/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output-dir "$out" --variant "$v" --steps $steps --lr 1e-4 \
    --history 8 --horizon 6 --stride 1 --constraint-form position \
    --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
    --floor-resting-z $FLOOR --seed 42 --eval-every 600 --device cuda \
    > $ROOT/work/logs/post_bounce_$v.log 2>&1
  say "  $v exit=$?"
done
: > $ROOT/work/.done-bounce_suite
say "=== bounce_suite done ==="
