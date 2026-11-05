#!/usr/bin/env bash
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
M=$R/bounce_r08_s42
FLOOR=1.60
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
say "=== bounce 评估(含地面过滤)==="
for v in pretrained jepa factual penalty counterfactual; do
  out=$M/post_mp_$v
  [ -f "$out/checkpoint.pt" ] || { say "  $v 无 checkpoint,跳过"; continue; }
  [ -f "$out/evaluation.json" ] && { say "  $v 已有"; continue; }
  srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
    env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
    "$PY" "$G/scripts/evaluate.py" --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output "$out/evaluation.json" \
    --posttrain-report "$out/posttrain_report.json" \
    --history 8 --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
    --floor-resting-z $FLOOR --label "bounce_r08__mp_$v" --device cuda \
    > $ROOT/work/logs/eval_bounce_$v.log 2>&1
  say "  $v exit=$?"
done
: > $ROOT/work/.done-bounce_eval
say "=== bounce_eval done ==="
