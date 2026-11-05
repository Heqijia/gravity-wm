#!/usr/bin/env bash
# The horizon curve past the training horizon, on one consistent window set.
#
# Training rolls out 6 steps.  Evaluating at horizon 12 and reading
# ``position_rmse_by_step`` gives h = 1..12 from a single pass, so every point
# on the curve comes from the same windows.  Running separate evaluations per
# horizon would not: the counterfactual floor filter accepts 14% of windows at
# h = 6 and 1.5% at h = 12, so the two populations differ and the curve would
# mix "longer horizon" with "different windows".
#
# 1.5% is a hard selection and the surviving windows are the high, rising ones
# whose counterfactual stays airborne for 1.9 s.  That is a caveat on the
# result, not a defect in it -- those are the windows on which the free-fall
# identity the constraint enforces is actually true at h = 12.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
M=$R/bounce_r08_s42
FLOOR=1.60
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
say "=== bounce horizon 12 评估(训练 horizon = 6)==="
for t in mp_pretrained mp_jepa mp_factual mp_penalty b3_counterfactual s43_counterfactual; do
  out=$M/post_$t
  [ -f "$out/checkpoint.pt" ] || continue
  [ -f "$out/evaluation_h12.json" ] && { say "  $t 已有"; continue; }
  srun --jobid=8093487 --overlap -n1 --cpus-per-task=4 --gres=gpu:2 \
    env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
    "$PY" "$G/scripts/evaluate.py" --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output "$out/evaluation_h12.json" \
    --posttrain-report "$out/posttrain_report.json" \
    --history 8 --horizon 12 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
    --floor-resting-z $FLOOR --label "bounce_r08__h12_$t" --device cuda \
    > $ROOT/work/logs/eval_h12_$t.log 2>&1
  say "  $t exit=$?"
done
: > $ROOT/work/.done-bounce_h12
say "=== bounce_h12 done ==="
