#!/usr/bin/env bash
# Which half of the two-dimensional counterfactual breaks it?
#
# On ball_ax neither remedy worked.  24000 steps moved C/eps only from 105.7 to
# 37.9 while the JEPA loss reached 7.6e12, and the staged arm was invalidated by
# a flaw in the staging itself (each stage recalibrated eps against the damaged
# model, so stage two's eps came out 773x the original and the constraint went
# vacuous).  So RESTORE on the 2D constraint reduces C slowly while damaging L
# fast -- qualitatively unlike the 1D case, where the rotating square drove
# C/eps from 100.9 to 1.1 in 4000 steps.
#
# Three runs decompose it.  All on the same ax_s42 checkpoint, same budget, same
# pinned tolerance, differing only in which components of the counterfactual
# acceleration are allowed to move:
#
#   gonly   g' ~ U(1, 9),  a_x' = 0        the scalar problem on the 4-coordinate
#                                          model -- if this behaves like
#                                          single_s42, the 4th coordinate and the
#                                          retrained encoder are not the problem
#   axonly  g' = 4,        a_x' ~ U(-3, 3) the horizontal direction alone
#   both    g' ~ U(1, 9),  a_x' ~ U(-3, 3) the 2D problem again, for reference
#
# If gonly works and axonly fails, the horizontal direction is intrinsically
# harder and the reason is worth finding.  If both work separately and only
# their combination fails, the two constraints conflict.  Either answer is a
# result; "the 2D experiment did not work" is not.
#
# eps is pinned to the ball_ax calibration (0.0141) in all three so the
# comparison is at one tolerance.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
LOGS=$ROOT/work/logs
M=$R/ax_s42
AX=3
EPS=0.0141
JOBID=${JOBID:-8093487}
GPU_NEXT=0
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
launch() {
  local name=$1; shift
  local gpu=$GPU_NEXT; GPU_NEXT=$(( (GPU_NEXT + 1) % 2 ))
  ( timeout --signal=TERM --kill-after=60 10800 \
      srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=5 --gres=gpu:2 \
        env CUDA_VISIBLE_DEVICES=$gpu PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
        "$@" > "$LOGS/$name.log" 2>&1
    printf '%s exit=%s\n' "$name" "$?" >> "$LOGS/_exits.txt" ) &
  say "  launched $name on gpu $gpu"
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
}
post() { # post <tag> <g-low> <g-high> <ax-low> <ax-high>
  local tag=$1 glo=$2 ghi=$3 axlo=$4 axhi=$5 out=$M/post_$1
  [ -f "$out/posttrain_report.json" ] && return
  launch "post_ax_s42_${tag}" "$PY" "$G/scripts/posttrain.py" \
    --checkpoint "$M/checkpoint.pt" --cache "$M/latents.pt" --probe-dir "$M/probes" \
    --output-dir "$out" --variant counterfactual --steps 8000 --lr 1e-4 \
    --history 8 --horizon 6 --stride 1 --constraint-form position \
    --epsilon-source factual_rollout --epsilon "$EPS" \
    --gravity-low "$glo" --gravity-high "$ghi" \
    --accel-x-index "$AX" --accel-x-low "$axlo" --accel-x-high "$axhi" \
    --seed 42 --eval-every 400 --device cuda
}
say "=== decomposing the two-dimensional counterfactual ==="
post dc_gonly  1.0 9.0  0.0  0.0
post dc_axonly 4.0 4.0001 -3.0 3.0
wait
for tag in dc_gonly dc_axonly; do
  out=$M/post_$tag
  [ -f "$out/checkpoint.pt" ] || continue
  [ -f "$out/evaluation.json" ] || launch "eval_ax_s42_${tag}" "$PY" "$G/scripts/evaluate.py" \
    --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" --probe-dir "$M/probes" \
    --output "$out/evaluation.json" --posttrain-report "$out/posttrain_report.json" \
    --history 8 --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
    --label "ax_s42__${tag}" --device cuda
  [ -f "$out/evaluation_ax.json" ] || launch "eval_ax_s42_${tag}_ax" "$PY" "$G/scripts/evaluate.py" \
    --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" --probe-dir "$M/probes" \
    --output "$out/evaluation_ax.json" --posttrain-report "$out/posttrain_report.json" \
    --history 8 --horizon 6 --stride 1 \
    --accel-x-index "$AX" --accel-x-min -3.0 --accel-x-max 3.0 --accel-x-step 0.25 \
    --label "ax_s42__ax_${tag}" --device cuda
done
wait
: > "$ROOT/work/.done-ax_decompose"
say "=== ax_decompose done ==="
