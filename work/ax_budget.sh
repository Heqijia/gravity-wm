#!/usr/bin/env bash
# The two-dimensional counterfactual, at the budget its C_0/eps implies.
#
# Measured on ball_ax before committing the runs: C_0/eps = 105.7, above the
# rotating square's 100.9 and well above ball_single's 55.0.  By the rule the
# six-model table establishes, a 4000-step run will spend every step in RESTORE,
# reach no OPTIMISE step, and therefore produce neither a certificate nor any
# protection for the JEPA loss -- the first RESTORE step alone already took the
# loss from 13.37 to 700.9.
#
# C_0 is larger here for a structural reason, not a tuning one: the
# counterfactual acceleration differs from the factual one in *two* components
# instead of one, so the initial violation is larger at the same eps.
#
# The two arms are the same remedies the square gets, so the comparison is
# controlled across both new environments:
#   b6    24000 steps at the calibrated eps -- more budget, nothing else
#   an    eps x 8 first, then the calibrated eps from there -- a staged
#         tolerance, so OPTIMISE steps happen in both stages
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
LOGS=$ROOT/work/logs
M=$R/ax_s42
AX=3
JOBID=${JOBID:-8093487}
GPU_NEXT=0
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
launch() {
  local name=$1; shift
  local gpu=$GPU_NEXT; GPU_NEXT=$(( (GPU_NEXT + 1) % 2 ))
  ( timeout --signal=TERM --kill-after=60 21600 \
      srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=5 --gres=gpu:2 \
        env CUDA_VISIBLE_DEVICES=$gpu PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
        "$@" > "$LOGS/$name.log" 2>&1
    printf '%s exit=%s\n' "$name" "$?" >> "$LOGS/_exits.txt" ) &
  say "  launched $name on gpu $gpu"
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
}
post() { # post <tag> <checkpoint> <steps> <epsilon-scale>
  local tag=$1 ckpt=$2 steps=$3 scale=$4 out=$M/post_$1
  [ -f "$out/posttrain_report.json" ] && return
  launch "post_ax_s42_${tag}" "$PY" "$G/scripts/posttrain.py" \
    --checkpoint "$ckpt" --cache "$M/latents.pt" --probe-dir "$M/probes" \
    --output-dir "$out" --variant counterfactual --steps "$steps" --lr 1e-4 \
    --history 8 --horizon 6 --stride 1 --constraint-form position \
    --epsilon-source factual_rollout --epsilon-scale "$scale" \
    --gravity-low 1.0 --gravity-high 9.0 \
    --accel-x-index "$AX" --accel-x-low -3.0 --accel-x-high 3.0 \
    --seed 42 --eval-every 800 --device cuda
}
say "=== ball_ax: more budget, and a staged tolerance ==="
post b6_counterfactual "$M/checkpoint.pt" 24000 1.0
post an1_counterfactual "$M/checkpoint.pt" 4000 8.0
wait
post an2_counterfactual "$M/post_an1_counterfactual/checkpoint.pt" 4000 1.0
wait
for tag in b6_counterfactual an1_counterfactual an2_counterfactual; do
  out=$M/post_$tag
  [ -f "$out/checkpoint.pt" ] || continue
  for h in 6 8; do
    target=$out/evaluation.json; [ "$h" = 8 ] && target=$out/evaluation_h8.json
    [ -f "$target" ] && continue
    launch "eval_ax_s42_${tag}_h$h" "$PY" "$G/scripts/evaluate.py" \
      --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
      --probe-dir "$M/probes" --output "$target" \
      --posttrain-report "$out/posttrain_report.json" \
      --history 8 --horizon "$h" --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
      --label "ax_s42__${tag}" --device cuda
  done
  [ -f "$out/evaluation_ax.json" ] || launch "eval_ax_s42_${tag}_ax" "$PY" "$G/scripts/evaluate.py" \
    --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output "$out/evaluation_ax.json" \
    --posttrain-report "$out/posttrain_report.json" \
    --history 8 --horizon 6 --stride 1 \
    --accel-x-index "$AX" --accel-x-min -3.0 --accel-x-max 3.0 --accel-x-step 0.25 \
    --label "ax_s42__ax_${tag}" --device cuda
done
wait
: > "$ROOT/work/.done-ax_budget"
say "=== ax_budget done ==="
