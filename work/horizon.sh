#!/usr/bin/env bash
# Error against rollout horizon, including two steps beyond the trained one.
#
# The evaluation is already multi-step -- an autoregressive rollout of H steps
# from 8 context frames -- and every sweep row records position_rmse_by_step,
# so the h = 1..6 curve comes from the runs that already exist.  It shows the
# method's advantage compounding: on narrow_s42's far band, position error
# grows 12.9x from h=1 to h=6 under continued JEPA training and 3.7x under the
# full method, so the gap widens from 1.85x at h=1 to 6.4x at h=6.
#
# What the existing runs cannot answer is whether that holds *past* the horizon
# the predictor was trained to roll out (rollout_horizon = 6).  The episodes are
# 16 frames and the context is 8, so H = 8 is the longest rollout the data
# admits: two steps of genuine extrapolation.  Post-training is untouched --
# this re-evaluates the same checkpoints at a longer horizon.
#
# The KKT block correctly reports feasible = null for these, because eps was
# calibrated at H = 6 and C's scale depends on the horizon; the certificate is
# read from the H = 6 evaluation and only the accuracy curve from this one.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
LOGS=$ROOT/work/logs
JOBID=${JOBID:-8093487}
GPU_NEXT=0
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
launch() {
  local name=$1; shift
  local gpu=$GPU_NEXT; GPU_NEXT=$(( (GPU_NEXT + 1) % 2 ))
  ( timeout --signal=TERM --kill-after=60 5400 \
      srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=5 --gres=gpu:2 \
        env CUDA_VISIBLE_DEVICES=$gpu PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
        "$@" > "$LOGS/$name.log" 2>&1
    printf '%s exit=%s\n' "$name" "$?" >> "$LOGS/_exits.txt" ) &
  say "  launched $name on gpu $gpu"
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
}
for model in narrow_s42 spatial_s42 spatial_s43 v2_s42; do
  hist=8; [ "$model" = v2_s42 ] && hist=6
  # v2 episodes are 12 frames: 6 + 6, so its only longer rollout is H = 6 from
  # a 6-frame context, which it already has.  Skip it.
  [ "$model" = v2_s42 ] && continue
  for variant in pretrained jepa factual penalty counterfactual; do
    src=$R/$model/post_mp_$variant
    [ -f "$src/checkpoint.pt" ] || continue
    out=$src/evaluation_h8.json
    [ -f "$out" ] && continue
    launch "evalh8_${model}_${variant}" "$PY" "$G/scripts/evaluate.py" \
      --checkpoint "$src/checkpoint.pt" --cache "$R/$model/latents.pt" \
      --probe-dir "$R/$model/probes" --output "$out" \
      --posttrain-report "$src/posttrain_report.json" \
      --history "$hist" --horizon 8 --stride 1 \
      --sweep-min 1.0 --sweep-max 9.0 --label "${model}__h8_${variant}" --device cuda
  done
done
wait
: > "$(dirname "$0")/.done-horizon"
say "=== done ==="
