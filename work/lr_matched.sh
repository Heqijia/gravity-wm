#!/usr/bin/env bash
# Rerun the Semigroup-JEPA baselines at a learning rate matched to their own
# gradient scale.
#
# Post-training takes a plain SGD step, so the step *length* is lr ||grad L||
# and one learning rate does not serve every predictor.  Measured at theta_0
# with scripts/measure_gradient_scale.py (24 batches, median):
#
#   model            L(theta_0)   ||grad L||   step at lr 1e-4   lr for step 0.045
#   spatial_s42            48.6          556             0.056            8.1e-5
#   lewm1step_s42         392.6        2 858             0.286            1.6e-5
#   ssm_s42                 8.26      32 768             3.28             1.4e-6
#
# The SSM's gradient is 59x the GRU's.  At lr 1e-4 its step is 3.28 -- and the
# 1e4 clip engaged at step 0, silently turning the step into a normalised one
# of length 1.0, which design.md warns about explicitly.  Continued JEPA
# training *alone* then diverges on it (L 8.7 -> 1320 in 200 steps), so the
# acceleration-form "SSM failure" and the position-form one are both failures
# of the step size, not of the constraint.  Reporting either as a result about
# the method would be wrong.
#
# lewm1step survived lr 1e-4 (C_end 0.435 inside eps 0.489, 1386 RESTORE
# steps) but its step was 5x the GRU's, so it is rerun too -- the comparison
# has to be at a step size chosen by the same rule for every base model.
#
# The diverged runs are kept as post_mpdiv_* rather than deleted.
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
  (
    timeout --signal=TERM --kill-after=60 5400 \
      srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=5 --gres=gpu:2 \
        env CUDA_VISIBLE_DEVICES=$gpu PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
        "$@" > "$LOGS/$name.log" 2>&1
    printf '%s exit=%s\n' "$name" "$?" >> "$LOGS/_exits.txt"
  ) &
  say "  launched $name on gpu $gpu"
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
}

VARIANTS="pretrained jepa factual penalty counterfactual"

say "=== baselines at a matched step size ==="
for spec in "ssm_s42 1.4e-6" "lewm1step_s42 1.6e-5"; do
  set -- $spec
  model=$1 lr=$2
  [ -f "$R/$model/probes/probe_eval.pt" ] || { say "  !! $model has no probes"; continue; }
  # set the previous, mis-scaled runs aside rather than deleting them
  for variant in $VARIANTS; do
    old=$R/$model/post_mp_$variant
    [ -d "$old" ] && [ ! -d "$R/$model/post_mpdiv_$variant" ] && mv "$old" "$R/$model/post_mpdiv_$variant"
  done
  say "  $model at lr=$lr"
  for variant in $VARIANTS; do
    out=$R/$model/post_mp_$variant
    [ -f "$out/posttrain_report.json" ] && continue
    launch "post_${model}_mplr_${variant}" "$PY" "$G/scripts/posttrain.py" \
      --checkpoint "$R/$model/checkpoint.pt" --cache "$R/$model/latents.pt" \
      --probe-dir "$R/$model/probes" --output-dir "$out" --variant "$variant" \
      --steps 4000 --lr "$lr" --history 8 --horizon 6 --stride 1 \
      --constraint-form position --epsilon-source factual_rollout \
      --gravity-low 1.0 --gravity-high 9.0 --seed 42 --eval-every 200 --device cuda
  done
done
wait
say "evaluating"
for model in ssm_s42 lewm1step_s42; do
  for variant in $VARIANTS; do
    out=$R/$model/post_mp_$variant
    [ -f "$out/checkpoint.pt" ] || continue
    [ -f "$out/evaluation.json" ] && continue
    launch "eval_${model}_mplr_${variant}" "$PY" "$G/scripts/evaluate.py" \
      --checkpoint "$out/checkpoint.pt" --cache "$R/$model/latents.pt" \
      --probe-dir "$R/$model/probes" --output "$out/evaluation.json" \
      --posttrain-report "$out/posttrain_report.json" \
      --history 8 --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
      --label "${model}__mp_${variant}" --device cuda
  done
done
wait
say "=== done ==="
