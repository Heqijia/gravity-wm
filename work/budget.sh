#!/usr/bin/env bash
# Is the missing certificate a step-budget limit rather than a method limit?
#
# The counterfactual runs land close to their tolerance but not always inside,
# and how close tracks how far outside they started:
#
#   model          C_0/eps   C_out/eps   certified   restore/optimise
#   spatial_s42        2.0        0.85   yes             2402 / 1598
#   narrow_s42        17.5        1.36   no              1767 / 2233
#   narrowfx_s42      26.9        0.88   yes             3240 /  760
#   single_s42        55.0        1.76   no              3708 /  292
#   ssm_s42           87.5      (none)   never left RESTORE
#
# single_s42 spent 93% of its budget restoring feasibility and still ended at
# 1.76 eps, so the obvious question is whether 4000 steps is simply not enough
# to cross 55x and then do any optimising.  This triples the budget on the two
# runs that missed.  If they certify, the certificate was budget-limited and
# the headline accuracy numbers are unaffected either way -- they already hold
# at 4000 steps.
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
  ( timeout --signal=TERM --kill-after=60 16200 \
      srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=5 --gres=gpu:2 \
        env CUDA_VISIBLE_DEVICES=$gpu PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
        "$@" > "$LOGS/$name.log" 2>&1
    printf '%s exit=%s\n' "$name" "$?" >> "$LOGS/_exits.txt" ) &
  say "  launched $name on gpu $gpu"
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
}
say "=== 12000 steps on the runs that missed their certificate ==="
for model in single_s42 narrow_s42; do
  out=$R/$model/post_b3_counterfactual
  [ -f "$out/posttrain_report.json" ] && continue
  launch "post_${model}_b3" "$PY" "$G/scripts/posttrain.py" \
    --checkpoint "$R/$model/checkpoint.pt" --cache "$R/$model/latents.pt" \
    --probe-dir "$R/$model/probes" --output-dir "$out" --variant counterfactual \
    --steps 12000 --lr 1e-4 --history 8 --horizon 6 --stride 1 \
    --constraint-form position --epsilon-source factual_rollout \
    --gravity-low 1.0 --gravity-high 9.0 --seed 42 --eval-every 400 --device cuda
done
wait
for model in single_s42 narrow_s42; do
  out=$R/$model/post_b3_counterfactual
  [ -f "$out/checkpoint.pt" ] || continue
  for h in 6 8; do
    target=$out/evaluation.json; [ "$h" = 8 ] && target=$out/evaluation_h8.json
    [ -f "$target" ] && continue
    launch "eval_${model}_b3_h$h" "$PY" "$G/scripts/evaluate.py" \
      --checkpoint "$out/checkpoint.pt" --cache "$R/$model/latents.pt" \
      --probe-dir "$R/$model/probes" --output "$target" \
      --posttrain-report "$out/posttrain_report.json" \
      --history 8 --horizon "$h" --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
      --label "${model}__b3_counterfactual" --device cuda
  done
done
wait
: > "$(dirname "$0")/.done-budget"
say "=== done ==="
