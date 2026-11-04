#!/usr/bin/env bash
# Is the SSM's failure "too far outside the tolerance" or "C is not smooth"?
#
# The SSM starts at C/eps = 87.5 (the GRU starts at 2.0) and spends all 4000
# steps in RESTORE, never once reaching OPTIMISE, so no certificate is even
# possible.  Worse, the first RESTORE steps *increase* C, from 1.60 to over 15,
# despite a step length of 6.3e-5 -- the Polyak linearisation is invalid at
# that scale because ||grad C|| is 2.5e4 (the GRU's is 1.09) and swings by 50x
# between steps.
#
# Relaxing eps by 10x puts the start at C/eps = 8.75.  If the method then works,
# the failure was the distance to the tolerance and a larger step budget or a
# staged tolerance would fix it.  If it still fails, the constraint landscape of
# this predictor is not smooth enough for a first-order method and that is a
# limitation to state, not a parameter to tune.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
LOGS=$ROOT/work/logs
M=$R/ssm_s42
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
for scale in 10 40; do
  out=$M/post_rx${scale}_counterfactual
  [ -f "$out/posttrain_report.json" ] || launch "post_ssm_s42_rx${scale}" \
    "$PY" "$G/scripts/posttrain.py" \
    --checkpoint "$M/checkpoint.pt" --cache "$M/latents.pt" --probe-dir "$M/probes" \
    --output-dir "$out" --variant counterfactual --steps 4000 --lr 1e-8 \
    --history 8 --horizon 6 --stride 1 --constraint-form position \
    --epsilon-source factual_rollout --epsilon-scale "$scale" \
    --gravity-low 1.0 --gravity-high 9.0 --seed 42 --eval-every 200 --device cuda
done
wait
for scale in 10 40; do
  out=$M/post_rx${scale}_counterfactual
  [ -f "$out/checkpoint.pt" ] || continue
  [ -f "$out/evaluation.json" ] && continue
  launch "eval_ssm_s42_rx${scale}" "$PY" "$G/scripts/evaluate.py" \
    --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" --probe-dir "$M/probes" \
    --output "$out/evaluation.json" --posttrain-report "$out/posttrain_report.json" \
    --history 8 --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
    --label "ssm_s42__rx${scale}_counterfactual" --device cuda
done
wait
: > "$(dirname "$0")/.done-ssm_tolerance"
say "=== done ==="
