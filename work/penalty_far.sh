#!/usr/bin/env bash
# The far end of the penalty frontier, and a second seed for the tolerance one.
#
# The tolerance sweep reaches C/eps = 0.96 at eps x 0.6 with a counterfactual
# error of 3.23 px.  The penalty sweep stops at lambda = 100, which is the
# first weight that lands inside the tolerance at all (C/eps = 0.88).  Claiming
# the method reaches a part of the frontier the penalty cannot requires pushing
# lambda until it either gets there or stops improving, so this runs
# lambda in {300, 1000, 3000}.
#
# Kept out of orchestrate.sh deliberately: bash reads a script incrementally,
# so editing one that is running shifts the byte offsets under it.  Run this
# only when orchestrate.sh is idle.
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

run() { # run <model> <tag> <variant> <seed> [extra...]
  local model=$1 tag=$2 variant=$3 seed=$4; shift 4
  local base=$R/$model out=$R/$model/post_$tag
  [ -f "$out/posttrain_report.json" ] || launch "post_${model}_${tag}" \
    "$PY" "$G/scripts/posttrain.py" \
    --checkpoint "$base/checkpoint.pt" --cache "$base/latents.pt" \
    --probe-dir "$base/probes" --output-dir "$out" --variant "$variant" \
    --steps "${STEPS:-4000}" --lr 1e-4 --history 8 --horizon 6 --stride 1 \
    --constraint-form position --epsilon-source factual_rollout \
    --gravity-low 1.0 --gravity-high 9.0 --seed "$seed" --eval-every 200 \
    --device cuda "$@"
}

say "=== penalty far end + second seed of the tolerance sweep ==="
for lam in 300.0 1000.0 3000.0; do
  run spatial_s42 "pq_lam$lam" penalty 42 --penalty-lambda "$lam"
done
for scale in 0.8 0.6; do
  run spatial_s43 "pq_eps$scale" counterfactual 43 --epsilon-scale "$scale"
done
for lam in 10.0 100.0 1000.0; do
  run spatial_s43 "pq_lam$lam" penalty 43 --penalty-lambda "$lam"
done
wait
say "post-training done; evaluating"
for d in "$R"/spatial_s4*/post_pq_*; do
  [ -f "$d/checkpoint.pt" ] || continue
  [ -f "$d/evaluation.json" ] && continue
  model=$(basename "$(dirname "$d")"); tag=$(basename "$d" | sed s/post_//)
  launch "eval_${model}_${tag}" "$PY" "$G/scripts/evaluate.py" \
    --checkpoint "$d/checkpoint.pt" --cache "$R/$model/latents.pt" \
    --probe-dir "$R/$model/probes" --output "$d/evaluation.json" \
    --posttrain-report "$d/posttrain_report.json" \
    --history 8 --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
    --label "${model}__${tag}" --device cuda
done
wait
say "=== done ==="
