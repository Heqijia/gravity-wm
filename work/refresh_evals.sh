#!/usr/bin/env bash
# Re-evaluate the runs whose evaluation predates the acceleration-diagnostic
# fix, so the `accel far` column means the same thing in every row.
#
# Those evaluations were produced at 10:25 with the constraint-form fix in
# place but before the diagnostic one, so `acceleration_rmse` holds a position
# residual (0.2-0.3) where the later rows hold the real acceleration residual
# (~5.0).  Every other column is unaffected.  Detected by the mixed scale in
# one table rather than by inspection, which is why the column is now read
# from a fixed form.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
LOGS=$ROOT/work/logs
JOBID=${JOBID:-8093487}
GPU_NEXT=0
# the commit that fixed the diagnostic; anything written before it is stale
FIX=$G/gwm/evaluate.py
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

stale=0
for d in "$R"/*/post_*; do
  [ -f "$d/evaluation.json" ] || continue
  [ -f "$d/checkpoint.pt" ] || continue
  [ "$d/evaluation.json" -nt "$FIX" ] && continue
  stale=$((stale + 1))
  model=$(basename "$(dirname "$d")"); tag=$(basename "$d" | sed s/post_//)
  hist=8; [ "$model" = v2_s42 ] && hist=6
  mv "$d/evaluation.json" "$d/evaluation.stale.json"
  launch "eval_${model}_${tag}_rf" "$PY" "$G/scripts/evaluate.py" \
    --checkpoint "$d/checkpoint.pt" --cache "$R/$model/latents.pt" \
    --probe-dir "$R/$model/probes" --output "$d/evaluation.json" \
    --posttrain-report "$d/posttrain_report.json" \
    --history "$hist" --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
    --label "${model}__${tag}" --device cuda
done
wait
say "=== refreshed $stale evaluations ==="
: > "$(dirname "$0")/.done-refresh_evals"
