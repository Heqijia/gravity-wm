#!/usr/bin/env bash
# Evaluate every finished post-training run, one at a time so the frontier
# sweep keeps the GPUs.
set -u
PY=/users/PAS3151/qijia387/Phys-WM/.venv-wm/bin/python
G=/users/PAS3151/qijia387/Phys-WM/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs
BASE=$R/spatial/pretrain_s42
for d in "$R"/spatial/post_s1_*; do
  [ -f "$d/checkpoint.pt" ] || continue
  [ -f "$d/evaluation.json" ] && continue
  name=$(basename "$d")
  variant=$($PY -c "import json;print(json.load(open('$d/posttrain_report.json'))['variant'])" 2>/dev/null || echo unknown)
  echo ">> $name ($variant)"
  srun --jobid=8093487 --overlap -n1 --cpus-per-task=3 --gres=gpu:2 \
    env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
    "$PY" "$G/scripts/evaluate.py" \
      --checkpoint "$d/checkpoint.pt" --cache "$BASE/latents.pt" \
      --probe-dir "$BASE/probes" --output "$d/evaluation.json" \
      --posttrain-report "$d/posttrain_report.json" \
      --history 8 --horizon 6 --stride 1 \
      --sweep-min 1.0 --sweep-max 9.0 --label "$name" --device cuda \
    > "/users/PAS3151/qijia387/Phys-WM/work/eval_$name.log" 2>&1
  $PY - "$d/evaluation.json" "$variant" <<'PYEOF'
import json, sys
p = sys.argv[1]
d = json.loads(open(p).read()); d["variant"] = sys.argv[2]
open(p, "w").write(json.dumps(d, indent=2, sort_keys=True))
PYEOF
  tail -12 "/users/PAS3151/qijia387/Phys-WM/work/eval_$name.log" | grep -viE "GpuFreq"
done
echo "=== evaluations done $(date +%H:%M) ==="
