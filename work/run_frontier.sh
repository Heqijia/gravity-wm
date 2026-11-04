#!/usr/bin/env bash
# The tolerance sweep: the switching method's accuracy is set by eps, so sweep
# it and compare the (accuracy, JEPA loss) frontier it traces -- with a
# feasibility certificate at every point -- against a fixed penalty weight,
# which gives one point whose feasibility cannot be known in advance.
set -u
PY=/users/PAS3151/qijia387/Phys-WM/.venv-wm/bin/python
G=/users/PAS3151/qijia387/Phys-WM/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs
BASE=$R/spatial/pretrain_s42
W=/users/PAS3151/qijia387/Phys-WM/work
STEPS=${STEPS:-4000}

run() {  # run <name> <extra args...>
  local name=$1; shift
  local out=$R/spatial/post_$name
  [ -f "$out/posttrain_report.json" ] && { echo "skip $name"; return; }
  rm -rf "$out"
  CUDA_VISIBLE_DEVICES=$GPU srun --jobid=8093487 --overlap -n1 --cpus-per-task=4 --gres=gpu:2 \
    env CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 \
    "$PY" "$G/scripts/posttrain.py" \
      --checkpoint "$BASE/checkpoint.pt" --cache "$BASE/latents.pt" \
      --probe-dir "$BASE/probes" --output-dir "$out" \
      --steps "$STEPS" --lr 1e-4 --gate 0.9 --stride 1 --history 8 --horizon 6 \
      --epsilon-source factual_rollout --gravity-low 1.0 --gravity-high 9.0 \
      --seed 42 --eval-every 200 --device cuda "$@" \
    > "$W/post_$name.log" 2>&1
}

jobs_list=(
  "s1_counterfactual|--variant counterfactual"
  "s1_penalty|--variant penalty --penalty-lambda 1.0"
  "s1_cf_eps0.3|--variant counterfactual --epsilon-scale 0.3"
  "s1_cf_eps0.1|--variant counterfactual --epsilon-scale 0.1"
  "s1_cf_eps0.03|--variant counterfactual --epsilon-scale 0.03"
  "s1_pen_lam0.3|--variant penalty --penalty-lambda 0.3"
  "s1_pen_lam3|--variant penalty --penalty-lambda 3.0"
  "s1_pen_lam10|--variant penalty --penalty-lambda 10.0"
)
i=0
for entry in "${jobs_list[@]}"; do
  name=${entry%%|*}; args=${entry#*|}
  GPU=$(( i % 2 )); i=$(( i + 1 ))
  ( GPU=$GPU; run "$name" $args ) &
  if [ "$(jobs -rp | wc -l)" -ge 2 ]; then wait -n; fi
done
wait
echo "=== frontier runs done $(date +%H:%M) ==="
for entry in "${jobs_list[@]}"; do
  name=${entry%%|*}
  printf '%-18s ' "$name"
  tr '\r' '\n' < "$W/post_$name.log" | grep -oE "step +3[0-9]{3} mode=[a-z]+ L=[0-9.e+-]+ C=[0-9.e+-]+" | tail -1
  echo
done
