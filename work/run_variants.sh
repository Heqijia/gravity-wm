#!/usr/bin/env bash
# Post-train the five variants on one pretrained checkpoint, two at a time.
set -u
PY=/users/PAS3151/qijia387/Phys-WM/.venv-wm/bin/python
G=/users/PAS3151/qijia387/Phys-WM/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs
BASE=$R/spatial/pretrain_s42
TAG=${TAG:-s1}
STRIDE=${STRIDE:-1}
STEPS=${STEPS:-4000}
LR=${LR:-1e-4}
gpu=0
pids=""
for variant in pretrained jepa factual penalty counterfactual; do
  out=$R/spatial/post_${TAG}_${variant}
  rm -rf "$out"
  CUDA_VISIBLE_DEVICES=$gpu srun --jobid=8093487 --overlap -n1 --cpus-per-task=3 --gres=gpu:2 \
    env CUDA_VISIBLE_DEVICES=$gpu PYTHONDONTWRITEBYTECODE=1 \
    "$PY" "$G/scripts/posttrain.py" \
      --checkpoint "$BASE/checkpoint.pt" --cache "$BASE/latents.pt" \
      --probe-dir "$BASE/probes" --output-dir "$out" --variant "$variant" \
      --steps "$STEPS" --lr "$LR" --gate 0.9 --stride "$STRIDE" \
      --history 8 --horizon 6 --epsilon-source factual_rollout \
      --gravity-low 1.0 --gravity-high 9.0 --seed 42 --eval-every 200 --device cuda \
    > "/users/PAS3151/qijia387/Phys-WM/work/post_${TAG}_${variant}.log" 2>&1 &
  pids="$pids $!:$variant"
  gpu=$(( (gpu + 1) % 2 ))
  # two in flight at a time
  if [ "$(jobs -rp | wc -l)" -ge 2 ]; then wait -n; fi
done
wait
for entry in $pids; do
  v=${entry#*:}
  printf '%-16s ' "$v"
  tail -3 "/users/PAS3151/qijia387/Phys-WM/work/post_${TAG}_${v}.log" | tr '\n' ' '
  echo
done
