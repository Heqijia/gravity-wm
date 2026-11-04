#!/usr/bin/env bash
# Periodically score every new checkpoint of the active runs.
set -u
PY=/users/PAS3151/qijia387/Phys-WM/.venv-wm/bin/python
G=/users/PAS3151/qijia387/Phys-WM/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs
LOG=/users/PAS3151/qijia387/Phys-WM/work/supervise.tsv
INTERVAL=${INTERVAL:-1200}
ROUNDS=${ROUNDS:-60}
for i in $(seq 1 "$ROUNDS"); do
  echo "=== pass $i  $(date '+%H:%M:%S') ==="
  srun --jobid=8093487 --overlap -n1 --cpus-per-task=4 --gres=gpu:2 \
    env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
    "$PY" "$G/scripts/supervise.py" \
      --runs "$R/ball_spatial_s42" "$R/ball_spatial_s43" "$R/ball_s42" \
      --data /fs/scratch/PAS3151/qijia387/gwm/data/ball_v1 \
      --log "$LOG" --fit-episodes 800 --test-episodes 250 --probe-epochs 150 --include-best \
    2>&1 | grep -viE "GpuFreq|^srun"
  sleep "$INTERVAL"
done
