#!/usr/bin/env bash
set -u
cd /users/PAS3151/qijia387/Phys-WM/gravity-wm
srun --jobid=8093487 --overlap -n1 --cpus-per-task=24 --gres=gpu:2 \
  env MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
  ../.venv-wm/bin/python scripts/generate_bounce.py \
    --output-dir /fs/scratch/PAS3151/qijia387/gwm/data/bounce_single \
    --train-episodes 16000 --heldout-train-gravity-episodes 800 \
    --test-episodes-per-gravity 32 --train-std 0.0 --workers 16
echo "GEN_EXIT=$?"
