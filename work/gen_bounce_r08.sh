#!/usr/bin/env bash
# Regenerate the bouncing dataset with the object half-extent the six
# contact-free datasets use.  At 0.45 the probe reached only 0.870 px with
# SNR 1.16, against 0.061 px / SNR 16.5 on ball_single, and the object being
# 44% of the diameter is an uncontrolled difference from every baseline that
# has nothing to do with the question.  Measured at 0.80 the scene is
# otherwise unchanged: restitution 0.897, 1.82 contacts per episode, 51% of
# windows usable by the constraint, 39% with a bounce in the horizon.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
cd $ROOT/gravity-wm
srun --jobid=8093487 --overlap -n1 --cpus-per-task=16 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
  ../.venv-wm/bin/python scripts/generate_bounce.py \
    --output-dir /fs/scratch/PAS3151/qijia387/gwm/data/bounce_single_r08 \
    --half-size 0.80 --train-episodes 16000 --heldout-train-gravity-episodes 800 \
    --test-episodes-per-gravity 32 --train-std 0.0 --workers 12
echo "GEN_EXIT=$?"
: > $ROOT/work/.done-gen_r08
