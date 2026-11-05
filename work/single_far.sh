#!/usr/bin/env bash
# A contact-free scene at a larger spatial scale, so the horizon question can
# be asked past h = 8.
#
# The limit is geometric and scale-invariant in the wrong way: keeping the
# counterfactual in frame for H steps needs a view of about g'(H dt)^2/2, and
# the signal g dt^2 measured in pixels is image_size/view times that, so
# signal_px falls as 1/H^2.  Measured acceptance for g' = 9 at 128 px:
#
#   view 13 (current), 16 frames : h=6 86%,  h=12 impossible (needs 20 frames)
#   view 40, 32 frames           : h=6 99%,  h=12 87%,  h=18 18%
#   view 60, 32 frames           : h=6 100%, h=12 95%,  h=18 61%
#
# view 40 is the working point: 0.328 px of signal against ball_single's
# 0.061 px probe is SNR ~5.4, while view 60 drops it to 3.6, too close to the
# gate of 3.  h = 18 at 18% still leaves ~20000 windows, which is nothing like
# the bouncing scene's zero.
#
# The ball radius scales with the view so it keeps its size in pixels; that is
# what holds the probe's pixel accuracy roughly fixed.  The training horizon
# stays 6, so 12 and 18 are genuine extrapolation.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
D=/fs/scratch/PAS3151/qijia387/gwm/data/single_far
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

while [ ! -f $ROOT/work/.done-budget_match_all ]; do sleep 60; done
say "=== 生成 single_far(视野 40,32 帧,g 恒为 4)==="
[ -f "$D/generation_manifest.json" ] || \
srun --jobid=8093487 --overlap -n1 --cpus-per-task=16 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
  "$PY" "$G/scripts/gen_data.py" --output "$D" \
    --frames 32 --view-height 40.0 --half-size 2.46 --dt 0.16 \
    --train-episodes 16000 --heldout-train-gravity-episodes 800 \
    --test-episodes-per-gravity 32 --train-mean 4.0 --train-std 0.0 \
    --workers 12 > $ROOT/work/logs/gen_single_far.log 2>&1
say "  生成 exit=$?"
: > $ROOT/work/.done-gen_single_far
