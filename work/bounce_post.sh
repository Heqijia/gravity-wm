#!/usr/bin/env bash
# Option A on the corrected bouncing dataset, plus a longer pretraining arm.
#
# The chosen model is bounce_r08_sig050 at epoch 7: probe 0.260 px, SNR 3.88,
# eps 0.607.  Epoch 7 and not checkpoint.pt, because the probes were trained on
# checkpoint_epoch_007.pt and the two files differ -- pairing a constraint with
# a probe trained on different weights would silently measure the wrong thing.
#
# GPU 0 runs the experiment.  GPU 1 extends pretraining to 21 epochs, because
# the probe was still improving monotonically at epoch 7 (0.413 -> 0.381 ->
# 0.260) rather than plateauing, so the foundation is not yet as good as it
# can be; the baselines reach 0.061 px.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
D=/fs/scratch/PAS3151/qijia387/gwm/data/bounce_single_r08
M=$R/bounce_r08_s42
FLOOR=1.60
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
one() { local gpu=$1; shift; srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 "$@"; }

# ---- GPU 1: 更长的预训练 ----
( say "  启动 21 epoch 预训练 on gpu 1"
  one 1 "$PY" "$G/scripts/pretrain.py" --config "$G/configs/bounce_r08_sig050_long.yaml" \
    --data "$D" --output-dir "$R/bounce_r08_sig050long_s42" \
    --readout spatial_map --gravity-scale 2.309 \
    > $ROOT/work/logs/pretrain_bounce_r08_sig050long.log 2>&1
  say "  21 epoch 预训练 exit=$?" ) &

# ---- GPU 0: 闸门 + 后训练 ----
mkdir -p "$M"
cp "$R/bounce_r08_sig050_s42/checkpoint_epoch_007.pt" "$M/checkpoint.pt"
for f in normalization_stats.json train_val_split.json readout.json; do
  cp "$R/bounce_r08_sig050_s42/$f" "$M/$f" 2>/dev/null
done
cp -r "$R/bounce_r08_sig050_s42_ep007/probes" "$M/probes" 2>/dev/null
[ -f "$M/latents.pt" ] || { say "  编码 cache"; one 0 "$PY" "$G/scripts/build_cache.py" \
  --checkpoint "$M/checkpoint.pt" --data "$D" --output "$M/latents.pt" --device cuda \
  > $ROOT/work/logs/cache_bounce_r08.log 2>&1; say "  cache exit=$?"; }

say "  测 C_0/eps(2 步探针,决定预算)"
one 0 "$PY" "$G/scripts/posttrain.py" --checkpoint "$M/checkpoint.pt" --cache "$M/latents.pt" \
  --probe-dir "$M/probes" --output-dir "$M/post_probe" --variant counterfactual \
  --steps 2 --lr 1e-4 --history 8 --horizon 6 --stride 1 --constraint-form position \
  --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
  --floor-resting-z $FLOOR --seed 42 --device cuda \
  > $ROOT/work/logs/probe2_bounce_r08.log 2>&1
say "  C_0/eps 探针 exit=$?"
: > $ROOT/work/.done-bounce_c0
say "=== bounce_c0 done ==="
wait
: > $ROOT/work/.done-bounce_post
