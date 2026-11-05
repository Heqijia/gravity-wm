#!/usr/bin/env bash
# Is the probe on the bouncing scene bad because the model is undertrained, or
# because of the scene itself?
#
# bounce_s42 probes at 0.870 px with SNR 1.16, against 0.061 px / SNR 16.5 on
# ball_single.  Two candidate causes, and they call for different fixes:
#   depth  -- 7 epochs (the gradient-step match) against the baselines' 40
#   scene  -- the ball is half_size 0.45 here and 0.80 everywhere else, and the
#             z axis probes 5x worse than x, which is where the bounce is
# Probing the epoch-1 checkpoint separates them: if the probe barely moves
# between epoch 1 and epoch 7, training depth is not the binding constraint.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
S=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
D=/fs/scratch/PAS3151/qijia387/gwm/data/bounce_single
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
one() { srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 "$@"; }

say "=== 诊断:探针精度随训练深度如何变化 ==="
for ep in 001 004; do
  d=$S/bounce_ep$ep
  [ -f "$S/bounce_s42/checkpoint_epoch_$ep.pt" ] || { say "  epoch_$ep 不存在"; continue; }
  mkdir -p "$d"
  cp "$S/bounce_s42/checkpoint_epoch_$ep.pt" "$d/checkpoint.pt"
  for f in normalization_stats.json train_val_split.json readout.json; do
    cp "$S/bounce_s42/$f" "$d/$f" 2>/dev/null
  done
  [ -f "$d/latents.pt" ] || one "$PY" "$G/scripts/build_cache.py" --checkpoint "$d/checkpoint.pt" \
    --data "$D" --output "$d/latents.pt" --device cuda > $ROOT/work/logs/cache_bounce_ep$ep.log 2>&1
  [ -f "$d/probes/probe_eval.pt" ] || one "$PY" "$G/scripts/train_probes.py" --cache "$d/latents.pt" \
    --output-dir "$d/probes" --device cuda --history 8 --horizon 6 --stride 1 \
    --max-episodes 6000 > $ROOT/work/logs/probes_bounce_ep$ep.log 2>&1
  say "  epoch_$ep 完成"
done
: > $ROOT/work/.done-bounce_diag
say "=== bounce_diag done ==="
