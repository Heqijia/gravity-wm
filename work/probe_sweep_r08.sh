#!/usr/bin/env bash
# Probe both r08 arms as a function of training depth.
#
# The sigreg term behaved as intended -- 4.38 -> 4.28 at weight 0.5, against
# 7.21 -> 9.40 at the default -- but sigreg is a proxy.  The quantity the
# study needs is whether the latents still carry position, and on the previous
# run that peaked at epoch 4 and then fell apart by epoch 7 while every loss
# kept improving.  So read the probe at several depths rather than only at the
# end, for both arms.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
D=/fs/scratch/PAS3151/qijia387/gwm/data/bounce_single_r08
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

sweep() { # sweep <arm> <gpu>
  local arm=$1 gpu=$2
  for ep in 002 004 007; do
    local src=$R/$arm/checkpoint_epoch_$ep.pt
    [ -f "$src" ] || { say "  $arm epoch_$ep 缺失"; continue; }
    local d=$R/${arm}_ep$ep
    [ -f "$d/probes/probe_eval.pt" ] && continue
    mkdir -p "$d"; cp "$src" "$d/checkpoint.pt"
    for f in normalization_stats.json train_val_split.json readout.json; do
      cp "$R/$arm/$f" "$d/$f" 2>/dev/null
    done
    srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
      env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
      "$PY" "$G/scripts/build_cache.py" --checkpoint "$d/checkpoint.pt" --data "$D" \
      --output "$d/latents.pt" --device cuda > $ROOT/work/logs/cache_${arm}_ep$ep.log 2>&1
    srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
      env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
      "$PY" "$G/scripts/train_probes.py" --cache "$d/latents.pt" --output-dir "$d/probes" \
      --device cuda --history 8 --horizon 6 --stride 1 --max-episodes 6000 \
      > $ROOT/work/logs/probes_${arm}_ep$ep.log 2>&1
    say "  $arm epoch_$ep 完成"
    rm -f "$d/latents.pt"   # 567M each; the probe report is what is wanted
  done
}
say "=== r08 探针扫描(两臂 × 3 个深度)==="
sweep bounce_r08_sig009_s42 0 &
sweep bounce_r08_sig050_s42 1 &
wait
: > $ROOT/work/.done-probe_sweep
say "=== probe_sweep done ==="
