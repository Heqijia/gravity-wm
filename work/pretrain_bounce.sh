#!/usr/bin/env bash
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
cd $ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
D=/fs/scratch/PAS3151/qijia387/gwm/data/bounce_single
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
run() { # run <name> <config> <gpu>
  local name=$1 cfg=$2 gpu=$3
  [ -f "$R/$name/checkpoint.pt" ] && { say "  $name 已存在,跳过"; return; }
  say "  启动 $name on gpu $gpu"
  srun --jobid=8093487 --overlap -n1 --cpus-per-task=10 --gres=gpu:2 \
    env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
    $ROOT/.venv-wm/bin/python scripts/pretrain.py \
      --config configs/$cfg --data "$D" --output-dir "$R/$name" \
      --readout spatial_map --gravity-scale 2.309 \
    > $ROOT/work/logs/pretrain_$name.log 2>&1
  say "  $name exit=$?"
}
say "=== bounce 预训练:梯度步匹配版 + 长训版 ==="
run bounce_s42      bounce_sg_jepa_gru.yaml      0 &
run bounce_long_s42 bounce_sg_jepa_gru_long.yaml 1 &
wait
: > $ROOT/work/.done-pretrain_bounce
say "=== pretrain_bounce done ==="
