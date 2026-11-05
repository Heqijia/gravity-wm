#!/usr/bin/env bash
# Two pretraining arms on the corrected bouncing dataset.
#
# bounce_s42 showed representation collapse rather than undertraining: the
# probe went 0.366 px (epoch 1) -> 0.164 (epoch 4) -> 0.870 (epoch 7) while the
# rollout loss kept improving, 0.0412 -> 0.0178.  The baselines' sigreg term
# *falls* over training (3.57 -> 2.91 on ball_single); this run's *rose*,
# 7.26 -> 8.43, and starts at twice the baseline value.  sigreg is the
# anti-collapse term, so raising its weight is the matched remedy rather than
# a blind sweep.  Both arms checkpoint every epoch so the probe can be read as
# a function of training depth instead of only at the end.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
cd $ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
D=/fs/scratch/PAS3151/qijia387/gwm/data/bounce_single_r08
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
run() {
  local name=$1 cfg=$2 gpu=$3
  [ -f "$R/$name/checkpoint.pt" ] && { say "  $name 已存在"; return; }
  say "  启动 $name on gpu $gpu"
  srun --jobid=8093487 --overlap -n1 --cpus-per-task=10 --gres=gpu:2 \
    env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
    $ROOT/.venv-wm/bin/python scripts/pretrain.py \
      --config configs/$cfg --data "$D" --output-dir "$R/$name" \
      --readout spatial_map --gravity-scale 2.309 \
    > $ROOT/work/logs/pretrain_$name.log 2>&1
  say "  $name exit=$?"
}
say "=== r08 预训练:sigreg 对照 ==="
run bounce_r08_sig009_s42 bounce_r08_sig009.yaml 0 &
run bounce_r08_sig050_s42 bounce_r08_sig050.yaml 1 &
wait
: > $ROOT/work/.done-pretrain_r08
say "=== pretrain_r08 done ==="
