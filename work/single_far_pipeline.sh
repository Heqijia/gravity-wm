#!/usr/bin/env bash
# Pretrain on single_far, gate the probe, then answer the horizon question.
#
# The gate matters more than usual here.  The scale analysis predicts SNR ~5.4
# from 0.328 px of signal against ball_single's 0.061 px probe, but that
# assumes the probe's pixel accuracy survives the rescaling, and the bouncing
# scene already showed a prediction like that being wrong by a factor of 3.
# If the probe fails the gate, the horizon curve measured on this model would
# be noise and the honest result is that the question stays unanswerable.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
S=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
D=/fs/scratch/PAS3151/qijia387/gwm/data/single_far
M=$S/single_far_s42
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
one() { local gpu=$1; shift; srun --jobid=8093487 --overlap -n1 --cpus-per-task=10 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 "$@"; }

while [ ! -f $ROOT/work/.done-gen_single_far ]; do sleep 60; done
[ -f "$D/generation_manifest.json" ] || { say "数据集未生成,终止"; exit 1; }

# 两臂,和 bounce 一样:默认 sigreg 与提权版。32 帧会不会塌缩是未知的。
say "=== single_far 预训练(两臂,梯度步匹配 7 epoch)==="
( [ -f "$S/single_far_sig009_s42/checkpoint.pt" ] || one 0 "$PY" "$G/scripts/pretrain.py" \
    --config "$G/configs/bounce_r08_sig009.yaml" --data "$D" \
    --output-dir "$S/single_far_sig009_s42" --readout spatial_map --gravity-scale 2.309 \
    > $ROOT/work/logs/pretrain_single_far_sig009.log 2>&1; say "  sig009 exit=$?" ) &
( [ -f "$S/single_far_sig050_s42/checkpoint.pt" ] || one 1 "$PY" "$G/scripts/pretrain.py" \
    --config "$G/configs/bounce_r08_sig050.yaml" --data "$D" \
    --output-dir "$S/single_far_sig050_s42" --readout spatial_map --gravity-scale 2.309 \
    > $ROOT/work/logs/pretrain_single_far_sig050.log 2>&1; say "  sig050 exit=$?" ) &
wait
: > $ROOT/work/.done-pretrain_single_far
say "=== single_far 预训练完成 ==="

gate() { # gate <arm> <gpu>
  local arm=$1 gpu=$2
  local d=$S/${arm}_s42
  [ -f "$d/checkpoint.pt" ] || return
  [ -f "$d/latents.pt" ] || one "$gpu" "$PY" "$G/scripts/build_cache.py" \
    --checkpoint "$d/checkpoint.pt" --data "$D" --output "$d/latents.pt" --device cuda \
    > $ROOT/work/logs/cache_${arm}.log 2>&1
  [ -f "$d/probes/probe_eval.pt" ] || one "$gpu" "$PY" "$G/scripts/train_probes.py" \
    --cache "$d/latents.pt" --output-dir "$d/probes" --device cuda \
    --history 8 --horizon 6 --stride 1 --max-episodes 6000 \
    > $ROOT/work/logs/probes_${arm}.log 2>&1
  say "  闸门 $arm exit=$?"
}
say "=== 探针闸门 ==="
gate single_far_sig009 0 & gate single_far_sig050 1 & wait
: > $ROOT/work/.done-single_far_gate
say "=== single_far_gate done ==="
