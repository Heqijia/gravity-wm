#!/usr/bin/env bash
# Does the 21-epoch arm give a better foundation?  The probe was still
# improving monotonically at epoch 7 (0.413 -> 0.381 -> 0.260 px) rather than
# plateauing, and the contact-free baselines reach 0.061 px, so the 7-epoch
# model that produced the headline is probably not the best available.  If the
# probe improves materially here, the whole result deserves a rerun on it.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
D=/fs/scratch/PAS3151/qijia387/gwm/data/bounce_single_r08
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
one() { local gpu=$1; shift; srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 "$@"; }

gate() { # gate <epoch> <gpu>
  local ep=$1 gpu=$2
  local src=$R/bounce_r08_sig050long_s42/checkpoint_epoch_$ep.pt
  [ -f "$src" ] || { say "  epoch_$ep 缺失"; return; }
  local d=$R/bounce_long_ep$ep
  [ -f "$d/probes/probe_eval.pt" ] && { say "  epoch_$ep 已有"; return; }
  mkdir -p "$d"; cp "$src" "$d/checkpoint.pt"
  for f in normalization_stats.json train_val_split.json readout.json; do
    cp "$R/bounce_r08_sig050long_s42/$f" "$d/$f" 2>/dev/null
  done
  one "$gpu" "$PY" "$G/scripts/build_cache.py" --checkpoint "$d/checkpoint.pt" --data "$D" \
    --output "$d/latents.pt" --device cuda > $ROOT/work/logs/cache_long_ep$ep.log 2>&1
  one "$gpu" "$PY" "$G/scripts/train_probes.py" --cache "$d/latents.pt" --output-dir "$d/probes" \
    --device cuda --history 8 --horizon 6 --stride 1 --max-episodes 6000 \
    > $ROOT/work/logs/probes_long_ep$ep.log 2>&1
  say "  epoch_$ep 完成"
  [ "$ep" = 021 ] || rm -f "$d/latents.pt"
}
say "=== 21 epoch 臂的探针闸门 ==="
( gate 014 0; gate 021 0 ) &
( gate 010 1; gate 018 1 ) &
wait
: > $ROOT/work/.done-long_gates
say "=== long_gates done ==="
