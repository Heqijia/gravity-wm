#!/usr/bin/env bash
# A second seed for the three long counterfactual runs that carry §6.18's table.
#
# `single_s42` 36000, `narrow_s42` 36000 and `sq_single_s42` 24000 are each
# n = 1, and they are the rows the four-model comparison rests on.  The flags
# are read back from each run's own posttrain_report.json command block, so
# the only thing that differs is the seed.
#
# Runs over ssh on a0004's three free cards: srun into the allocation still
# fails with a protocol authentication error after the 09:02 controller
# reboot, but the node itself is healthy.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
S=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
NODE=a0004
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

rsh() { # rsh <gpu> <logfile> <command string>
  local gpu=$1 log=$2; shift 2
  ssh -q -o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=60 "$NODE" \
    "CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 $*" > "$log" 2>&1
}

run() { # run <model> <steps> <gpu>
  local m=$1 steps=$2 gpu=$3
  local M=$S/$m
  local out=$M/post_s43b_counterfactual
  if [ ! -f "$out/posttrain_report.json" ]; then
    rsh "$gpu" "$ROOT/work/logs/ssh_post_${m}_s43b.log" \
      "$PY $G/scripts/posttrain.py --checkpoint $M/checkpoint.pt --cache $M/latents.pt \
       --probe-dir $M/probes --output-dir $out --variant counterfactual --steps $steps \
       --lr 1e-4 --history 8 --horizon 6 --stride 1 --constraint-form position \
       --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
       --batch-size 192 --constraint-batch-size 384 --seed 43 --eval-every 1200 --device cuda"
    say "  $m seed43 post exit=$?"
  fi
  [ -f "$out/checkpoint.pt" ] || return
  if [ ! -f "$out/evaluation.json" ]; then
    rsh "$gpu" "$ROOT/work/logs/ssh_eval_${m}_s43b.log" \
      "$PY $G/scripts/evaluate.py --checkpoint $out/checkpoint.pt --cache $M/latents.pt \
       --probe-dir $M/probes --output $out/evaluation.json \
       --posttrain-report $out/posttrain_report.json --history 8 --horizon 6 --stride 1 \
       --sweep-min 1.0 --sweep-max 9.0 --label ${m}__s43b_counterfactual --device cuda"
    say "  $m seed43 eval exit=$?"
  fi
}

say "=== 三个主结果补 seed 43(a0004 卡 1/2/3)==="
run single_s42    36000 1 &
run narrow_s42    36000 2 &
run sq_single_s42 24000 3 &
wait
: > $ROOT/work/.done-seed43_mains
say "=== seed43_mains done ==="
