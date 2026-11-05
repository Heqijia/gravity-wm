#!/usr/bin/env bash
#SBATCH -A pas3151
#SBATCH -p quad
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH -t 08:00:00
#SBATCH -J gwm-resume
#SBATCH -o /users/PAS3151/qijia387/Phys-WM/work/logs/sb_resume_%j.out
# The interactive allocations lost their step credentials when the controller
# rebooted at 09:02, and the nodes are draining for maintenance, so the work
# cannot be driven from this session.  Queuing it as a batch job means it
# starts on its own whenever the cluster comes back, with no session attached.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
S=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
export MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

sf() { # single_far seed 43, the three arms the outage took
  local M=$S/single_far_sig009_s42
  for v in counterfactual factual penalty; do
    local out=$M/post_s43_$v
    [ -f "$out/posttrain_report.json" ] || \
    CUDA_VISIBLE_DEVICES=0 "$PY" "$G/scripts/posttrain.py" --checkpoint "$M/checkpoint.pt" \
      --cache "$M/latents.pt" --probe-dir "$M/probes" --output-dir "$out" --variant "$v" \
      --steps 12000 --lr 1e-4 --history 8 --horizon 6 --stride 1 --constraint-form position \
      --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 --seed 43 \
      --eval-every 600 --device cuda > $ROOT/work/logs/sb_post_sf43_$v.log 2>&1
    say "  sf43 post $v exit=$?"
    [ -f "$out/checkpoint.pt" ] || continue
    for H in 6 12 18; do
      [ -f "$out/eval_h$H.json" ] && continue
      CUDA_VISIBLE_DEVICES=0 "$PY" "$G/scripts/evaluate.py" --checkpoint "$out/checkpoint.pt" \
        --cache "$M/latents.pt" --probe-dir "$M/probes" --output "$out/eval_h$H.json" \
        --posttrain-report "$out/posttrain_report.json" --history 8 --horizon $H --stride 1 \
        --sweep-min 1.0 --sweep-max 9.0 --label "single_far__s43_${v}__h$H" --device cuda \
        > $ROOT/work/logs/sb_eval_sf43_${v}_h$H.log 2>&1
    done
    say "  sf43 eval $v 完成"
  done
}

bl() { # bounce on the 21-epoch foundation, the three arms the outage took
  local M=$S/bounce_long_ep021
  for v in counterfactual factual penalty; do
    local out=$M/post_mp_$v
    [ -f "$out/posttrain_report.json" ] || \
    CUDA_VISIBLE_DEVICES=1 "$PY" "$G/scripts/posttrain.py" --checkpoint "$M/checkpoint.pt" \
      --cache "$M/latents.pt" --probe-dir "$M/probes" --output-dir "$out" --variant "$v" \
      --steps 36000 --lr 1e-4 --history 8 --horizon 6 --stride 1 --constraint-form position \
      --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
      --floor-resting-z 1.60 --seed 42 --eval-every 1200 --device cuda \
      > $ROOT/work/logs/sb_post_bl21_$v.log 2>&1
    say "  bl21 post $v exit=$?"
    [ -f "$out/checkpoint.pt" ] || continue
    [ -f "$out/evaluation.json" ] || \
    CUDA_VISIBLE_DEVICES=1 "$PY" "$G/scripts/evaluate.py" --checkpoint "$out/checkpoint.pt" \
      --cache "$M/latents.pt" --probe-dir "$M/probes" --output "$out/evaluation.json" \
      --posttrain-report "$out/posttrain_report.json" --history 8 --horizon 6 --stride 1 \
      --sweep-min 1.0 --sweep-max 9.0 --floor-resting-z 1.60 --label "bounce_long21__$v" \
      --device cuda > $ROOT/work/logs/sb_eval_bl21_$v.log 2>&1
    say "  bl21 eval $v exit=$?"
  done
}
say "=== sbatch 恢复批次 on $(hostname) ==="
sf & bl & wait
: > $ROOT/work/.done-sb_resume
say "=== sb_resume done ==="
