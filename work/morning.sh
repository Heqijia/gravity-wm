#!/usr/bin/env bash
# Two things worth the idle cards.
#
# GPU 0  single_far seed 43.  The horizon result that just came in is n = 1,
#        and it is the only measurement in the study past h = 8.
# GPU 1  bounce on the 21-epoch foundation.  Section 6.17's result was measured
#        at probe SNR 3.88, and the 21-epoch arm reaches 7.73 with eps falling
#        from 0.607 to 0.243 -- so those numbers are a floor, and this says by
#        how much.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
S=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
one() { local gpu=$1; shift; srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 "$@"; }

sf() { # single_far seed 43
  local M=$S/single_far_sig009_s42
  for v in jepa factual penalty counterfactual; do
    local out=$M/post_s43_$v
    [ -f "$out/posttrain_report.json" ] || \
    one 0 "$PY" "$G/scripts/posttrain.py" --checkpoint "$M/checkpoint.pt" --cache "$M/latents.pt" \
      --probe-dir "$M/probes" --output-dir "$out" --variant "$v" --steps 12000 --lr 1e-4 \
      --history 8 --horizon 6 --stride 1 --constraint-form position \
      --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
      --seed 43 --eval-every 600 --device cuda > $ROOT/work/logs/post_sf43_$v.log 2>&1
    say "  sf43 post $v exit=$?"
    [ -f "$out/checkpoint.pt" ] || continue
    for H in 6 12 18; do
      [ -f "$out/eval_h$H.json" ] && continue
      one 0 "$PY" "$G/scripts/evaluate.py" --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
        --probe-dir "$M/probes" --output "$out/eval_h$H.json" \
        --posttrain-report "$out/posttrain_report.json" --history 8 --horizon $H --stride 1 \
        --sweep-min 1.0 --sweep-max 9.0 --label "single_far__s43_${v}__h$H" --device cuda \
        > $ROOT/work/logs/eval_sf43_${v}_h$H.log 2>&1
    done
    say "  sf43 eval $v 完成"
  done
}

bl() { # bounce on the 21-epoch foundation
  local M=$S/bounce_long_ep021
  [ -f "$M/latents.pt" ] || { say "  bounce_long_ep021 缺 cache"; return; }
  # C_0/eps first: the budget rule needs it before committing
  [ -f "$M/post_probe/posttrain_report.json" ] || \
  one 1 "$PY" "$G/scripts/posttrain.py" --checkpoint "$M/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output-dir "$M/post_probe" --variant counterfactual --steps 2 \
    --lr 1e-4 --history 8 --horizon 6 --stride 1 --constraint-form position \
    --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
    --floor-resting-z 1.60 --seed 42 --device cuda > $ROOT/work/logs/probe2_bl21.log 2>&1
  say "  bl21 C_0/eps 探针 exit=$?"
  for v in pretrained jepa factual penalty counterfactual; do
    local out=$M/post_mp_$v
    local steps=36000; [ "$v" = pretrained ] && steps=0
    [ -f "$out/posttrain_report.json" ] || \
    one 1 "$PY" "$G/scripts/posttrain.py" --checkpoint "$M/checkpoint.pt" --cache "$M/latents.pt" \
      --probe-dir "$M/probes" --output-dir "$out" --variant "$v" --steps $steps --lr 1e-4 \
      --history 8 --horizon 6 --stride 1 --constraint-form position \
      --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
      --floor-resting-z 1.60 --seed 42 --eval-every 1200 --device cuda \
      > $ROOT/work/logs/post_bl21_$v.log 2>&1
    say "  bl21 post $v exit=$?"
    [ -f "$out/checkpoint.pt" ] || continue
    [ -f "$out/evaluation.json" ] || \
    one 1 "$PY" "$G/scripts/evaluate.py" --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
      --probe-dir "$M/probes" --output "$out/evaluation.json" \
      --posttrain-report "$out/posttrain_report.json" --history 8 --horizon 6 --stride 1 \
      --sweep-min 1.0 --sweep-max 9.0 --floor-resting-z 1.60 \
      --label "bounce_long21__$v" --device cuda > $ROOT/work/logs/eval_bl21_$v.log 2>&1
    say "  bl21 eval $v exit=$?"
  done
}
say "=== 早晨批次:single_far seed43 (gpu0) + bounce 21ep 地基 (gpu1) ==="
sf & bl & wait
: > $ROOT/work/.done-morning
say "=== morning done ==="
