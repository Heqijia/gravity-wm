#!/usr/bin/env bash
# single_far: the five variants, then the horizon curve at 6, 12 and 18.
#
# Budget 12000 for every arm.  C_0/eps = 3.8 here, the lowest of any model
# except spatial's 2.0, so by the rule of section 6.7 even 4000 would reach
# OPTIMISE; 12000 is taken for all five so the comparison is matched from the
# start rather than repaired afterwards, which is what section 6.18 had to do
# for four earlier models.
#
# The horizon curve comes from single evaluations at horizon 12 and 18 read
# through position_rmse_by_step, so h = 1..12 and h = 1..18 each come from one
# consistent window set.  Training rolls out 6, so 12 and 18 are extrapolation.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
M=/fs/scratch/PAS3151/qijia387/gwm/runs/suite/single_far_sig009_s42
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
one() { local gpu=$1; shift; srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 "$@"; }

post() { # post <variant> <gpu>
  local v=$1 gpu=$2
  local out=$M/post_mp_$v
  local steps=12000; [ "$v" = pretrained ] && steps=0
  [ -f "$out/posttrain_report.json" ] || \
  one "$gpu" "$PY" "$G/scripts/posttrain.py" --checkpoint "$M/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output-dir "$out" --variant "$v" --steps $steps --lr 1e-4 \
    --history 8 --horizon 6 --stride 1 --constraint-form position \
    --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
    --seed 42 --eval-every 600 --device cuda > $ROOT/work/logs/post_sf_$v.log 2>&1
  say "  post $v exit=$?"
  for H in 6 12 18; do
    local tag=h$H
    [ -f "$out/eval_$tag.json" ] && continue
    [ -f "$out/checkpoint.pt" ] || return
    one "$gpu" "$PY" "$G/scripts/evaluate.py" --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
      --probe-dir "$M/probes" --output "$out/eval_$tag.json" \
      --posttrain-report "$out/posttrain_report.json" \
      --history 8 --horizon $H --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
      --label "single_far__${v}__$tag" --device cuda > $ROOT/work/logs/eval_sf_${v}_$tag.log 2>&1
    say "  eval $v $tag exit=$?"
  done
}
say "=== single_far 五变体 @12000 + horizon 6/12/18 ==="
( post pretrained 0; post jepa 0; post counterfactual 0 ) &
( post factual 1;    post penalty 1 ) &
wait
: > $ROOT/work/.done-single_far_suite
say "=== single_far_suite done ==="
