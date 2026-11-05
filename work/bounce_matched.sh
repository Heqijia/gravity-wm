#!/usr/bin/env bash
# The baselines at the budget the headline counterfactual used.
#
# The comparison as it stands gives the method 36000 steps and every baseline
# 12000, because 36000 is what C_0/eps = 36.2 demanded to bring the constraint
# inside its tolerance.  That reasoning explains the method's budget and says
# nothing about the baselines', so the headline's 8.8x enjoys three times the
# compute.  At the matched 12000 the method still gives 8.191 px against
# 26.171, but the right table reports both arms at both budgets.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
M=$R/bounce_r08_s42
FLOOR=1.60
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
one() { local gpu=$1; shift; srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 "$@"; }

run() { # run <tag> <variant> <steps> <seed> <gpu>
  local tag=$1 v=$2 steps=$3 sd=$4 gpu=$5
  local out=$M/post_$tag
  [ -f "$out/posttrain_report.json" ] || \
  one "$gpu" "$PY" "$G/scripts/posttrain.py" --checkpoint "$M/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output-dir "$out" --variant "$v" --steps "$steps" --lr 1e-4 \
    --history 8 --horizon 6 --stride 1 --constraint-form position \
    --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
    --floor-resting-z $FLOOR --seed "$sd" --eval-every 1200 --device cuda \
    > $ROOT/work/logs/post_bounce_$tag.log 2>&1
  say "  post $tag exit=$?"
  [ -f "$out/checkpoint.pt" ] || return
  [ -f "$out/evaluation.json" ] || \
  one "$gpu" "$PY" "$G/scripts/evaluate.py" --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output "$out/evaluation.json" \
    --posttrain-report "$out/posttrain_report.json" \
    --history 8 --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
    --floor-resting-z $FLOOR --label "bounce_r08__$tag" --device cuda \
    > $ROOT/work/logs/eval_bounce_$tag.log 2>&1
  say "  eval $tag exit=$?"
}
say "=== 基线补到 36000 步(与主结果预算匹配)==="
( run b36_jepa     jepa    36000 42 0
  run b36_penalty  penalty 36000 42 0 ) &
( run b36_factual  factual 36000 42 1
  run b36_cf12k_s43 counterfactual 12000 43 1 ) &
wait
: > $ROOT/work/.done-bounce_matched
say "=== bounce_matched done ==="
