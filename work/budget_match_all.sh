#!/usr/bin/env bash
# Give every baseline the budget its counterfactual arm was given.
#
# Four models compare a long counterfactual run against 4000-step baselines:
# single and narrow at 36000 against 4000 (9x), square and ball_ax at 24000
# against 4000 (6x).  The reasoning that set the method's budget came from
# C_0/eps and says nothing about the baselines', so every "more budget" row in
# sections 6.8, 6.9 and 6.12 currently reads against an arm that was stopped
# early for no stated reason.  The headline five-variant tables are unaffected:
# those are 4000 against 4000.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
S=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

job() { # job <model> <variant> <steps> <gpu> [extra...]
  local model=$1 v=$2 steps=$3 gpu=$4; shift 4
  local dir=$S/$model
  local out=$dir/post_bm${steps}_$v
  [ -f "$dir/latents.pt" ] || { say "  $model 无 cache,跳过"; return; }
  if [ ! -f "$out/posttrain_report.json" ]; then
    srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
      env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
      "$PY" "$G/scripts/posttrain.py" --checkpoint "$dir/checkpoint.pt" --cache "$dir/latents.pt" \
      --probe-dir "$dir/probes" --output-dir "$out" --variant "$v" --steps "$steps" --lr 1e-4 \
      --history 8 --horizon 6 --stride 1 --constraint-form position \
      --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
      --seed 42 --eval-every 1200 --device cuda "$@" \
      > $ROOT/work/logs/post_${model}_bm${steps}_$v.log 2>&1
    say "  post $model/$v@$steps exit=$?"
  fi
  [ -f "$out/checkpoint.pt" ] || return
  [ -f "$out/evaluation.json" ] && return
  srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
    env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
    "$PY" "$G/scripts/evaluate.py" --checkpoint "$out/checkpoint.pt" --cache "$dir/latents.pt" \
    --probe-dir "$dir/probes" --output "$out/evaluation.json" \
    --posttrain-report "$out/posttrain_report.json" \
    --history 8 --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
    --label "${model}__bm${steps}_$v" --device cuda "$@" \
    > $ROOT/work/logs/eval_${model}_bm${steps}_$v.log 2>&1
  say "  eval $model/$v@$steps exit=$?"
}

# Wait on the marker file, not on pgrep.  ``pgrep -f bounce_matched.sh``
# matched this script's own launching shell, whose command line happened to
# contain that string, so the script waited for its own parent and never
# started while both GPUs sat idle.
say "=== 为四个模型的基线补齐预算 ==="

( for v in jepa factual penalty; do job sq_single_s42 $v 24000 0; done
  for v in jepa factual penalty; do job single_s42    $v 36000 0; done ) &
( for v in jepa factual penalty; do job ax_s42        $v 24000 1; done
  for v in jepa factual penalty; do job narrow_s42    $v 36000 1; done ) &
wait
: > $ROOT/work/.done-budget_match_all
say "=== budget_match_all done ==="
