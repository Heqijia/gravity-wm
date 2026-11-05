#!/usr/bin/env bash
# Redo what the two bugs invalidated, then evaluate everything.
#
#   factual        the floor filter was applied only in the counterfactual
#                  branch, so the sharpest control measured bounce residuals:
#                  C_0/eps = 621 against the counterfactual arm's 36.2
#   counterfactual 12000 steps took C/eps only from 36.2 to 6.48 with zero
#                  OPTIMISE steps, where single_s42 reached 0.73 from 55.0 in
#                  the same budget.  36000 is the next rung (RESULTS.md 6.7).
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
M=$R/bounce_r08_s42
FLOOR=1.60
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
one() { srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 "$@"; }

post() { # post <tag> <variant> <steps>
  local tag=$1 v=$2 steps=$3
  local out=$M/post_$tag
  [ -f "$out/posttrain_report.json" ] && { say "  $tag 已有"; return; }
  one "$PY" "$G/scripts/posttrain.py" --checkpoint "$M/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output-dir "$out" --variant "$v" --steps "$steps" --lr 1e-4 \
    --history 8 --horizon 6 --stride 1 --constraint-form position \
    --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
    --floor-resting-z $FLOOR --seed 42 --eval-every 600 --device cuda \
    > $ROOT/work/logs/post_bounce_$tag.log 2>&1
  say "  $tag exit=$?"
}
evaluate() { # evaluate <tag>
  local tag=$1
  local out=$M/post_$tag
  [ -f "$out/checkpoint.pt" ] || return
  [ -f "$out/evaluation.json" ] && return
  one "$PY" "$G/scripts/evaluate.py" --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output "$out/evaluation.json" \
    --posttrain-report "$out/posttrain_report.json" \
    --history 8 --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
    --floor-resting-z $FLOOR --label "bounce_r08__$tag" --device cuda \
    > $ROOT/work/logs/eval_bounce_$tag.log 2>&1
  say "  eval $tag exit=$?"
}

say "=== 重跑被 bug 作废的 factual, 加预算的 counterfactual, 然后全部评估 ==="
rm -rf "$M/post_mp_factual"
post mp_factual factual 12000
post b3_counterfactual counterfactual 36000
for tag in mp_pretrained mp_jepa mp_factual mp_penalty mp_counterfactual b3_counterfactual; do
  evaluate "$tag"
done
: > $ROOT/work/.done-bounce_fix
say "=== bounce_fix done ==="
