#!/usr/bin/env bash
# A second post-training seed for the bouncing result, which is n = 1.
#
# Seed 43 of post-training only: the pretrained model is shared, so this
# controls the optimiser's own noise rather than pretraining noise.  That is
# the cheaper half of n = 2 and it is the half that can run now; a second
# pretraining seed needs the other GPU, which is busy until the 21-epoch arm
# finishes.
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

run() { # run <tag> <variant> <steps>
  local tag=$1 v=$2 steps=$3
  local out=$M/post_$tag
  [ -f "$out/posttrain_report.json" ] || \
  one "$PY" "$G/scripts/posttrain.py" --checkpoint "$M/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output-dir "$out" --variant "$v" --steps "$steps" --lr 1e-4 \
    --history 8 --horizon 6 --stride 1 --constraint-form position \
    --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
    --floor-resting-z $FLOOR --seed 43 --eval-every 600 --device cuda \
    > $ROOT/work/logs/post_bounce_$tag.log 2>&1
  say "  post $tag exit=$?"
  [ -f "$out/checkpoint.pt" ] || return
  [ -f "$out/evaluation.json" ] || \
  one "$PY" "$G/scripts/evaluate.py" --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output "$out/evaluation.json" \
    --posttrain-report "$out/posttrain_report.json" \
    --history 8 --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
    --floor-resting-z $FLOOR --label "bounce_r08__$tag" --device cuda \
    > $ROOT/work/logs/eval_bounce_$tag.log 2>&1
  say "  eval $tag exit=$?"
}
say "=== bounce seed 43 ==="
run s43_counterfactual counterfactual 36000
run s43_jepa           jepa           12000
run s43_factual        factual        12000
run s43_penalty        penalty        12000
: > $ROOT/work/.done-bounce_s43
say "=== bounce_seed43 done ==="
