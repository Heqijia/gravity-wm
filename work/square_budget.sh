#!/usr/bin/env bash
# The rotating square, at the budget its own C_0/eps implies.
#
# sq_single_s42 reached the accuracy (2.903 px, response 1.055, 9.1x better than
# continued JEPA training) but paid 6.3x on the JEPA loss, and the reason is
# measurable rather than mysterious:
#
#   model            eps      C_0/eps  restore  optim   L ratio
#   spatial_s42      0.0668       2.0     2402   1598     0.52
#   narrow_s42       0.0517      17.5     1767   2233     0.23
#   narrowfx_s42     0.0255      26.9     3240    760     0.69
#   single_s42       0.0137      55.0     3708    292     0.92
#   ssm_s42          0.0182      87.5     4000      0    92.89
#   sq_single_s42    0.0082     100.9     4000      0     6.32
#
# The two runs with *zero* OPTIMISE steps are exactly the two whose loss blew
# up; every run with at least 292 of them ended at or below its starting loss.
# With no OPTIMISE step there is also no certificate, because Theorem 1(c) reads
# it off one.
#
# And C_0/eps is predictable: it correlates -0.88 with log eps, because eps is
# calibrated against the predictor's *own* factual error while C_0 is set by how
# wrong its counterfactual is.  A predictor that is better factually gets a
# tighter tolerance and therefore starts relatively further outside it.  The
# square's probe is the best of any model here (0.072 px on the training split),
# which is exactly why its eps is the tightest and its C_0/eps the largest.
#
# Two arms, because they test different remedies for the same cause:
#   b6    24000 steps at the calibrated eps -- more budget, nothing else
#   an    eps x 8 first (C_0/eps ~ 12.6), then the calibrated eps from there --
#         a staged tolerance, so OPTIMISE steps happen in both stages
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
LOGS=$ROOT/work/logs
M=$R/sq_single_s42
JOBID=${JOBID:-8093487}
GPU_NEXT=0
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
launch() {
  local name=$1; shift
  local gpu=$GPU_NEXT; GPU_NEXT=$(( (GPU_NEXT + 1) % 2 ))
  ( timeout --signal=TERM --kill-after=60 21600 \
      srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=5 --gres=gpu:2 \
        env CUDA_VISIBLE_DEVICES=$gpu PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
        "$@" > "$LOGS/$name.log" 2>&1
    printf '%s exit=%s\n' "$name" "$?" >> "$LOGS/_exits.txt" ) &
  say "  launched $name on gpu $gpu"
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
}
post() { # post <tag> <checkpoint> <steps> <epsilon-scale>
  local tag=$1 ckpt=$2 steps=$3 scale=$4 out=$M/post_$1
  [ -f "$out/posttrain_report.json" ] && return
  launch "post_sq_single_s42_${tag}" "$PY" "$G/scripts/posttrain.py" \
    --checkpoint "$ckpt" --cache "$M/latents.pt" --probe-dir "$M/probes" \
    --output-dir "$out" --variant counterfactual --steps "$steps" --lr 1e-4 \
    --history 8 --horizon 6 --stride 1 --constraint-form position \
    --epsilon-source factual_rollout --epsilon-scale "$scale" \
    --gravity-low 1.0 --gravity-high 9.0 --seed 42 --eval-every 800 --device cuda
}

say "=== square: more budget, and a staged tolerance ==="
post b6_counterfactual "$M/checkpoint.pt" 24000 1.0
post an1_counterfactual "$M/checkpoint.pt" 4000 8.0
wait
post an2_counterfactual "$M/post_an1_counterfactual/checkpoint.pt" 4000 1.0
wait
for tag in b6_counterfactual an1_counterfactual an2_counterfactual; do
  out=$M/post_$tag
  [ -f "$out/checkpoint.pt" ] || continue
  for h in 6 8; do
    target=$out/evaluation.json; [ "$h" = 8 ] && target=$out/evaluation_h8.json
    [ -f "$target" ] && continue
    launch "eval_sq_single_s42_${tag}_h$h" "$PY" "$G/scripts/evaluate.py" \
      --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
      --probe-dir "$M/probes" --output "$target" \
      --posttrain-report "$out/posttrain_report.json" \
      --history 8 --horizon "$h" --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
      --label "sq_single_s42__${tag}" --device cuda
  done
done
wait
: > "$ROOT/work/.done-square_budget"
say "=== square_budget done ==="
