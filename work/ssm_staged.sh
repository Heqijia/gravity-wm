#!/usr/bin/env bash
# Staged tolerance on the SSM, using only existing flags.
#
# The tolerance test settled why the SSM failed.  It starts at C/eps = 87.5
# (the GRU starts at 2.0) and never reaches OPTIMISE in 4000 steps:
#
#   eps scale   C_0/eps   pos far   resp far   JEPA L
#   x1             87.5     28.980     -0.558    476.4   never left RESTORE
#   x10             8.75    44.283     -0.693   3247.4   still fails
#   x40             2.2     16.385     +0.517      6.8   works
#
# At x40 the SSM starts where the GRU starts, and the method then does what it
# does on the GRU: position error 22.964 -> 16.385, response 0.378 -> 0.517,
# and the JEPA loss *improves* on the pretrained 8.551.  So the failure was the
# distance to the tolerance, not the smoothness of C -- and the usable range is
# roughly C_0/eps <= 2-3, since x10 (8.75) still fails.
#
# That is what a staged tolerance is for: reach x40, then restart from the
# result with a tighter one.  Each stage recalibrates eps against its own
# starting point, so stage two begins near its own boundary rather than 9x
# outside it.  No code change -- just --epsilon-scale and the previous stage's
# checkpoint.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
LOGS=$ROOT/work/logs
M=$R/ssm_s42
JOBID=${JOBID:-8093487}
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
run() { # run <name> <checkpoint> <outdir> <epsilon-scale>
  local name=$1 ckpt=$2 out=$3 scale=$4
  if [ ! -f "$out/posttrain_report.json" ]; then
    say "  $name (eps x $scale) from $(basename "$(dirname "$ckpt")")"
    timeout --signal=TERM --kill-after=60 5400 \
      srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=5 --gres=gpu:2 \
        env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
        "$PY" "$G/scripts/posttrain.py" \
        --checkpoint "$ckpt" --cache "$M/latents.pt" --probe-dir "$M/probes" \
        --output-dir "$out" --variant counterfactual --steps 4000 --lr 1e-8 \
        --history 8 --horizon 6 --stride 1 --constraint-form position \
        --epsilon-source factual_rollout --epsilon-scale "$scale" \
        --gravity-low 1.0 --gravity-high 9.0 --seed 42 --eval-every 200 \
        --device cuda > "$LOGS/$name.log" 2>&1
    printf '%s exit=%s\n' "$name" "$?" >> "$LOGS/_exits.txt"
  fi
  if [ -f "$out/checkpoint.pt" ] && [ ! -f "$out/evaluation.json" ]; then
    srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=5 --gres=gpu:2 \
      env CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
      "$PY" "$G/scripts/evaluate.py" \
      --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
      --probe-dir "$M/probes" --output "$out/evaluation.json" \
      --posttrain-report "$out/posttrain_report.json" \
      --history 8 --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
      --label "ssm_s42__$(basename "$out" | sed s/post_//)" --device cuda \
      > "$LOGS/eval_$name.log" 2>&1
  fi
}

say "=== SSM, staged tolerance: x40 -> x10 -> x4 -> x1 ==="
run ssm_stage1 "$M/post_rx40_counterfactual/checkpoint.pt" "$M/post_st2_counterfactual" 10
run ssm_stage2 "$M/post_st2_counterfactual/checkpoint.pt"  "$M/post_st3_counterfactual" 4
run ssm_stage3 "$M/post_st3_counterfactual/checkpoint.pt"  "$M/post_st4_counterfactual" 1
: > "$(dirname "$0")/.done-ssm_staged"
say "=== done ==="
