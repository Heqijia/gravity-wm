#!/usr/bin/env bash
# The SSM baseline at a step size chosen by stability, not by gradient norm.
#
# The gradient-scale rule (scripts/measure_gradient_scale.py) put the SSM at
# lr 1.4e-6 and it still diverged: its ||grad L|| at theta_0 is 3.3e4 but the
# median along the trajectory is 2.4e5, so the norm at the starting point
# underestimates the step the model will actually take by an order of
# magnitude.  Measured directly, 600 steps of the `jepa` control:
#
#   lr        L after 600 steps     ||d||
#   1.4e-6              1 854       pinned at the 1e4 clip
#   1e-7                  291       pinned at the 1e4 clip
#   1e-8                    7.20    pinned at the 1e4 clip
#   1e-9                    6.65    1190-3737, off the clip
#
# So the SSM's usable learning rate is 1e-8 to 1e-9, ten thousand times
# smaller than the GRU's 1e-4.  1e-8 is the primary run; 1e-9 is a sensitivity
# check on `jepa` and `counterfactual`, and is the only setting where the
# gradient clip stops engaging.
#
# Caveat recorded in the results: the Polyak RESTORE step is self-scaling and
# therefore lr-independent, while OPTIMISE uses lr.  Lowering lr by 10^4 shifts
# the balance between the two modes by the same factor, so the SSM runs are not
# the same algorithm-as-tuned that the GRU runs are.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
LOGS=$ROOT/work/logs
JOBID=${JOBID:-8093487}
GPU_NEXT=0
M=$R/ssm_s42
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
launch() {
  local name=$1; shift
  local gpu=$GPU_NEXT; GPU_NEXT=$(( (GPU_NEXT + 1) % 2 ))
  (
    timeout --signal=TERM --kill-after=60 5400 \
      srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=5 --gres=gpu:2 \
        env CUDA_VISIBLE_DEVICES=$gpu PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
        "$@" > "$LOGS/$name.log" 2>&1
    printf '%s exit=%s\n' "$name" "$?" >> "$LOGS/_exits.txt"
  ) &
  say "  launched $name on gpu $gpu"
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
}
run() { # run <tag> <variant> <lr>
  local tag=$1 variant=$2 lr=$3 out=$M/post_$1
  [ -f "$out/posttrain_report.json" ] && return
  launch "post_ssm_s42_${tag}" "$PY" "$G/scripts/posttrain.py" \
    --checkpoint "$M/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output-dir "$out" --variant "$variant" \
    --steps 4000 --lr "$lr" --history 8 --horizon 6 --stride 1 \
    --constraint-form position --epsilon-source factual_rollout \
    --gravity-low 1.0 --gravity-high 9.0 --seed 42 --eval-every 200 --device cuda
}

# the mis-scaled lr 1.4e-6 runs move aside; mp_ becomes the lr 1e-8 ones
for variant in pretrained jepa factual penalty counterfactual; do
  [ -d "$M/post_mp_$variant" ] && [ ! -d "$M/post_mplr_$variant" ] &&
    mv "$M/post_mp_$variant" "$M/post_mplr_$variant"
done
say "=== SSM at lr 1e-8 (primary) ==="
for variant in pretrained jepa factual penalty counterfactual; do
  run "mp_$variant" "$variant" 1e-8
done
say "=== SSM at lr 1e-9 (sensitivity) ==="
run "lo_jepa" jepa 1e-9
run "lo_counterfactual" counterfactual 1e-9
wait
say "evaluating"
for tag in mp_pretrained mp_jepa mp_factual mp_penalty mp_counterfactual lo_jepa lo_counterfactual; do
  out=$M/post_$tag
  [ -f "$out/checkpoint.pt" ] || continue
  [ -f "$out/evaluation.json" ] && continue
  launch "eval_ssm_s42_${tag}" "$PY" "$G/scripts/evaluate.py" \
    --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output "$out/evaluation.json" \
    --posttrain-report "$out/posttrain_report.json" \
    --history 8 --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
    --label "ssm_s42__$tag" --device cuda
done
wait
say "=== done ==="
