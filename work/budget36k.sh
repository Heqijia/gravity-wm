#!/usr/bin/env bash
# Is the feasibility certificate attainable in the regime the story rests on?
#
# Nothing certifies on ball_narrow or ball_single, which is awkward because
# those are exactly the regimes where the method's accuracy gain is 9-10x.  The
# step budget is the reason, and it is measurable: tripling it from 4000 to
# 12000 moved C/eps monotonically toward the tolerance and improved *both*
# axes.
#
#   model          4000 steps          12000 steps        threshold
#   narrow_s42     1.36                1.070              1.042   (misses by 2.7%)
#   single_s42     1.76                1.566              1.105
#
#   pos far        3.398 -> 3.250      JEPA L  13.682 -> 8.496   (2-seed means)
#
# narrow_s42 misses its own tolerance by 2.7% after 12000 steps.  Tripling again
# settles whether the certificate is reachable at all here or whether something
# other than the budget blocks it; either answer is worth having, because C3 --
# "the returned iterate carries Theorem 1's certificate" -- is the claim that
# distinguishes the method from a penalty, and it is currently unsupported in
# the headline regime.
#
# 36000 steps is about 2.5 h per job, two jobs, both GPUs.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
LOGS=$ROOT/work/logs
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
say "=== 36000 steps, the two runs closest to their tolerance ==="
for model in narrow_s42 single_s42; do
  out=$R/$model/post_b9_counterfactual
  [ -f "$out/posttrain_report.json" ] && continue
  launch "post_${model}_b9" "$PY" "$G/scripts/posttrain.py" \
    --checkpoint "$R/$model/checkpoint.pt" --cache "$R/$model/latents.pt" \
    --probe-dir "$R/$model/probes" --output-dir "$out" --variant counterfactual \
    --steps 36000 --lr 1e-4 --history 8 --horizon 6 --stride 1 \
    --constraint-form position --epsilon-source factual_rollout \
    --gravity-low 1.0 --gravity-high 9.0 --seed 42 --eval-every 1200 --device cuda
done
wait
for model in narrow_s42 single_s42; do
  out=$R/$model/post_b9_counterfactual
  [ -f "$out/checkpoint.pt" ] || continue
  for h in 6 8; do
    target=$out/evaluation.json; [ "$h" = 8 ] && target=$out/evaluation_h8.json
    [ -f "$target" ] && continue
    launch "eval_${model}_b9_h$h" "$PY" "$G/scripts/evaluate.py" \
      --checkpoint "$out/checkpoint.pt" --cache "$R/$model/latents.pt" \
      --probe-dir "$R/$model/probes" --output "$target" \
      --posttrain-report "$out/posttrain_report.json" \
      --history 8 --horizon "$h" --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
      --label "${model}__b9_counterfactual" --device cuda
  done
done
wait
: > "$ROOT/work/.done-budget36k"
say "=== budget36k done ==="
