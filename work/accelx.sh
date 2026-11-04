#!/usr/bin/env bash
# Experiment B: the counterfactual is an acceleration *vector*, not a magnitude.
#
# Every experiment so far asks the model about a different gravity *magnitude*,
# so the counterfactual set is one-dimensional and the answer always lies on the
# vertical axis the training data already explores.  A reviewer's first question
# is whether the gain is an artefact of that: a predictor could do well by
# learning a single scalar knob.
#
# ball_ax adds a horizontal acceleration as a fourth action coordinate, held at
# exactly zero in every training episode, and the post-training draws
#
#     g'   ~ U(1, 9)        a_x' ~ U(-3, 3)
#
# so the counterfactual acceleration (a_x', -g') ranges over a two-dimensional
# set of which the training data occupies a single point, (0, -4).  The model
# has never seen *any* horizontal acceleration, so the constraint is the only
# thing that can teach the conditioning path what one does -- the same regime as
# ball_single, one dimension up.
#
# Nothing in the method changes.  The violation, Lemma 1's targets and
# Proposition 1's exact path were already functions of a planar acceleration
# vector; only the scalar-to-vector conversion and the action conditioning knew
# otherwise, and both are now general (gwm.constraint.acceleration_vector,
# condition_on_acceleration).  The scalar path is unchanged: a re-run of an
# existing evaluation differs in no reported column.
#
# Both normalising scales have to be fixed, because both coordinates are
# constant in the training data and would otherwise collapse to the 1e-6 floor:
#   gravity  2.309 = 8/sqrt(12), the spread of U(1, 9)
#   a_x      1.732 = 6/sqrt(12), the spread of U(-3, 3)
#
# Two evaluations per checkpoint, reported as two one-dimensional slices rather
# than a grid: the gravity sweep at a_x = 0 (comparable with every other
# experiment) and the a_x sweep at g = 4 (the direction the model has never
# seen).  A grid costs the product and says little more.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
W=$ROOT/work
R=/fs/scratch/PAS3151/qijia387/gwm/runs
S=$R/suite
DATA=/fs/scratch/PAS3151/qijia387/gwm/data/ball_ax
LOGS=$W/logs
JOBID=${JOBID:-8093487}
GPU_NEXT=0
TIMEOUT=${TIMEOUT:-5400}
AX_INDEX=3
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
launch() {
  local name=$1; shift
  local gpu=$GPU_NEXT; GPU_NEXT=$(( (GPU_NEXT + 1) % 2 ))
  ( timeout --signal=TERM --kill-after=60 "$TIMEOUT" \
      srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=5 --gres=gpu:2 \
        env CUDA_VISIBLE_DEVICES=$gpu PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
        "$@" > "$LOGS/$name.log" 2>&1
    printf '%s exit=%s\n' "$name" "$?" >> "$LOGS/_exits.txt" ) &
  say "  launched $name on gpu $gpu"
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
}

[ -f "$DATA/generation_manifest.json" ] || { say "!! $DATA missing; run gen_data with --with-accel-x"; exit 1; }
name=ax_s42
out=$R/ball_$name
dir=$S/$name

if [ ! -f "$out/best.pt" ]; then
  cfg=$W/ball_$name.yaml
  "$PY" - "$cfg" <<'YAMLEOF'
import sys, yaml
cfg = yaml.safe_load(open("/users/PAS3151/qijia387/Phys-WM/gravity-wm/configs/ball_sg_jepa_gru.yaml"))
cfg["name"] = "ball_ax"
cfg["train"]["epochs"] = 14
cfg["model"]["action_dim"] = 4       # the fourth coordinate is a_x
open(sys.argv[1], "w").write(yaml.safe_dump(cfg, sort_keys=False))
YAMLEOF
  TIMEOUT=21600 launch "train_$name" "$PY" "$G/scripts/pretrain.py" \
    --config "$cfg" --data "$DATA" --output-dir "$out" --readout spatial_map \
    --gravity-scale 2.309 --accel-x-scale 1.732 --accel-x-index "$AX_INDEX" \
    --device cuda
  wait
fi
[ -f "$out/best.pt" ] || { say "!! training produced no checkpoint"; exit 1; }

mkdir -p "$dir"
if [ ! -f "$dir/checkpoint.pt" ]; then
  cp "$out/best.pt" "$dir/checkpoint.pt"
  for f in normalization_stats.json train_val_split.json readout.json; do
    [ -f "$out/$f" ] && cp "$out/$f" "$dir/$f"
  done
fi
[ -f "$dir/latents.pt" ] || { launch "cache_$name" "$PY" "$G/scripts/build_cache.py" \
  --checkpoint "$dir/checkpoint.pt" --data "$DATA" --output "$dir/latents.pt" --device cuda; wait; }
[ -f "$dir/probes/probe_eval.pt" ] || { launch "probes_$name" "$PY" "$G/scripts/train_probes.py" \
  --cache "$dir/latents.pt" --output-dir "$dir/probes" --device cuda \
  --history 8 --horizon 6 --stride 1 --max-episodes 6000; wait; }
[ -f "$dir/probes/probe_eval.pt" ] || { say "!! no probes"; exit 1; }

say "PHASE accelx: five variants, two-dimensional counterfactual"
for variant in pretrained jepa factual penalty counterfactual; do
  o=$dir/post_mp_$variant
  [ -f "$o/posttrain_report.json" ] && continue
  launch "post_${name}_mp_$variant" "$PY" "$G/scripts/posttrain.py" \
    --checkpoint "$dir/checkpoint.pt" --cache "$dir/latents.pt" \
    --probe-dir "$dir/probes" --output-dir "$o" --variant "$variant" \
    --steps 4000 --lr 1e-4 --history 8 --horizon 6 --stride 1 \
    --constraint-form position --epsilon-source factual_rollout \
    --gravity-low 1.0 --gravity-high 9.0 \
    --accel-x-index "$AX_INDEX" --accel-x-low -3.0 --accel-x-high 3.0 \
    --seed 42 --eval-every 200 --device cuda
done
wait

say "PHASE accelx: evaluating the gravity slice and the horizontal slice"
for variant in pretrained jepa factual penalty counterfactual; do
  o=$dir/post_mp_$variant
  [ -f "$o/checkpoint.pt" ] || continue
  # the gravity slice at a_x = 0, directly comparable with every other model
  for h in 6 8; do
    target=$o/evaluation.json; [ "$h" = 8 ] && target=$o/evaluation_h8.json
    [ -f "$target" ] || launch "eval_${name}_mp_${variant}_h$h" "$PY" "$G/scripts/evaluate.py" \
      --checkpoint "$o/checkpoint.pt" --cache "$dir/latents.pt" \
      --probe-dir "$dir/probes" --output "$target" \
      --posttrain-report "$o/posttrain_report.json" \
      --history 8 --horizon "$h" --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
      --label "${name}__mp_${variant}" --device cuda
  done
  # the horizontal slice at g = 4: the direction never seen in training
  [ -f "$o/evaluation_ax.json" ] || launch "eval_${name}_mp_${variant}_ax" "$PY" "$G/scripts/evaluate.py" \
    --checkpoint "$o/checkpoint.pt" --cache "$dir/latents.pt" \
    --probe-dir "$dir/probes" --output "$o/evaluation_ax.json" \
    --posttrain-report "$o/posttrain_report.json" \
    --history 8 --horizon 6 --stride 1 \
    --accel-x-index "$AX_INDEX" --accel-x-min -3.0 --accel-x-max 3.0 --accel-x-step 0.25 \
    --label "${name}__ax_${variant}" --device cuda
done
wait
: > "$W/.done-accelx"
say "=== accelx done ==="
