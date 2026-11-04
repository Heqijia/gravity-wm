#!/usr/bin/env bash
# Experiment A: the same single-gravity regime with a shape that has visible
# orientation.
#
# ball_single's object is a disc: its silhouette is the same whatever its
# rotation, so the probe only ever has to find a centre.  --shape square gives
# the object an angle drawn from U(-pi, pi) and a spin from U(-0.6, 0.6), so the
# silhouette changes from frame to frame while the centre follows the same free
# fall.  That tests whether the spatial_map read-out's sub-pixel position code
# survives a changing appearance, which is the assumption the whole constraint
# rests on -- and it is the assumption the published [CLS] read-out already
# failed.
#
# Everything else is held at ball_single's settings: g == 4.0 exactly, the
# fixed 2.309 gravity normalisation, 14 epochs, the same launch window.
#
# One new base model, Semigroup-JEPA with the spatial read-out, trained by the
# same upstream loop as every other model here:
#
#   sq_single_s42   data: square_single, g == 4.0 exactly, rotating square
#
# --gravity-scale 2.309 as on ball_single.  With g constant the measured scale
# collapses to the 1e-6 floor and a counterfactual g' in [1, 9] would reach 5e6
# in normalised units; 2.309 is the spread of the counterfactual law itself,
# 8/sqrt(12) for g' ~ U(1, 9).  Every gravity the model is conditioned on then
# lands within about +-2, and the hard part is untouched: the training data
# still carries no information about what gravity does.  On this dataset that is
# exact -- one epsilon of gravity variation, zero signal -- so the constraint is
# the only thing that can teach the conditioning path anything at all.
#
# The launch distribution is identical across all three datasets: rise_range is
# computed over the union of training and held-out gravities, so it is
# [0.806, 1.194] here as well and the shape of the launch carries no gravity
# information.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
W=$ROOT/work
R=/fs/scratch/PAS3151/qijia387/gwm/runs
S=$R/suite
DATA=/fs/scratch/PAS3151/qijia387/gwm/data
LOGS=$W/logs
JOBID=${JOBID:-8093487}
GPU_NEXT=0
TIMEOUT=${TIMEOUT:-5400}
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

# ---------------------------------------------------------------- data
if [ ! -f "$DATA/square_single/generation_manifest.json" ]; then
  rm -rf "$DATA/square_single"
  say "generating square_single (g == 4.0 exactly, rotating square)"
  srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=10 \
    "$PY" "$G/scripts/gen_data.py" --output "$DATA/square_single" --shape square \
    --train-episodes 16000 --heldout-train-gravity-episodes 800 \
    --test-episodes-per-gravity 32 --train-std 0 --workers 9 \
    > "$LOGS/gen_square.log" 2>&1
fi
[ -f "$DATA/square_single/generation_manifest.json" ] || { say "!! square_single generation failed"; exit 1; }

# ---------------------------------------------------------------- pretrain
for spec in "sq_single_s42 $DATA/square_single"; do
  set -- $spec
  name=$1 data=$2
  out=$R/ball_$name
  [ -f "$out/best.pt" ] && continue
  cfg=$W/ball_$name.yaml
  "$PY" - "$cfg" "$name" <<'YAMLEOF'
import sys, yaml
cfg = yaml.safe_load(open("/users/PAS3151/qijia387/Phys-WM/gravity-wm/configs/ball_sg_jepa_gru.yaml"))
cfg["name"] = f"ball_{sys.argv[2]}"
cfg["train"]["epochs"] = 14
open(sys.argv[1], "w").write(yaml.safe_dump(cfg, sort_keys=False))
YAMLEOF
  TIMEOUT=21600 launch "train_$name" "$PY" "$G/scripts/pretrain.py" \
    --config "$cfg" --data "$data" --output-dir "$out" \
    --readout spatial_map --gravity-scale 2.309 --device cuda
done
wait

# ---------------------------------------------------------------- cache, probes
for spec in "sq_single_s42 $DATA/square_single"; do
  set -- $spec
  name=$1 data=$2
  src=$R/ball_$name dir=$S/$name
  [ -f "$src/best.pt" ] || { say "  !! no checkpoint for $name"; continue; }
  mkdir -p "$dir"
  if [ ! -f "$dir/checkpoint.pt" ]; then
    cp "$src/best.pt" "$dir/checkpoint.pt"
    for f in normalization_stats.json train_val_split.json readout.json; do
      [ -f "$src/$f" ] && cp "$src/$f" "$dir/$f"
    done
  fi
  [ -f "$dir/latents.pt" ] || launch "cache_$name" "$PY" "$G/scripts/build_cache.py" \
    --checkpoint "$dir/checkpoint.pt" --data "$data" --output "$dir/latents.pt" --device cuda
done
wait
for name in sq_single_s42; do
  dir=$S/$name
  [ -f "$dir/latents.pt" ] || continue
  [ -f "$dir/probes/probe_eval.pt" ] || launch "probes_$name" "$PY" "$G/scripts/train_probes.py" \
    --cache "$dir/latents.pt" --output-dir "$dir/probes" --device cuda \
    --history 8 --horizon 6 --stride 1 --max-episodes 6000
done
wait

# ---------------------------------------------------------------- post-train
for name in sq_single_s42; do
  dir=$S/$name
  [ -f "$dir/probes/probe_eval.pt" ] || { say "  !! $name has no probes"; continue; }
  for variant in pretrained jepa factual penalty counterfactual; do
    out=$dir/post_mp_$variant
    [ -f "$out/posttrain_report.json" ] && continue
    launch "post_${name}_mp_$variant" "$PY" "$G/scripts/posttrain.py" \
      --checkpoint "$dir/checkpoint.pt" --cache "$dir/latents.pt" \
      --probe-dir "$dir/probes" --output-dir "$out" --variant "$variant" \
      --steps 4000 --lr 1e-4 --history 8 --horizon 6 --stride 1 \
      --constraint-form position --epsilon-source factual_rollout \
      --gravity-low 1.0 --gravity-high 9.0 --seed 42 --eval-every 200 --device cuda
  done
done
wait

# ---------------------------------------------------------------- evaluate, H = 6 and 8
for name in sq_single_s42; do
  dir=$S/$name
  for variant in pretrained jepa factual penalty counterfactual; do
    out=$dir/post_mp_$variant
    [ -f "$out/checkpoint.pt" ] || continue
    for h in 6 8; do
      target=$out/evaluation.json
      [ "$h" = 8 ] && target=$out/evaluation_h8.json
      [ -f "$target" ] && continue
      launch "eval_${name}_mp_${variant}_h$h" "$PY" "$G/scripts/evaluate.py" \
        --checkpoint "$out/checkpoint.pt" --cache "$dir/latents.pt" \
        --probe-dir "$dir/probes" --output "$target" \
        --posttrain-report "$out/posttrain_report.json" \
        --history 8 --horizon "$h" --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
        --label "${name}__mp_${variant}" --device cuda
    done
  done
done
wait
: > "$(dirname "$0")/.done-square"
say "=== done ==="
