#!/usr/bin/env bash
# Full pipeline for the gravity-constrained post-training study.
#
# Stages, in order:
#   data      contact-free projectile dataset (CPU-parallel MuJoCo rendering)
#   pretrain  Semigroup-JEPA world models, one per seed, one GPU each
#   cache     encode every episode once with each frozen encoder
#   probes    position probe h and independent evaluation probe h_eval
#   posttrain the five variants per seed, two at a time across the two GPUs
#   evaluate  counterfactual sweep, Corollary 2 bound, ablation metrics
#   summary   the comparison table
#   figures   the four panels of the figure set
#
# Usage: bash run_all.sh <stage> [stage ...]
set -euo pipefail

ROOT=${ROOT:-/users/PAS3151/qijia387/Phys-WM}
PY=${PY:-$ROOT/.venv-wm/bin/python}
GWM=$ROOT/gravity-wm
WORK=${WORK:-/fs/scratch/PAS3151/qijia387/gwm}
DATA=${DATA:-$WORK/data/ball_v1}
RUNS=${RUNS:-$WORK/runs}
TAG=${TAG:-main}
SEEDS=${SEEDS:-"42 43"}
VARIANTS=${VARIANTS:-"pretrained jepa factual penalty counterfactual"}
NGPU=${NGPU:-2}
CONFIG=${CONFIG:-$GWM/configs/ball_sg_jepa_gru.yaml}
# The published [CLS] read-out does not expose object position in the latent,
# so the position probe lands at chance and the constraint has no margin
# (0.68 against a required 3).  See docs/design.md.  "spatial_map" keeps the
# spatial token axis and reaches a sub-pixel probe after one epoch.
READOUT=${READOUT:-spatial_map}

# Constraint shape; history+horizon must not exceed the episode length.
HISTORY=${HISTORY:-8}
HORIZON=${HORIZON:-6}
STRIDE=${STRIDE:-3}
STEPS=${STEPS:-3000}
LR=${LR:-2.0e-4}
GATE=${GATE:-0.9}
GLOW=${GLOW:-1.0}
GHIGH=${GHIGH:-9.0}
# real_latents is the method note's c_real; factual_rollout also carries the
# predictor's own rollout error, which makes the tolerance attainable.
EPSILON_SOURCE=${EPSILON_SOURCE:-factual_rollout}

mkdir -p "$RUNS" "$WORK/logs"

# Run a command on a free GPU, keeping at most NGPU in flight.
declare -a PIDS=()
declare -a SLOTS=()
submit() {
  local name=$1; shift
  local gpu=-1
  while [ $gpu -lt 0 ]; do
    for slot in $(seq 0 $((NGPU - 1))); do
      local pid=${SLOTS[$slot]:-}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then gpu=$slot; break; fi
    done
    [ $gpu -lt 0 ] && sleep 3
  done
  echo ">> [gpu $gpu] $name"
  ( CUDA_VISIBLE_DEVICES=$gpu srun --jobid="${SLURM_HOLD_JOBID:?set SLURM_HOLD_JOBID}" \
      --overlap -n1 --cpus-per-task=10 --gres=gpu:$NGPU \
      env CUDA_VISIBLE_DEVICES=$gpu "$@" ) > "$WORK/logs/${TAG}_${name}.log" 2>&1 &
  local pid=$!
  SLOTS[$gpu]=$pid
  PIDS+=("$pid:$name")
}
join() {
  local failed=0
  for entry in "${PIDS[@]}"; do
    local pid=${entry%%:*} name=${entry#*:}
    if wait "$pid"; then echo "   ok   $name"; else echo "   FAIL $name (see $WORK/logs/${TAG}_${name}.log)"; failed=1; fi
  done
  PIDS=(); SLOTS=()
  [ $failed -eq 0 ] || { echo "stage failed"; exit 1; }
}

stage_data() {
  [ -d "$DATA" ] && { echo "data already present: $DATA"; return; }
  $PY "$GWM/scripts/gen_data.py" --output "$DATA" \
    --train-episodes "${TRAIN_EPISODES:-16000}" \
    --heldout-train-gravity-episodes "${HELDOUT_EPISODES:-600}" \
    --test-episodes-per-gravity "${TEST_PER_G:-32}" \
    --test-gravity-min "$GLOW" --test-gravity-max "$GHIGH" \
    --workers "${GEN_WORKERS:-20}"
}

stage_pretrain() {
  for seed in $SEEDS; do
    local out=$RUNS/$TAG/pretrain_s$seed
    [ -f "$out/checkpoint.pt" ] && { echo "skip pretrain s$seed"; continue; }
    local cfg=$WORK/logs/${TAG}_pretrain_s$seed.yaml
    $PY - "$CONFIG" "$seed" "$cfg" <<'PYEOF'
import sys, yaml
payload = yaml.safe_load(open(sys.argv[1]).read())
payload["train"]["seed"] = int(sys.argv[2])
payload["name"] = f"{payload['name']}_s{sys.argv[2]}"
open(sys.argv[3], "w").write(yaml.safe_dump(payload, sort_keys=False))
PYEOF
    submit "pretrain_s$seed" $PY "$GWM/scripts/pretrain.py" --config "$cfg" \
      --data "$DATA" --output-dir "$out" --readout "$READOUT" --device cuda
  done
  join
}

stage_cache() {
  for seed in $SEEDS; do
    local out=$RUNS/$TAG/pretrain_s$seed
    [ -f "$out/latents.pt" ] && { echo "skip cache s$seed"; continue; }
    submit "cache_s$seed" $PY "$GWM/scripts/build_cache.py" \
      --checkpoint "$out/checkpoint.pt" --data "$DATA" --output "$out/latents.pt" --device cuda
  done
  join
}

stage_probes() {
  for seed in $SEEDS; do
    local out=$RUNS/$TAG/pretrain_s$seed
    [ -f "$out/probes/probe_eval.pt" ] && { echo "skip probes s$seed"; continue; }
    submit "probes_s$seed" $PY "$GWM/scripts/train_probes.py" \
      --cache "$out/latents.pt" --output-dir "$out/probes" --device cuda \
      --history "$HISTORY" --horizon "$HORIZON" --stride "$STRIDE"
  done
  join
}

stage_posttrain() {
  for seed in $SEEDS; do
    local base=$RUNS/$TAG/pretrain_s$seed
    for variant in $VARIANTS; do
      local out=$RUNS/$TAG/post_${variant}_s$seed
      [ -f "$out/checkpoint.pt" ] && { echo "skip post $variant s$seed"; continue; }
      submit "post_${variant}_s$seed" $PY "$GWM/scripts/posttrain.py" \
        --checkpoint "$base/checkpoint.pt" --cache "$base/latents.pt" \
        --probe-dir "$base/probes" --output-dir "$out" --variant "$variant" \
        --steps "$STEPS" --lr "$LR" --gate "$GATE" \
        --epsilon-source "$EPSILON_SOURCE" \
        --gravity-low "$GLOW" --gravity-high "$GHIGH" \
        --history "$HISTORY" --horizon "$HORIZON" --stride "$STRIDE" \
        --seed "$seed" --device cuda
    done
  done
  join
}

stage_evaluate() {
  for seed in $SEEDS; do
    local base=$RUNS/$TAG/pretrain_s$seed
    for variant in $VARIANTS; do
      local out=$RUNS/$TAG/post_${variant}_s$seed
      submit "eval_${variant}_s$seed" $PY "$GWM/scripts/evaluate.py" \
        --checkpoint "$out/checkpoint.pt" --cache "$base/latents.pt" \
        --probe-dir "$base/probes" --output "$out/evaluation.json" \
        --posttrain-report "$out/posttrain_report.json" \
        --history "$HISTORY" --horizon "$HORIZON" --stride "$STRIDE" \
        --sweep-min "$GLOW" --sweep-max "$GHIGH" \
        --label "${variant}__s$seed" --device cuda
    done
  done
  join
  for seed in $SEEDS; do
    for variant in $VARIANTS; do
      $PY - "$RUNS/$TAG/post_${variant}_s$seed/evaluation.json" "$variant" <<'PYEOF'
import json, sys
path = sys.argv[1]
payload = json.loads(open(path).read())
payload["variant"] = sys.argv[2]
open(path, "w").write(json.dumps(payload, indent=2, sort_keys=True))
PYEOF
    done
  done
}

stage_supervise() {
  $PY "$GWM/scripts/supervise.py" \
    --runs $(for seed in $SEEDS; do printf '%s ' "$RUNS/$TAG/pretrain_s$seed"; done) \
    --data "$DATA" --log "$RUNS/$TAG/supervise.tsv" --include-best \
    --history "$HISTORY" --device cuda
}

stage_summary() {
  local files=()
  for seed in $SEEDS; do
    for variant in $VARIANTS; do
      local file=$RUNS/$TAG/post_${variant}_s$seed/evaluation.json
      [ -f "$file" ] && files+=("$file")
    done
  done
  [ ${#files[@]} -gt 0 ] || { echo "no evaluations yet"; return 1; }
  $PY "$GWM/scripts/report.py" --evaluations "${files[@]}" \
    --output "$RUNS/$TAG/report.json" | tee "$RUNS/$TAG/report.txt"
  $PY "$GWM/scripts/summarize.py" --evaluations "${files[@]}" \
    --output "$RUNS/$TAG/summary.json" | tee "$RUNS/$TAG/summary.txt"
}

stage_figures() {
  local files=() seed_one
  seed_one=$(echo $SEEDS | awk '{print $1}')
  for seed in $SEEDS; do
    for variant in $VARIANTS; do
      local file=$RUNS/$TAG/post_${variant}_s$seed/evaluation.json
      [ -f "$file" ] && files+=("$file")
    done
  done
  MPLBACKEND=Agg $PY "$GWM/scripts/figures.py" \
    ${files[@]+--evaluations "${files[@]}"} \
    --posttrain-report "$RUNS/$TAG/post_counterfactual_s${seed_one}/posttrain_report.json" \
    --supervise-log "$RUNS/$TAG/supervise.tsv" \
    --output "$RUNS/$TAG/figures.png"
}

for stage in "$@"; do
  echo "=== stage: $stage ==="
  "stage_$stage"
done
