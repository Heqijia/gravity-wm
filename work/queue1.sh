#!/usr/bin/env bash
# Queue 1: shore up what is weakest before adding anything new.
#
# The headline result -- single_s42, 11.2x better than continued JEPA training
# with the response going -0.373 -> 0.935 -- rests on ONE seed.  A second seed
# on the two models that carry the story matters more than a new environment.
#
# Then the tolerance/penalty frontier in the regime that actually separates the
# methods.  On ball_v1 a swept penalty (lambda = 300) beat the method's whole
# certified frontier, which is the honest weak point of the comparison.  On
# ball_single the penalty reaches only response 0.019 -- gravity-blind -- at
# lambda = 1, so the question is whether sweeping lambda rescues it there too.
# If it does not, the frontier claim is secure in the regime the method is for.
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

post() { # post <model> <tag> <variant> [extra...]
  local model=$1 tag=$2 variant=$3; shift 3
  local dir=$S/$model out=$S/$model/post_$tag
  [ -f "$out/posttrain_report.json" ] && return
  [ -f "$dir/probes/probe_eval.pt" ] || { say "  !! $model has no probes"; return; }
  launch "post_${model}_${tag}" "$PY" "$G/scripts/posttrain.py" \
    --checkpoint "$dir/checkpoint.pt" --cache "$dir/latents.pt" \
    --probe-dir "$dir/probes" --output-dir "$out" --variant "$variant" \
    --steps 4000 --lr 1e-4 --history 8 --horizon 6 --stride 1 \
    --constraint-form position --epsilon-source factual_rollout \
    --gravity-low 1.0 --gravity-high 9.0 --eval-every 200 --device cuda "$@"
}
evaluate() { # evaluate <model> <tag>
  local model=$1 tag=$2 dir=$S/$1 out=$S/$1/post_$2
  [ -f "$out/checkpoint.pt" ] || return
  for h in 6 8; do
    local target=$out/evaluation.json
    [ "$h" = 8 ] && target=$out/evaluation_h8.json
    [ -f "$target" ] && continue
    launch "eval_${model}_${tag}_h$h" "$PY" "$G/scripts/evaluate.py" \
      --checkpoint "$out/checkpoint.pt" --cache "$dir/latents.pt" \
      --probe-dir "$dir/probes" --output "$target" \
      --posttrain-report "$out/posttrain_report.json" \
      --history 8 --horizon "$h" --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
      --label "${model}__${tag}" --device cuda
  done
}

# ---------------------------------------------------------- 1. second seeds
say "PHASE seeds: a second seed on the two models that carry the story"
for spec in "single_s43 $DATA/ball_single 43" "narrowfx_s43 $DATA/ball_narrow 43"; do
  set -- $spec
  name=$1 data=$2 seed=$3
  out=$R/ball_$name
  [ -f "$out/best.pt" ] && continue
  cfg=$W/ball_$name.yaml
  "$PY" - "$cfg" "$name" "$seed" <<'YAMLEOF'
import sys, yaml
cfg = yaml.safe_load(open("/users/PAS3151/qijia387/Phys-WM/gravity-wm/configs/ball_sg_jepa_gru.yaml"))
cfg["name"] = f"ball_{sys.argv[2]}"
cfg["train"]["epochs"] = 14
cfg["train"]["seed"] = int(sys.argv[3])
open(sys.argv[1], "w").write(yaml.safe_dump(cfg, sort_keys=False))
YAMLEOF
  TIMEOUT=21600 launch "train_$name" "$PY" "$G/scripts/pretrain.py" \
    --config "$cfg" --data "$data" --output-dir "$out" \
    --readout spatial_map --gravity-scale 2.309 --device cuda
done
wait
for spec in "single_s43 $DATA/ball_single" "narrowfx_s43 $DATA/ball_narrow"; do
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
for name in single_s43 narrowfx_s43; do
  dir=$S/$name
  [ -f "$dir/latents.pt" ] || continue
  [ -f "$dir/probes/probe_eval.pt" ] || launch "probes_$name" "$PY" "$G/scripts/train_probes.py" \
    --cache "$dir/latents.pt" --output-dir "$dir/probes" --device cuda \
    --history 8 --horizon 6 --stride 1 --max-episodes 6000
done
wait
for name in single_s43 narrowfx_s43; do
  for variant in pretrained jepa factual penalty counterfactual; do
    post "$name" "mp_$variant" "$variant" --seed 43
  done
done
wait
for name in single_s43 narrowfx_s43; do
  for variant in pretrained jepa factual penalty counterfactual; do
    evaluate "$name" "mp_$variant"
  done
done
wait
say "PHASE seeds done"

# ---------------------------------------------------------- 2. frontier on single
say "PHASE frontier: tolerance against penalty weight, on ball_single"
for scale in 0.8 0.6 0.45; do
  post single_s42 "pq_eps$scale" counterfactual --seed 42 --epsilon-scale "$scale"
done
for lam in 10.0 100.0 300.0 1000.0; do
  post single_s42 "pq_lam$lam" penalty --seed 42 --penalty-lambda "$lam"
done
wait
for scale in 0.8 0.6 0.45; do evaluate single_s42 "pq_eps$scale"; done
for lam in 10.0 100.0 300.0 1000.0; do evaluate single_s42 "pq_lam$lam"; done
wait
say "PHASE frontier done"
: > "$W/.done-queue1"
say "=== queue1 done ==="
