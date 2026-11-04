#!/usr/bin/env bash
# Queue 3: the studies that are currently measured only where they are weakest.
#
# The ablations and the strided generalisation were run on ball_v1, where the
# pretrained predictor already extrapolates gravity (far-band response 0.706)
# so the constraint is barely active and every knob looked irrelevant -- gamma,
# hysteresis, the projection and the in-frame filter all landed within 0.7 px
# of default.  That is a real finding about that regime and a useless one about
# the method.  On ball_single the pretrained response is -0.430, the constraint
# does all the work, and the same knobs should separate if they matter at all.
#
# The Semigroup-JEPA predictor variants are also only trained on ball_v1, so
# "the method helps sg-JEPA" currently rests on one architecture in the regime
# that matters.  Training the one-step LeWM objective on ball_single fixes that.
# The SSM is left out deliberately: it starts at C_0/eps = 87 and spends every
# step restoring feasibility, which is a documented limitation, not a result.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
W=$ROOT/work
R=/fs/scratch/PAS3151/qijia387/gwm/runs
S=$R/suite
DATA=/fs/scratch/PAS3151/qijia387/gwm/data/ball_single
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
    --steps 4000 --lr 1e-4 --history 8 --horizon 6 \
    --constraint-form position --epsilon-source factual_rollout \
    --gravity-low 1.0 --gravity-high 9.0 --seed 42 --eval-every 200 --device cuda "$@"
}
evaluate() { # evaluate <model> <tag> [stride]
  local model=$1 tag=$2 stride=${3:-1} dir=$S/$1 out=$S/$1/post_$2
  [ -f "$out/checkpoint.pt" ] || return
  [ -f "$out/evaluation.json" ] && return
  launch "eval_${model}_${tag}" "$PY" "$G/scripts/evaluate.py" \
    --checkpoint "$out/checkpoint.pt" --cache "$dir/latents.pt" \
    --probe-dir "$dir/probes" --output "$out/evaluation.json" \
    --posttrain-report "$out/posttrain_report.json" \
    --history 8 --horizon 6 --stride "$stride" --sweep-min 1.0 --sweep-max 9.0 \
    --label "${model}__${tag}" --device cuda
}

say "PHASE ab_single: the method's knobs where the constraint actually works"
post single_s42 ab_gate0.5     counterfactual --stride 1 --gate 0.5
post single_s42 ab_nohyst      counterfactual --stride 1 --no-hysteresis
post single_s42 ab_noproject   counterfactual --stride 1 --no-project
post single_s42 ab_noframe     counterfactual --stride 1 --no-in-frame-filter
post single_s42 ab_restore_eps counterfactual --stride 1 --restore-direction eps
post single_s42 ab_restore_nrm counterfactual --stride 1 --restore-direction normalized
post single_s42 ab_eps_real    counterfactual --stride 1 --epsilon-source real_latents
post single_s42 ab_accelform   counterfactual --stride 1 --constraint-form acceleration
wait
for tag in ab_gate0.5 ab_nohyst ab_noproject ab_noframe ab_restore_eps ab_restore_nrm ab_eps_real ab_accelform; do
  evaluate single_s42 "$tag"
done
wait
say "PHASE ab_single done"

say "PHASE strides_single: the strided generalisation of Lemma 1"
for m in 2 3 4; do
  post single_s42 "m${m}_counterfactual" counterfactual --stride "$m"
done
wait
for m in 2 3 4; do evaluate single_s42 "m${m}_counterfactual" "$m"; done
wait
say "PHASE strides_single done"

say "PHASE lewm_single: the one-step LeWM objective on ball_single"
name=lewm_single_s42
out=$R/ball_$name
if [ ! -f "$out/best.pt" ]; then
  cfg=$W/ball_$name.yaml
  "$PY" - "$cfg" <<'YAMLEOF'
import sys, yaml
cfg = yaml.safe_load(open("/users/PAS3151/qijia387/Phys-WM/gravity-wm/configs/ball_sg_jepa_gru.yaml"))
cfg["name"] = "ball_lewm_single"
cfg["train"]["epochs"] = 14
# the one-step LeWM objective: predict the next latent, no discounted rollout
cfg["objective"]["prediction_weight"] = 1.0
cfg["objective"]["rollout_weight"] = 0.0
open(sys.argv[1], "w").write(yaml.safe_dump(cfg, sort_keys=False))
YAMLEOF
  TIMEOUT=21600 launch "train_$name" "$PY" "$G/scripts/pretrain.py" \
    --config "$cfg" --data "$DATA" --output-dir "$out" \
    --readout spatial_map --gravity-scale 2.309 --device cuda
  wait
fi
if [ -f "$out/best.pt" ]; then
  dir=$S/$name
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
  # the step size comes from the model's own gradient scale, not from a default
  srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=4 --gres=gpu:2 \
    env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 "$PY" \
    "$G/scripts/measure_gradient_scale.py" --checkpoint "$dir/checkpoint.pt" \
    --cache "$dir/latents.pt" --device cuda > "$LOGS/gradscale_$name.log" 2>&1
  lr=$("$PY" -c "
import json,sys
try:
    for line in open('$LOGS/gradscale_$name.log'):
        pass
    text=open('$LOGS/gradscale_$name.log').read()
    payload=json.loads(text[text.index('{'):text.rindex('}')+1])
    print(f\"{payload['lr_for_relative_step']:.3g}\")
except Exception:
    print('1e-4')
")
  say "  $name learning rate from its own gradient scale: $lr"
  for variant in pretrained jepa factual penalty counterfactual; do
    o=$S/$name/post_mp_$variant
    [ -f "$o/posttrain_report.json" ] && continue
    launch "post_${name}_mp_$variant" "$PY" "$G/scripts/posttrain.py" \
      --checkpoint "$S/$name/checkpoint.pt" --cache "$S/$name/latents.pt" \
      --probe-dir "$S/$name/probes" --output-dir "$o" --variant "$variant" \
      --steps 4000 --lr "$lr" --history 8 --horizon 6 --stride 1 \
      --constraint-form position --epsilon-source factual_rollout \
      --gravity-low 1.0 --gravity-high 9.0 --seed 42 --eval-every 200 --device cuda
  done
  wait
  for variant in pretrained jepa factual penalty counterfactual; do
    evaluate "$name" "mp_$variant"
  done
  wait
fi
say "PHASE lewm_single done"
: > "$W/.done-queue3"
say "=== queue3 done ==="
