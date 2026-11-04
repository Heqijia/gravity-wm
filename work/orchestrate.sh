#!/usr/bin/env bash
# Unattended experiment suite for the gravity-constrained post-training study.
#
# Runs for a day or two without supervision, two GPU jobs at a time.  Every
# step is idempotent: it checks for its own output first, so the script can be
# killed and restarted and will pick up where it stopped.  Nothing is deleted;
# a stage is redone only if its output directory is removed by hand.
#
# Phases, in order of what the conclusions depend on:
#
#   main      the five variants on two seeds -- the headline comparison
#   forms     Definition 1 against Proposition 1's position form -- the
#             comparison that decides whether the objective is the right one
#   frontier  the tolerance sweep against a penalty-weight sweep, which is the
#             comparison that actually separates the two methods: the switching
#             method targets a tolerance and certifies it, the penalty trades
#             at a fixed rate and certifies nothing
#   baselines train the Semigroup-JEPA variants to compare against: the
#             one-step LeWM objective and the SSM predictor
#   basepost  post-train and evaluate on those
#   clspost   the same on the published [CLS] read-out, where the constraint
#             has no margin -- the negative result, run rather than asserted
#   ablate    gamma, the tolerance source, the RESTORE direction, hysteresis
#   strides   the strided generalisation of Lemma 1, m = 1..4
#   narrow    a near-single training gravity, the regime Remark 11 says the
#             separation needs -- on N(4, 0.5^2) the pretrained predictor
#             already extrapolates and there is nothing to separate
#   report    tables and figures over everything that finished
#
# usage: setsid nohup bash orchestrate.sh [phase ...] >> orchestrate.log 2>&1 &
set -u

ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
W=$ROOT/work
R=/fs/scratch/PAS3151/qijia387/gwm/runs
DATA=/fs/scratch/PAS3151/qijia387/gwm/data/ball_v1
DATA2=/fs/scratch/PAS3151/qijia387/gwm/data/ball_v2
JOBID=${JOBID:-8093487}
STEPS=${STEPS:-4000}
LOGS=$W/logs
mkdir -p "$LOGS"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------- job pool

GPU_NEXT=0
# Every job gets a wall-clock cap.  Without one a single hung job blocks the
# pool on `wait -n` for as long as the allocation lasts, and an unattended run
# would quietly stop making progress.  TIMEOUT is in seconds and callers raise
# it for the pretraining stages.
TIMEOUT=${TIMEOUT:-5400}
launch() { # launch <logname> <command...>
  local name=$1; shift
  local gpu=$GPU_NEXT
  GPU_NEXT=$(( (GPU_NEXT + 1) % 2 ))
  local limit=$TIMEOUT
  (
    timeout --signal=TERM --kill-after=60 "$limit" \
      srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=5 --gres=gpu:2 \
        env CUDA_VISIBLE_DEVICES=$gpu PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
        "$@" > "$LOGS/$name.log" 2>&1
    code=$?
    [ "$code" = 124 ] && printf '%s TIMED OUT after %ss\n' "$name" "$limit" >> "$LOGS/_exits.txt"
    printf '%s exit=%s\n' "$name" "$code" >> "$LOGS/_exits.txt"
  ) &
  say "  launched $name on gpu $gpu (cap ${limit}s)"
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
}

# ---------------------------------------------------------------- stages

prepare_model() { # prepare_model <name> <checkpoint-source-dir> [data]
  local name=$1 src=$2 data=${3:-$DATA}
  local dir=$R/suite/$name
  mkdir -p "$dir"
  if [ ! -f "$dir/checkpoint.pt" ]; then
    cp "$src/best.pt" "$dir/checkpoint.pt" 2>/dev/null ||
      cp "$src/checkpoint.pt" "$dir/checkpoint.pt"
    for f in normalization_stats.json train_val_split.json readout.json; do
      [ -f "$src/$f" ] && cp "$src/$f" "$dir/$f"
    done
    [ -f "$dir/readout.json" ] || echo '{"readout": "cls"}' > "$dir/readout.json"
    say "  staged $name from $src"
  fi
  if [ ! -f "$dir/latents.pt" ]; then
    launch "cache_$name" "$PY" "$G/scripts/build_cache.py" \
      --checkpoint "$dir/checkpoint.pt" --data "$data" \
      --output "$dir/latents.pt" --device cuda
  fi
}

prepare_probes() { # prepare_probes <name>
  local name=$1 dir=$R/suite/$1
  [ -f "$dir/probes/probe_eval.pt" ] && return
  [ -f "$dir/latents.pt" ] || { say "  !! $name has no cache, skipping probes"; return; }
  launch "probes_$name" "$PY" "$G/scripts/train_probes.py" \
    --cache "$dir/latents.pt" --output-dir "$dir/probes" --device cuda \
    --history 8 --horizon 6 --stride 1 --max-episodes 6000
}

post() { # post <model> <tag> <variant> [extra args...]
  local model=$1 tag=$2 variant=$3; shift 3
  local base=$R/suite/$model
  local out=$R/suite/$model/post_$tag
  [ -f "$out/posttrain_report.json" ] && return
  [ -f "$base/probes/probe_eval.pt" ] || { say "  !! $model has no probes, skipping $tag"; return; }
  launch "post_${model}_${tag}" "$PY" "$G/scripts/posttrain.py" \
    --checkpoint "$base/checkpoint.pt" --cache "$base/latents.pt" \
    --probe-dir "$base/probes" --output-dir "$out" --variant "$variant" \
    --steps "$STEPS" --lr 1e-4 --history "${HISTORY:-8}" --horizon 6 \
    --epsilon-source factual_rollout --gravity-low 1.0 --gravity-high 9.0 \
    --seed 42 --eval-every 200 --device cuda "$@"
}

evaluate() { # evaluate <model> <tag> [stride]
  local model=$1 tag=$2 stride=${3:-1}
  local base=$R/suite/$model out=$R/suite/$model/post_$tag
  [ -f "$out/evaluation.json" ] && return
  [ -f "$out/checkpoint.pt" ] || return
  launch "eval_${model}_${tag}" "$PY" "$G/scripts/evaluate.py" \
    --checkpoint "$out/checkpoint.pt" --cache "$base/latents.pt" \
    --probe-dir "$base/probes" --output "$out/evaluation.json" \
    --posttrain-report "$out/posttrain_report.json" \
    --history "${HISTORY:-8}" --horizon 6 --stride "$stride" \
    --sweep-min 1.0 --sweep-max 9.0 --label "${model}__${tag}" --device cuda
}

label_evaluations() { # stamp each evaluation with its variant, for the report
  $PY - "$R/suite" <<'PYEOF'
import json, sys
from pathlib import Path
for report in Path(sys.argv[1]).glob("*/post_*/posttrain_report.json"):
    evaluation = report.parent / "evaluation.json"
    if not evaluation.is_file():
        continue
    payload = json.loads(evaluation.read_text())
    variant = json.loads(report.read_text())["variant"]
    if payload.get("variant") != variant:
        payload["variant"] = variant
        evaluation.write_text(json.dumps(payload, indent=2, sort_keys=True))
PYEOF
}

VARIANTS="pretrained jepa factual penalty counterfactual"

# ---------------------------------------------------------------- phases

phase_main() {
  say "PHASE main: five variants on two seeds, stride 1"
  prepare_model spatial_s42 "$R/ball_spatial_s42"
  prepare_model spatial_s43 "$R/ball_spatial_s43"
  wait
  prepare_probes spatial_s42
  prepare_probes spatial_s43
  wait
  for model in spatial_s42 spatial_s43; do
    for variant in $VARIANTS; do post "$model" "s1_$variant" "$variant" --stride 1; done
  done
  wait
  for model in spatial_s42 spatial_s43; do
    for variant in $VARIANTS; do evaluate "$model" "s1_$variant" 1; done
  done
  wait
  label_evaluations
  say "PHASE main done"
}

phase_mainpos() {
  # The headline comparison, with the constraint in the form that works.
  # Measured on seed 42: at the same tolerance the position form takes the
  # counterfactual position error from 5.93 px to 4.45 and the JEPA loss from
  # 21.0 to 19.1 -- better on both axes than continued JEPA training -- while
  # the acceleration form of Definition 1 is indistinguishable from doing
  # nothing.  See phase_forms and docs/design.md for why.
  say "PHASE mainpos: the five variants with Proposition 1's position form"
  for model in spatial_s42 spatial_s43; do
    for variant in $VARIANTS; do
      post "$model" "mp_$variant" "$variant" --stride 1 --constraint-form position
    done
  done
  wait
  for model in spatial_s42 spatial_s43; do
    for variant in $VARIANTS; do evaluate "$model" "mp_$variant" 1; done
  done
  wait
  label_evaluations
  say "PHASE mainpos done"
}

phase_allpos() {
  # Every other base model in Proposition 1's form.  The baselines were run
  # only in the acceleration form of Definition 1, which the forms phase shows
  # is the wrong objective to descend -- on the SSM predictor it drove the
  # JEPA loss to infinity and on the one-step LeWM objective it left the
  # counterfactual error worse than the pretrained model.  Those numbers say
  # nothing about the method, only about the form, so each base model is rerun
  # in the form that works.  Checkpoints, caches and probes already exist, so
  # this is post-training and evaluation only.
  say "PHASE allpos: position form on every other base model"
  for model in lewm1step_s42 ssm_s42 cls_s42 narrow_s42 v2_s42; do
    local dir=$R/suite/$model
    [ -f "$dir/probes/probe_eval.pt" ] || { say "  !! $model has no probes, skipping"; continue; }
    # bash does not scope `VAR=v func` to the call reliably, so set it plainly.
    HISTORY=8
    [ "$model" = v2_s42 ] && HISTORY=6
    for variant in $VARIANTS; do
      post "$model" "mp_$variant" "$variant" --stride 1 --constraint-form position
    done
  done
  wait
  for model in lewm1step_s42 ssm_s42 cls_s42 narrow_s42 v2_s42; do
    HISTORY=8
    [ "$model" = v2_s42 ] && HISTORY=6
    for variant in $VARIANTS; do evaluate "$model" "mp_$variant" 1; done
  done
  wait
  unset HISTORY
  label_evaluations
  say "PHASE allpos done"
}

phase_posfrontier() {
  say "PHASE posfrontier: the tolerance sweep, position form"
  for scale in 0.8 0.6 0.45 0.3; do
    post spatial_s42 "pq_eps$scale" counterfactual --stride 1 \
      --constraint-form position --epsilon-scale "$scale"
  done
  for lam in 1.0 10.0 100.0; do
    post spatial_s42 "pq_lam$lam" penalty --stride 1 \
      --constraint-form position --penalty-lambda "$lam"
  done
  wait
  for scale in 0.8 0.6 0.45 0.3; do evaluate spatial_s42 "pq_eps$scale" 1; done
  for lam in 1.0 10.0 100.0; do evaluate spatial_s42 "pq_lam$lam" 1; done
  wait
  label_evaluations
  say "PHASE posfrontier done"
}

phase_forms() {
  # Definition 1 constrains the squared second differences of the decoded
  # positions; Proposition 1's equivalent-zero-set form constrains the
  # distance to the counterfactual trajectory itself.  Measured, residuals
  # costing the same in position differ by >100x in the squared form, so its
  # gradient is dominated by the unsystematic component -- which is cheap to
  # remove and nearly irrelevant to position accuracy.  This is the comparison
  # that decides whether the method's objective is the right one.
  say "PHASE forms: Definition 1 against Proposition 1's position form"
  for variant in counterfactual factual penalty; do
    post spatial_s42 "pf_$variant" "$variant" --stride 1 --constraint-form position
  done
  for scale in 0.6 0.3; do
    post spatial_s42 "pf_eps$scale" counterfactual --stride 1 \
      --constraint-form position --epsilon-scale "$scale"
  done
  wait
  for variant in counterfactual factual penalty; do evaluate spatial_s42 "pf_$variant" 1; done
  for scale in 0.6 0.3; do evaluate spatial_s42 "pf_eps$scale" 1; done
  wait
  label_evaluations
  say "PHASE forms done"
}

phase_frontier() {
  say "PHASE frontier: tolerance sweep against penalty-weight sweep"
  # The usable window for the tolerance is narrow and sits just below 1.0:
  # below the predictor's own factual violation the problem is infeasible and
  # RESTORE destroys the model, above the counterfactual level the constraint
  # never engages.  Sweep finely inside it, and keep two coarse points outside
  # to show both failure modes.
  for scale in 1.0 0.9 0.8 0.7 0.6 0.4; do
    post spatial_s42 "fr_eps$scale" counterfactual --stride 1 --epsilon-scale "$scale"
  done
  for lam in 0.1 0.3 1.0 3.0 10.0; do
    post spatial_s42 "fr_lam$lam" penalty --stride 1 --penalty-lambda "$lam"
  done
  wait
  for scale in 1.0 0.9 0.8 0.7 0.6 0.4; do evaluate spatial_s42 "fr_eps$scale" 1; done
  for lam in 0.1 0.3 1.0 3.0 10.0; do evaluate spatial_s42 "fr_lam$lam" 1; done
  wait
  label_evaluations
  say "PHASE frontier done"
}

phase_baselines() {
  say "PHASE baselines: train the Semigroup-JEPA variants to compare against"
  for spec in "lewm1step:prediction" "ssm:ssm"; do
    local name=${spec%%:*} kind=${spec##*:}
    local out=$R/ball_${name}_s42
    [ -f "$out/best.pt" ] && { say "  skip $name"; continue; }
    $PY - "$kind" "$W/ball_${name}_s42.yaml" <<'PYEOF'
import sys, yaml
kind, path = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open("/users/PAS3151/qijia387/Phys-WM/gravity-wm/configs/ball_sg_jepa_gru.yaml"))
cfg["train"]["epochs"] = 14          # past the validation optimum seen at 10-11
if kind == "prediction":
    # Original LeWM: one-step teacher forcing instead of the discounted rollout
    cfg["name"] = "ball_lewm1step"
    cfg["objective"]["prediction_weight"] = 1.0
    cfg["objective"]["rollout_weight"] = 0.0
else:
    cfg["name"] = "ball_ssm"
    cfg["model"]["predictor_kind"] = "ssm"
open(path, "w").write(yaml.safe_dump(cfg, sort_keys=False))
PYEOF
    TIMEOUT=21600 launch "train_$name" "$PY" "$G/scripts/pretrain.py" \
      --config "$W/ball_${name}_s42.yaml" --data "$DATA" \
      --output-dir "$out" --readout spatial_map --device cuda
  done
  wait
  say "PHASE baselines done"
}

phase_basepost() {
  say "PHASE basepost: post-train on the Semigroup-JEPA variants"
  for name in lewm1step ssm; do
    [ -f "$R/ball_${name}_s42/best.pt" ] || { say "  !! no $name checkpoint"; continue; }
    prepare_model "${name}_s42" "$R/ball_${name}_s42"
  done
  wait
  for name in lewm1step ssm; do prepare_probes "${name}_s42"; done
  wait
  for name in lewm1step ssm; do
    for variant in $VARIANTS; do post "${name}_s42" "s1_$variant" "$variant" --stride 1; done
  done
  wait
  for name in lewm1step ssm; do
    for variant in $VARIANTS; do evaluate "${name}_s42" "s1_$variant" 1; done
  done
  wait
  label_evaluations
  say "PHASE basepost done"
}

phase_clspost() {
  say "PHASE clspost: the published [CLS] read-out, where the margin is 0.5"
  prepare_model cls_s42 "$R/ball_s42"
  wait
  prepare_probes cls_s42
  wait
  for variant in $VARIANTS; do post cls_s42 "s1_$variant" "$variant" --stride 1; done
  wait
  for variant in $VARIANTS; do evaluate cls_s42 "s1_$variant" 1; done
  wait
  label_evaluations
  say "PHASE clspost done"
}

phase_ablate() {
  say "PHASE ablate: the method's own knobs"
  post spatial_s42 ab_gate0.5     counterfactual --stride 1 --constraint-form position --gate 0.5
  post spatial_s42 ab_eps_real    counterfactual --stride 1 --constraint-form position --epsilon-source real_latents
  post spatial_s42 ab_restore_eps counterfactual --stride 1 --constraint-form position --restore-direction eps
  post spatial_s42 ab_restore_nrm counterfactual --stride 1 --constraint-form position --restore-direction normalized
  post spatial_s42 ab_nohyst      counterfactual --stride 1 --constraint-form position --no-hysteresis
  post spatial_s42 ab_noproject   counterfactual --stride 1 --constraint-form position --no-project
  post spatial_s42 ab_noframe     counterfactual --stride 1 --constraint-form position --no-in-frame-filter
  wait
  for tag in ab_gate0.5 ab_eps_real ab_restore_eps ab_restore_nrm ab_nohyst ab_noproject ab_noframe; do
    evaluate spatial_s42 "$tag" 1
  done
  wait
  label_evaluations
  say "PHASE ablate done"
}

phase_strides() {
  say "PHASE strides: the strided generalisation of Lemma 1"
  for m in 2 3 4; do post spatial_s42 "m${m}_counterfactual" counterfactual --stride "$m" --constraint-form position; done
  for m in 2 3 4; do post spatial_s42 "m${m}_pretrained" pretrained --stride "$m"; done
  wait
  for m in 2 3 4; do
    evaluate spatial_s42 "m${m}_counterfactual" "$m"
    evaluate spatial_s42 "m${m}_pretrained" "$m"
  done
  wait
  label_evaluations
  say "PHASE strides done"
}

phase_v2() {
  say "PHASE v2: replicate on the shorter, coarser-sampled dataset"
  [ -d "$DATA2" ] || { say "  !! $DATA2 missing"; return; }
  local out=$R/ball_v2_s42
  if [ ! -f "$out/best.pt" ]; then
    $PY - "$W/ball_v2_s42.yaml" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open("/users/PAS3151/qijia387/Phys-WM/gravity-wm/configs/ball_sg_jepa_gru.yaml"))
cfg["name"] = "ball_v2"
cfg["train"]["epochs"] = 14
cfg["model"]["history_size"] = 6     # the v2 episodes are 12 frames: 6 + 6
cfg["objective"]["rollout_horizon"] = 6
open(sys.argv[1], "w").write(yaml.safe_dump(cfg, sort_keys=False))
PYEOF
    TIMEOUT=21600 launch "train_v2" "$PY" "$G/scripts/pretrain.py" \
      --config "$W/ball_v2_s42.yaml" --data "$DATA2" \
      --output-dir "$out" --readout spatial_map --device cuda
    wait
  fi
  [ -f "$out/best.pt" ] || { say "  !! v2 training produced no checkpoint"; return; }
  prepare_model v2_s42 "$out" "$DATA2"
  wait
  # v2 episodes are 12 frames, so the constraint window must be 6 + 6
  local dir=$R/suite/v2_s42
  [ -f "$dir/probes/probe_eval.pt" ] || launch "probes_v2_s42" "$PY" "$G/scripts/train_probes.py" \
    --cache "$dir/latents.pt" --output-dir "$dir/probes" --device cuda \
    --history 6 --horizon 6 --stride 1 --max-episodes 6000
  wait
  for variant in $VARIANTS; do
    local o=$dir/post_s1_$variant
    [ -f "$o/posttrain_report.json" ] && continue
    launch "post_v2_${variant}" "$PY" "$G/scripts/posttrain.py" \
      --checkpoint "$dir/checkpoint.pt" --cache "$dir/latents.pt" \
      --probe-dir "$dir/probes" --output-dir "$o" --variant "$variant" \
      --steps "$STEPS" --lr 1e-4 --history 6 --horizon 6 --stride 1 \
      --epsilon-source factual_rollout --gravity-low 1.0 --gravity-high 9.0 \
      --seed 42 --eval-every 200 --device cuda
  done
  wait
  for variant in $VARIANTS; do
    local o=$dir/post_s1_$variant
    [ -f "$o/evaluation.json" ] && continue
    [ -f "$o/checkpoint.pt" ] || continue
    launch "eval_v2_${variant}" "$PY" "$G/scripts/evaluate.py" \
      --checkpoint "$o/checkpoint.pt" --cache "$dir/latents.pt" \
      --probe-dir "$dir/probes" --output "$o/evaluation.json" \
      --posttrain-report "$o/posttrain_report.json" \
      --history 6 --horizon 6 --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
      --label "v2__$variant" --device cuda
  done
  wait
  label_evaluations
  say "PHASE v2 done"
}

phase_narrow() {
  # The regime the method is actually for.  Theorem 2 separates the full
  # method from the ablations only when the pretrained predictor responds
  # *incorrectly* to counterfactual gravities -- Remark 11 says so outright.
  # Measured on the paper's own training distribution, N(4, 0.5^2), it does
  # not: the predictor's counterfactual violation is only 20% above its
  # factual one (25.1 against 20.9), where a gravity-blind predictor would sit
  # at 21.9.  Semigroup-JEPA has learned to extrapolate gravity, so there is
  # little for the constraint to fix.
  #
  # Training on an essentially single gravity removes that head start without
  # changing anything else, and is the setting in which the constraint is the
  # only signal the predictor ever gets about gravities it has not seen.
  say "PHASE narrow: a near-single training gravity, where theta_0 cannot extrapolate"
  local data=/fs/scratch/PAS3151/qijia387/gwm/data/ball_narrow
  if [ ! -f "$data/generation_manifest.json" ]; then
    rm -rf "$data"
    say "  generating the narrow-gravity dataset"
    srun --jobid="$JOBID" --overlap -n1 --cpus-per-task=10 \
      "$PY" "$G/scripts/gen_data.py" --output "$data" --shape ball \
      --train-episodes 16000 --heldout-train-gravity-episodes 800 \
      --test-episodes-per-gravity 32 --train-std 0.05 --workers 9 \
      > "$LOGS/gen_narrow.log" 2>&1
  fi
  [ -f "$data/generation_manifest.json" ] || { say "  !! narrow dataset missing"; return; }
  local out=$R/ball_narrow_s42
  if [ ! -f "$out/best.pt" ]; then
    $PY - "$W/ball_narrow_s42.yaml" <<'YAMLEOF'
import sys, yaml
cfg = yaml.safe_load(open("/users/PAS3151/qijia387/Phys-WM/gravity-wm/configs/ball_sg_jepa_gru.yaml"))
cfg["name"] = "ball_narrow"
cfg["train"]["epochs"] = 14
open(sys.argv[1], "w").write(yaml.safe_dump(cfg, sort_keys=False))
YAMLEOF
    TIMEOUT=21600 launch "train_narrow" "$PY" "$G/scripts/pretrain.py" \
      --config "$W/ball_narrow_s42.yaml" --data "$data" \
      --output-dir "$out" --readout spatial_map --device cuda
    wait
  fi
  [ -f "$out/best.pt" ] || { say "  !! narrow training produced no checkpoint"; return; }
  prepare_model narrow_s42 "$out" "$data"
  wait
  prepare_probes narrow_s42
  wait
  for variant in $VARIANTS; do post narrow_s42 "s1_$variant" "$variant" --stride 1; done
  wait
  for variant in $VARIANTS; do evaluate narrow_s42 "s1_$variant" 1; done
  wait
  label_evaluations
  say "PHASE narrow done"
}

phase_report() {
  say "PHASE report"
  label_evaluations
  mkdir -p "$R/suite/report"
  for model in "$R"/suite/*/; do
    local name
    name=$(basename "$model")
    [ "$name" = "report" ] && continue
    local files
    files=$(ls "$model"/post_*/evaluation.json 2>/dev/null)
    [ -z "$files" ] && continue
    # shellcheck disable=SC2086
    for group in mp_ s1_ pq_ pf_ fr_ ab_ m2_ m3_ m4_; do
      # shellcheck disable=SC2086
      $PY "$G/scripts/report.py" --evaluations $files --group "$group" \
        --output "$R/suite/report/${name}_${group%_}.json" \
        > "$R/suite/report/${name}_${group%_}.txt" 2>&1
      [ -s "$R/suite/report/${name}_${group%_}.txt" ] || rm -f "$R/suite/report/${name}_${group%_}".*
    done
    say "  wrote report for $name"
  done
  local all
  all=$(ls "$R"/suite/*/post_*/evaluation.json 2>/dev/null)
  if [ -n "$all" ]; then
    # shellcheck disable=SC2086
    $PY "$G/scripts/figures.py" --evaluations $all \
      --posttrain-report "$R/suite/spatial_s42/post_s1_counterfactual/posttrain_report.json" \
      --supervise-log "$W/supervise.tsv" \
      --output "$R/suite/report/figures.png" >> "$LOGS/report.log" 2>&1
  fi
  say "PHASE report done"
}

# ---------------------------------------------------------------- main

PHASES=${*:-"mainpos posfrontier narrow basepost clspost ablate strides v2 frontier report"}
say "=== suite starting: $PHASES ==="
for phase in $PHASES; do
  if ! declare -F "phase_$phase" > /dev/null; then
    say "!! unknown phase $phase"
    continue
  fi
  "phase_$phase"
  phase_report
done
say "=== suite finished ==="
