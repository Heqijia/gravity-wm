#!/usr/bin/env bash
# Can a shorter context buy a longer horizon on the existing 16-frame data?
#
# history 4 + horizon 12 fits in 16 frames, so h = 12 would need no new data
# and no new pretraining.  Two things have to hold first.
#
#   1. The predictor was built with num_frames = 8.  It accepts a 4-frame
#      window without error, but if its accuracy drops then an h = 12 curve
#      measured that way confounds "longer horizon" with "shorter context".
#      The control is history 4 vs history 8 at the *same* horizon 6.
#   2. The counterfactual must stay in frame for 12 steps.  At g' = 9 the
#      object falls 16.6 world units in 1.92 s and the view is 13, so the
#      in-frame filter may reject the far band entirely -- the failure that
#      made h = 12 unmeasurable on the bouncing scene.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
M=$R/single_s42
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
one() { srun --jobid=8093487 --overlap -n1 --cpus-per-task=4 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 "$@"; }

run() { # run <tag> <history> <horizon> <variant-dir>
  local tag=$1 hist=$2 hor=$3 vd=$4
  local out=$M/$vd
  [ -f "$out/eval_${tag}.json" ] && { say "  $vd/$tag 已有"; return; }
  one "$PY" "$G/scripts/evaluate.py" --checkpoint "$out/checkpoint.pt" --cache "$M/latents.pt" \
    --probe-dir "$M/probes" --output "$out/eval_${tag}.json" \
    --posttrain-report "$out/posttrain_report.json" \
    --history "$hist" --horizon "$hor" --stride 1 --sweep-min 1.0 --sweep-max 9.0 \
    --label "single_s42__${vd}__${tag}" --device cuda \
    > $ROOT/work/logs/eval_${vd}_${tag}.log 2>&1
  say "  $vd/$tag exit=$?"
}
say "=== history/horizon 可行性探针 ==="
for vd in post_mp_pretrained post_mp_counterfactual; do
  run h8h6  8 6  "$vd"     # 基准
  run h4h6  4 6  "$vd"     # 只变上下文长度 —— 这是对照
  run h4h12 4 12 "$vd"     # 目标设置
  run h2h14 2 14 "$vd"     # 更极端,看趋势
done
: > $ROOT/work/.done-hist_probe
say "=== hist_probe done ==="
