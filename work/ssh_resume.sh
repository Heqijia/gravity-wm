#!/usr/bin/env bash
# Drive the existing allocation over ssh instead of srun.
#
# The controller rebooted at 09:02 and every srun into jobs 8093487 and
# 8236469 now fails with a protocol authentication error.  The nodes
# themselves are fine: ssh works, torch sees the cards, scratch is readable.
# Only SLURM's step-launch path is broken, so all six cards sat idle for three
# hours while the work waited on a queue it did not need to be in.
#
# The nodes are draining@ for maintenance, so anything running may be killed
# when they reboot.  Every stage is guarded by its own output file, so
# relaunching resumes rather than repeats.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
PY=$ROOT/.venv-wm/bin/python
G=$ROOT/gravity-wm
S=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

rsh() { # rsh <node> <gpu> <logfile> <command string>
  local node=$1 gpu=$2 log=$3; shift 3
  ssh -q -o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=60 "$node" \
    "CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 $*" > "$log" 2>&1
}

sf() { # single_far seed 43 -- counterfactual first, the only arm that concludes
  local node=$1 gpu=$2 seed=$3 pfx=$4
  local M=$S/single_far_sig009_s42
  for v in counterfactual factual penalty; do
    local out=$M/post_${pfx}_$v
    if [ ! -f "$out/posttrain_report.json" ]; then
      rsh "$node" "$gpu" "$ROOT/work/logs/ssh_post_${pfx}_$v.log" \
        "$PY $G/scripts/posttrain.py --checkpoint $M/checkpoint.pt --cache $M/latents.pt \
         --probe-dir $M/probes --output-dir $out --variant $v --steps 12000 --lr 1e-4 \
         --history 8 --horizon 6 --stride 1 --constraint-form position \
         --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 --seed $seed \
         --eval-every 600 --device cuda"
      say "  $pfx post $v exit=$?"
    fi
    [ -f "$out/checkpoint.pt" ] || continue
    for H in 6 12 18; do
      [ -f "$out/eval_h$H.json" ] && continue
      rsh "$node" "$gpu" "$ROOT/work/logs/ssh_eval_${pfx}_${v}_h$H.log" \
        "$PY $G/scripts/evaluate.py --checkpoint $out/checkpoint.pt --cache $M/latents.pt \
         --probe-dir $M/probes --output $out/eval_h$H.json \
         --posttrain-report $out/posttrain_report.json --history 8 --horizon $H --stride 1 \
         --sweep-min 1.0 --sweep-max 9.0 --label single_far__${pfx}_${v}__h$H --device cuda"
    done
    say "  $pfx eval $v 完成"
  done
}

bl() { # bounce on the 21-epoch foundation
  local node=$1 gpu=$2
  local M=$S/bounce_long_ep021
  for v in counterfactual factual penalty; do
    local out=$M/post_mp_$v
    if [ ! -f "$out/posttrain_report.json" ]; then
      rsh "$node" "$gpu" "$ROOT/work/logs/ssh_post_bl21_$v.log" \
        "$PY $G/scripts/posttrain.py --checkpoint $M/checkpoint.pt --cache $M/latents.pt \
         --probe-dir $M/probes --output-dir $out --variant $v --steps 36000 --lr 1e-4 \
         --history 8 --horizon 6 --stride 1 --constraint-form position \
         --epsilon-source factual_rollout --gravity-low 1 --gravity-high 9 \
         --floor-resting-z 1.60 --seed 42 --eval-every 1200 --device cuda"
      say "  bl21 post $v exit=$?"
    fi
    [ -f "$out/checkpoint.pt" ] || continue
    if [ ! -f "$out/evaluation.json" ]; then
      rsh "$node" "$gpu" "$ROOT/work/logs/ssh_eval_bl21_$v.log" \
        "$PY $G/scripts/evaluate.py --checkpoint $out/checkpoint.pt --cache $M/latents.pt \
         --probe-dir $M/probes --output $out/evaluation.json \
         --posttrain-report $out/posttrain_report.json --history 8 --horizon 6 --stride 1 \
         --sweep-min 1.0 --sweep-max 9.0 --floor-resting-z 1.60 \
         --label bounce_long21__$v --device cuda"
      say "  bl21 eval $v exit=$?"
    fi
  done
}

say "=== ssh 直连恢复:6 卡全开 ==="
sf a0012 0 43 s43 &        # a0012 卡0: single_far seed 43
bl a0012 1 &               # a0012 卡1: bounce 21ep 地基
sf a0004 0 44 s44 &        # a0004 卡0: single_far seed 44,给 horizon 结果第三个种子
wait
: > $ROOT/work/.done-ssh_resume
say "=== ssh_resume done ==="
