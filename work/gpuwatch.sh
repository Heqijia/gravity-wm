#!/usr/bin/env bash
# Report when a stage finishes, when anything fails, and when the GPUs are
# genuinely idle.
#
# Instantaneous utilisation alone is the wrong signal: a latent-cache build
# holds 6.7 GB and spends most of each sample at 0% because it is I/O bound,
# which reads as idle.  Allocated memory is the reliable indicator that work
# is resident, so a GPU counts as busy if either is non-trivial.  The failure
# that matters most here is the opposite one -- a finished stage leaving both
# cards empty while nobody notices -- which has cost about six hours so far.
set -u
cd /users/PAS3151/qijia387/Phys-WM
idle=0; seen=""
# Seed the ledger with what is already finished, silently.  Reporting the
# backlog on the first pass buries the one event that matters in a dozen that
# do not, and an over-talkative monitor gets shut off automatically.
for m in work/.done-*; do [ -f "$m" ] && seen="$seen$m "; done
for f in $(ls -t work/logs/*.log 2>/dev/null | head -25); do
  grep -qiE "Traceback|CUDA out of memory|unbound variable|exit code [1-9]" "$f" 2>/dev/null \
    && seen="$seen ERR:$f "
done
echo "[$(date +%H:%M)] 监控启动:已忽略 $(ls work/.done-* 2>/dev/null | wc -l) 个旧标记,只报新事件"
while true; do
  raw=$(srun --jobid=8093487 --overlap -n1 nvidia-smi \
        --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null \
        | grep -E "^[01],")
  busy=$(echo "$raw" | awk -F'[, ]+' '$2>5 || $4>500' | wc -l)
  jobs=$(pgrep -cf "scripts/(pretrain|posttrain|evaluate|build_cache|train_probes|generate_bounce)\.py" || true)
  jobs=${jobs:-0}

  for m in work/.done-*; do
    [ -f "$m" ] || continue
    case "$seen" in *"$m "*) continue ;; esac
    seen="$seen$m "
    echo "[$(date +%H:%M)] 完成: $(basename "${m#*.done-}")  gpu=[$(echo "$raw" | tr '\n' ';')] jobs=$jobs"
  done

  for f in $(ls -t work/logs/*.log 2>/dev/null | head -25); do
    grep -qiE "Traceback|CUDA out of memory|unbound variable|exit code [1-9]" "$f" 2>/dev/null || continue
    case "$seen" in *"ERR:$f "*) continue ;; esac
    seen="$seen ERR:$f "
    echo "[$(date +%H:%M)] FAIL $(basename "$f")"; grep -iE "Error|Traceback" "$f" | tail -1
  done

  if [ "$busy" -eq 0 ]; then
    idle=$((idle+1))
    if [ "$idle" -ge 2 ]; then
      echo "[$(date +%H:%M)] IDLE 两张卡空闲 >=10 分钟 (jobs=$jobs) — 派任务"
      idle=0
    fi
  else
    idle=0
  fi
  sleep 300
done
