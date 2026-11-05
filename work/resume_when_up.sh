#!/usr/bin/env bash
# Ascend went into cluster-wide maintenance at 09:02 (controller reboot, 269
# nodes draining, every partition inactive).  Both allocations still show
# RUNNING but step launches fail with a protocol authentication error, so the
# work cannot proceed until the cluster is back.  This waits for a step to
# launch again, then restarts the batch -- with retry this time.
set -u
ROOT=/users/PAS3151/qijia387/Phys-WM
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

say "=== 等待集群恢复(每 5 分钟探一次)==="
probe() {
  for j in 8093487 8236469; do
    if timeout 90 srun --jobid=$j --overlap -n1 hostname >/dev/null 2>&1; then echo "$j"; return 0; fi
  done
  return 1
}
tries=0
while :; do
  if JOB=$(probe); then say "  集群恢复,可用作业 $JOB"; break; fi
  tries=$((tries+1))
  [ $((tries % 12)) -eq 0 ] && say "  仍在维护中($((tries*5)) 分钟)"
  sleep 300
done
echo "$JOB" > $ROOT/work/.active_jobid
say "=== 重启早晨批次(带重试)==="
setsid nohup bash $ROOT/work/morning.sh >> $ROOT/work/chain.log 2>&1 &
say "  已启动"
