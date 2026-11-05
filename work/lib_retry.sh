# srun with retry.  The controller went unreachable at 09:01 and again at
# 09:27; every srun in flight returned 1 within seconds, each loop burned
# through its remaining variants in under a minute, and both cards then sat
# idle for two hours.  A transient control-plane failure should cost a retry,
# not a batch.
srun_retry() {
  local gpu=$1; shift
  local try=0 rc=0
  while [ $try -lt 5 ]; do
    srun --jobid=8093487 --overlap -n1 --cpus-per-task=8 --gres=gpu:2 \
      env CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 "$@"
    rc=$?
    [ $rc -eq 0 ] && return 0
    try=$((try+1))
    printf '[%s]   srun rc=%s,第 %s 次重试前等 120s\n' "$(date +%H:%M:%S)" "$rc" "$try"
    sleep 120
  done
  return $rc
}
