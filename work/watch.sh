#!/usr/bin/env bash
# One line per state change in the queue, so an idle GPU shows up as an event.
# Emits on: a phase starting or finishing, a non-zero exit, and a GPU that has
# been idle for two consecutive checks with queued work still pending.
set -u
cd /users/PAS3151/qijia387/Phys-WM/work
last_exit=$(wc -l < logs/_exits.txt)
last_log=$(wc -l < chain.log)
idle=0
while true; do
  now_exit=$(wc -l < logs/_exits.txt)
  if [ "$now_exit" -gt "$last_exit" ]; then
    tail -n $(( now_exit - last_exit )) logs/_exits.txt | awk '$0!~/exit=0$/ {print "FAILED " $0}'
    last_exit=$now_exit
  fi
  now_log=$(wc -l < chain.log)
  if [ "$now_log" -gt "$last_log" ]; then
    tail -n $(( now_log - last_log )) chain.log | grep -E "PHASE|=== |!!" || true
    last_log=$now_log
  fi
  running=$(pgrep -c -f "scripts/(posttrain|evaluate|pretrain|build_cache|train_probes|gen_data)\.py" 2>/dev/null || echo 0)
  if [ "$running" = "0" ]; then
    idle=$(( idle + 1 ))
    [ "$idle" = 3 ] && echo "IDLE: no training or evaluation job running for ~3 minutes"
  else
    idle=0
  fi
  sleep 60
done
