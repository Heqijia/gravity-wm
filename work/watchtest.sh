#!/usr/bin/env bash
# Run two watchers concurrently for 60s and count the collector steps they create.
W=/users/PAS3151/qijia387/Phys-WM/tools/gpuwatch.sh
timeout 60 bash "$W" -i 10 >/dev/null 2>&1 &
sleep 3
timeout 60 bash "$W" -i 10 >/dev/null 2>&1 &
for i in 1 2 3 4 5 6; do
  sleep 10
  printf 't=%02ds  collector steps: %s   owner=%s\n' "$((i*10))" \
    "$(squeue -u "$USER" -s -h -o '%j' 2>/dev/null | grep -cx bash || echo 0)" \
    "$(cat "$HOME/.cache/gpuwatch"/*.owner 2>/dev/null | tr '\n' ' ')"
done
wait
sleep 15
printf 'after both quit: collector steps: %s  heartbeat: %s\n' \
  "$(squeue -u "$USER" -s -h -o '%j' 2>/dev/null | grep -cx bash || echo 0)" \
  "$([ -f "$HOME/.cache/gpuwatch/heartbeat" ] && echo present || echo gone)"
