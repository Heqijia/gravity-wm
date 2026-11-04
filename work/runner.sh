#!/usr/bin/env bash
# Run the queues named in queue.list, in order, skipping those already done.
#
# The list is a data file, not part of this script, because bash reads a script
# incrementally: editing one that is running shifts the byte offsets under it
# and the shell resumes mid-statement.  That happened three times while this
# study was running.  Appending to queue.list is safe at any moment; editing
# runner.sh is not, and now never necessary.
set -u
cd /users/PAS3151/qijia387/Phys-WM/work
while :; do
  pending=0
  while read -r marker script _; do
    case "$marker" in ''|\#*) continue ;; esac
    [ -f ".done-$marker" ] && continue
    [ -f "$script" ] || { printf '[%s] RUNNER missing %s\n' "$(date +%H:%M:%S)" "$script"; continue; }
    # already running as a child of an earlier runner?  wait it out instead
    if pgrep -f "bash $script" | grep -qv "^$$\$"; then
      printf '[%s] RUNNER %s already running; waiting\n' "$(date +%H:%M:%S)" "$script"
      while pgrep -f "bash $script" | grep -qv "^$$\$"; do sleep 30; done
      continue
    fi
    printf '[%s] RUNNER starting %s\n' "$(date +%H:%M:%S)" "$script"
    bash "$script"
    pending=1
    break
  done < queue.list
  [ "$pending" = 0 ] && break
done
printf '[%s] RUNNER queue empty\n' "$(date +%H:%M:%S)"
