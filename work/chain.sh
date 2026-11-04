#!/usr/bin/env bash
# Wait for the orchestrator to go idle, then run the queued one-off sweeps in
# order.  Kept separate because orchestrate.sh must not be edited while it runs.
set -u
cd /users/PAS3151/qijia387/Phys-WM/work
while pgrep -f "bash orchestrate.sh" | grep -qv "^$$\$"; do sleep 60; done
printf '[%s] orchestrator idle; starting queued sweeps\n' "$(date +%H:%M:%S)"
bash lr_matched.sh
bash penalty_far.sh
printf '[%s] queued sweeps done\n' "$(date +%H:%M:%S)"
