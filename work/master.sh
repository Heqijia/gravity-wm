#!/usr/bin/env bash
# One chain for the remaining queues, in priority order.  Replaces the
# per-queue runners: each waited on a marker, which made inserting a queue in
# the middle impossible without killing the lot.
set -u
cd /users/PAS3151/qijia387/Phys-WM/work
step() { # step <marker> <script>
  [ -f ".done-$1" ] && { printf '[%s] %s already done\n' "$(date +%H:%M:%S)" "$1"; return; }
  printf '[%s] MASTER starting %s\n' "$(date +%H:%M:%S)" "$2"
  bash "$2"
}
# accelx is already running; wait for it rather than starting a second copy
until [ -f .done-accelx ]; do sleep 60; done
step ax_budget      ax_budget.sh       # B at the budget its C_0/eps implies
step ax_decompose   ax_decompose.sh    # which half of the 2D counterfactual breaks it
step queue3         queue3.sh          # ablations, strides, LeWM on ball_single
step budget36k      budget36k.sh       # is the certificate reachable at all?
step square_budget  square_budget.sh   # A at its implied budget, plus annealing
printf '[%s] MASTER all queues done\n' "$(date +%H:%M:%S)"
