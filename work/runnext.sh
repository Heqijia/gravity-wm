#!/usr/bin/env bash
# Wait on a marker file, not on a process name: pgrep -f for a script name also
# matches the shell command line that launched it, so such a loop can never exit.
set -u
cd /users/PAS3151/qijia387/Phys-WM/work
until [ -f .done-refresh_evals ]; do sleep 30; done
bash ssm_tolerance.sh
