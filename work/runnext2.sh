#!/usr/bin/env bash
set -u
cd /users/PAS3151/qijia387/Phys-WM/work
until [ -f .done-ssm_tolerance ]; do sleep 30; done
bash horizon.sh
