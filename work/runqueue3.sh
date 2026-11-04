#!/usr/bin/env bash
set -u
cd /users/PAS3151/qijia387/Phys-WM/work
until [ -f .done-square ]; do sleep 60; done
bash accelx.sh
