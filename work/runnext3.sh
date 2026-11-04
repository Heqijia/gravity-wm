#!/usr/bin/env bash
set -u
cd /users/PAS3151/qijia387/Phys-WM/work
until [ -f .done-ssm_staged ]; do sleep 60; done
bash single.sh
