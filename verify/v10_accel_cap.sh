#!/usr/bin/env bash
# V10 計画した加速度を、ロボットは出せるか。上限を 9 / 15 / 25 m/s2 にして、実際に出た加速度（p95）を測る。
source "$(dirname "$0")/_lib.sh"
for c in 9 15 25; do printf "A_CAP=%-3s " $c; A_CAP=$c quiet python sim/transport.py aluminium A 5 | grep -o "a_plan=.*success=[0-9/]*"; done
