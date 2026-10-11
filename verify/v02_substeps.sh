#!/usr/bin/env bash
# V02 時間刻み（substeps）で、理論との一致が変わる。substeps=1 と 10 で S4b の表を比べる。
source "$(dirname "$0")/_lib.sh"
patch_copy sim/s4b_mu_sweep.py $W/s4b_sub1.py "substeps=10" "substeps=1"
echo "== substeps=1";  quiet python $W/s4b_sub1.py | grep -A7 "^backend"
echo "== substeps=10"; quiet python sim/s4b_mu_sweep.py | grep -A7 "^backend"
