#!/usr/bin/env bash
# V03 接触の μ は「2 物体の大きいほう」。指の μ を既定のままにすると、箱の μ を変えても結果が変わらない。
source "$(dirname "$0")/_lib.sh"
patch_copy sim/s4_grasp_one.py $W/s4_default_finger.py \
  '                          material=gs.materials.Rigid(friction=0.01))' '                          )'
for mu in 0.05 0.1 0.5; do
  printf "finger default, cube mu=%s: " $mu; quiet python $W/s4_default_finger.py $mu 5 10 | grep -o "cube z=.*HELD\|cube z=.*DROPPED"
  printf "finger 0.01,    cube mu=%s: " $mu; quiet python sim/s4_grasp_one.py $mu 5 10 | grep -o "cube z=.*HELD\|cube z=.*DROPPED"
done
