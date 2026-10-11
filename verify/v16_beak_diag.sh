#!/usr/bin/env bash
# V16 くちばしが理論より多くの力を要する原因を、①押し付け力 ②持ち上げの加速度 ③ずれ（滑り）④摩擦の近似 の順に切り分ける。
source "$(dirname "$0")/_lib.sh"
A=third_party/microduck_rl/src/mjlab_microduck/robot/microduck/assets
test -d $A || { echo "先に scripts/fetch_microduck_assets.sh"; exit 1; }
python verify/beak_probe.py $A pyramidal
python verify/beak_probe.py $A elliptic
echo "== 20 通りの表：多角錐（既定）"
patch_copy sim/beak_mu_sweep.py $W/beak_pyramidal.py \
  "rigid_options=gs.options.RigidOptions(friction_cone=gs.friction_cone.elliptic, impratio=10.0)" "rigid_options=gs.options.RigidOptions()"
quiet python $W/beak_pyramidal.py $A | grep -A6 "^object"
echo "== 20 通りの表：楕円錐（本線）"
quiet python sim/beak_mu_sweep.py $A | grep -A6 "^object"
