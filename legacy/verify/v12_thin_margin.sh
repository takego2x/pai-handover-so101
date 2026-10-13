#!/usr/bin/env bash
# V12 余裕が薄いと、環境の数だけで結果が分かれる。当時の条件（摩擦は既定の多角錐、安全率は「加速度の 0.8 倍」）に戻し、
#     μ が下限ちょうどの seed 1004 を、1 環境と 5 環境で回す。比較として、今の条件（楕円錐、摩擦力に 0.85）でも回す。
source "$(dirname "$0")/_lib.sh"
patch_copy sim/transport.py $W/transport_old_safety.py \
  'return min(A_CAP, max(0.0, SAFETY * 2 * mu_assumed * FORCE / MASS - G))' 'return min(A_CAP, 0.8 * a_max(mu_assumed))'
python - <<'PY'
p = "outputs/verify/transport_old_safety.py"; s = open(p).read()
old = "RigidOptions(box_box_detection=True, friction_cone=gs.friction_cone.elliptic, impratio=10.0)"
assert old in s; open(p, "w").write(s.replace(old, "RigidOptions(box_box_detection=True)"))
PY
sed -i '' 's#sys.path.insert(0, str(Path(__file__).resolve().parent))#sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "sim"))#' $W/transport_old_safety.py 2>/dev/null || \
sed -i 's#sys.path.insert(0, str(Path(__file__).resolve().parent))#sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "sim"))#' $W/transport_old_safety.py
for n in 1 5; do
  quiet python $W/transport_old_safety.py POM C $n --seed0 1004 --json $W/old_n$n.json > /dev/null
  python -c "import json;r=json.load(open('$W/old_n$n.json'));print('old (pyramidal + accel x0.8), n_envs=$n: seed1004 success =', r['success'][0], ' margin =', r['margin'][0])"
done
for n in 1 5; do
  quiet python sim/transport.py POM C $n --seed0 1004 --json $W/new_n$n.json > /dev/null
  python -c "import json;r=json.load(open('$W/new_n$n.json'));print('now (elliptic + friction x0.85), n_envs=$n: seed1004 success =', r['success'][0], ' margin =', r['margin'][0])"
done
