"""試し：潰れる箱（柔らかさ k と強度を持つ箱）を作り、SO-101 の指の角度だけから硬さを測れるかを確かめる。

箱 ＝ 本体（自由に動く）＋ 外側の面の板（すべり関節でつなぐ）。板の関節は「位置 0 に戻ろうとするばね」：
  板を押す力 ＝ k × 沈み量（ただし 強度 まで）。強度を超えると、板は沈みきる（＝潰れた）。
決まり：柔らかい箱ほど弱い（強度 ＝ k × 3 mm）。

  docker compose run --rm sim python handover/spikes/soft_box.py
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import so101 as s  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SIDE, PLATE, TRAVEL = 0.03, 0.010, 0.008       # 箱の一辺、板の厚さ、沈みきる量。板は沈みきる量より厚くする
                                                # （板が 4 mm で、沈みきる前に指が本体の面に当たり、潰れが 4 mm で止まった）


def box_xml(path, k, strength, mass):
    """本体＋板の MJCF。板は +x の面（指が閉じる向きに置くと、動く側の指が板を押す）"""
    h = SIDE / 2
    core_hx = (SIDE - PLATE) / 2
    m_plate = mass * 0.1
    xml = f"""<mujoco model="soft_box">
  <worldbody>
    <body name="core" pos="0 0 {h}">
      <freejoint/>
      <geom type="box" size="{core_hx} {h} {h}" pos="{-PLATE / 2} 0 0" mass="{mass - m_plate}" friction="1 0.005 0.0001" rgba="0.72 0.55 0.36 1"/>
      <body name="plate" pos="{h - PLATE / 2} 0 0">
        <joint name="squash" type="slide" axis="1 0 0" range="{-TRAVEL} 0.001" damping="2"/>
        <geom type="box" size="{PLATE / 2} {h} {h}" mass="{m_plate}" friction="1 0.005 0.0001" rgba="0.60 0.45 0.30 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="spring" joint="squash" kp="{k}" forcerange="{-strength} {strength}"/>
  </actuator>
</mujoco>
"""
    Path(path).write_text(xml)
    return str(path)


CASES = [  # (名前, k [N/m], 重さ kg)。強度 ＝ k × 3 mm
    ("soft", 2000.0, 0.10),
    ("mid", 5000.0, 0.10),
    ("hard", 20000.0, 0.10),
]
if os.environ.get("CALIB"):
    CASES = [(f"k{int(k)}", k, 0.10) for k in [2500.0, 3500.0, 5000.0, 8000.0, 12000.0, 20000.0]]
R = 0.22
sc = s.make_scene()
sc.add_entity(s.gs.morphs.Plane())
ent = s.add_so101(sc, (0, 0), 0)
boxes = []
for i, (name, k, m) in enumerate(CASES):
    f = box_xml(ROOT / "outputs" / "handover" / f"soft_box_{name}.xml", k, k * 0.003, m)
    b = sc.add_entity(s.gs.morphs.MJCF(file=f, pos=(R, 0.4 + 0.2 * i, 0)), material=s.gs.materials.Rigid(friction=1.0))
    boxes.append(b)
sc.build(n_envs=1)
for b, (name, k, m) in zip(boxes, CASES):
    print(name, "dofs", b.n_dofs, "kp", b.get_dofs_kp().cpu().numpy().round(1).tolist(), "force range", [x.cpu().numpy().round(1).tolist() for x in b.get_dofs_force_range()])
    b.control_dofs_position(np.zeros((1, 1)), [b.n_dofs - 1])
A = s.Arm(sc, ent, (0, 0), 0, 1)
for _ in range(30):
    sc.step()

import os
if os.environ.get("TRACE"):
    b = boxes[1]
    sc.reset(); b.set_pos(np.array([[R, 0, 0]]))
    ent.set_dofs_position(np.tile(s.FOLD, (1, 1))); ent.control_dofs_position(np.tile(s.FOLD, (1, 1))); A.last = None
    for _ in range(30): sc.step()
    s.move(sc, [(A, 0.0, R - s.JAW, 0.08, s.OPEN)], 120); s.move(sc, [(A, 0.0, R - s.JAW, 0.02, s.OPEN)], 100)
    print("before close: tip", A.tip().round(3).tolist(), "box", b.get_pos().cpu().numpy().round(3).tolist())
    A.set_grip_tau(s.TAU_MAX); s.move(sc, [(A, 0.0, R - s.JAW, 0.02, s.PRE)], 40)
    A.set_grip_tau(0.3)
    for k_ in range(6):
        s.move(sc, [(A, 0.0, R - s.JAW, 0.02, s.CLOSE)], 50)
        c = ent.get_contacts(with_entity=b)
        print(f"  t {k_ * 0.5 + 0.5:.1f}s jaw {float(ent.get_dofs_position()[0, 5]):.3f} squeeze {float(s.squeeze(A, b)[0]):.2f} box {b.get_pos().cpu().numpy().round(4).tolist()} "
              f"contacts {[ (int(x), int(y)) for x, y, v in zip(c['link_a'].cpu().numpy()[0], c['link_b'].cpu().numpy()[0], c['valid_mask'].cpu().numpy()[0]) if v]}")
    print("links A", [(l.idx, l.name) for l in ent.links][-2:], "box links", [(l.idx, l.name) for l in b.links])
    raise SystemExit
# 試す箱を、順に原点の前（22 cm）へ置き直して測る。1 回だけ閉じ、開かずに力の上限だけを変えて、指の角度を読む
R_JAW = 1 / 14.5 * 1.0   # 指の関節の軸から、箱に触れる所までの長さ ≒ 1 ÷ (14.5 N/(N·m)) [m]（H1c で測った傾きの逆数）
TAUS = [0.3, 0.4]
res = {}
for b, (name, k, m) in zip(boxes, CASES):
    sc.reset()
    for bb in boxes:
        bb.control_dofs_position(np.zeros((1, 1)), [bb.n_dofs - 1])   # 初めの状態に戻すと、板のばねの指示も消えるので出し直す
    b.set_pos(np.array([[R, 0, 0]]))
    ent.set_dofs_position(np.tile(s.FOLD, (1, 1))); ent.control_dofs_position(np.tile(s.FOLD, (1, 1))); A.last = None
    for _ in range(30):
        sc.step()
    az, r = 0.0, R - s.JAW
    A.set_grip_tau(s.TAU_MAX)   # 前の箱の最後の力の上限が残っていると、指が開ききらず、下りる途中で箱の上に当たって押しのけた
    s.move(sc, [(A, az, r, 0.08, s.OPEN)], 120)
    s.move(sc, [(A, az, r, 0.02, s.OPEN)], 100)
    A.set_grip_tau(s.TAU_MAX); s.move(sc, [(A, az, r, 0.02, s.PRE)], 40)
    if os.environ.get("DBG"):
        c = ent.get_contacts()
        names = {l.idx: l.name for l in ent.links}
        for bb in boxes:
            for l in bb.links: names[l.idx] = f"{bb.idx}:{l.name}"
        print(f"   DBG {name}: tip {A.tip().round(3).tolist()} box {b.get_pos().cpu().numpy().round(3).tolist()} jaw {float(ent.get_dofs_position()[0, 5]):.2f} "
              f"force range jaw {[x.cpu().numpy()[0, 5] for x in ent.get_dofs_force_range()]} contacts "
              f"{[(names.get(int(x), x), names.get(int(y), y)) for x, y, v in zip(c['link_a'].cpu().numpy()[0], c['link_b'].cpu().numpy()[0], c['valid_mask'].cpu().numpy()[0]) if v]}")
    rows = []
    for j, tau in enumerate(([0.25, 0.3, 0.35, 0.4] if os.environ.get('CALIB') else TAUS + [0.6, 0.8, 1.2])):
        A.set_grip_tau(tau)
        s.move(sc, [(A, az, r, 0.02, s.CLOSE)], 300 if j == 0 else 150)
        q_jaw = float(ent.get_dofs_position().cpu().numpy()[0, 5])
        q_sq = float(b.get_dofs_position().cpu().numpy()[0, -1])
        F = float(s.squeeze(A, b)[0])
        rows.append((tau, F, q_jaw, q_sq))
    res[name] = rows
    print(f"== {name}: k {k:.0f} N/m  強度 {k * 0.003:.1f} N")
    for tau, F, qj, qs in rows:
        print(f"   τ {tau:.1f}  挟む力 {F:5.2f} N  指の角度 {qj:+.4f}  板の沈み（本当）{-qs * 1000:5.2f} mm  F/k {min(F / k, TRAVEL) * 1000:5.2f} mm  {'潰れた' if qs < -TRAVEL * 0.9 else ''}")
# 推定：力を数段階に変えたときの、指の角度の変わり方（直線の傾き dq/dF）。柔らかいほど大きい
print("箱      本当の k   強度    傾き dq/dF [rad/N]   1/k [mm/N]")
for name, k, m in CASES:
    pts = [(F, qj) for tau, F, qj, qs in res[name] if tau <= 0.41 and F > 0]
    if len(pts) >= 2:
        Fs, qs_ = np.array(pts).T
        slope = np.polyfit(Fs, qs_, 1)[0]
    else:
        slope = float("nan")
    print(f"{name:6s} {k:8.0f}  {k * 0.003:5.1f} N   {slope:+.5f}            {1000 / k:.3f}   点 {len(pts)}")
