"""試し：ロボットの摩擦を下げたとき、箱の摩擦の違いで「同じ力でも滑る」が起きるか。あわせて、指の角度による硬さの校正を測り直す。

  docker compose run --rm sim python handover/spikes/soft_mu.py [--robot-mu 0.1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import so101 as s  # noqa: E402
import softbox as sb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--robot-mu", type=float, default=0.1)
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[2]
R = 0.22
LIFT = [(f"m300_mu{mu}", 20000.0, 0.30, mu) for mu in (0.3, 0.5, 1.0)]
CAL = [(f"k{int(k)}_m{m}_mu{mu}", k, m, mu) for k in (2500.0, 3500.0, 5000.0, 8000.0, 12000.0, 20000.0) for m, mu in ((0.10, 0.6), (0.30, 1.0))]
LIFT = LIFT if __import__("os").environ.get("LIFT") else []
sc = s.make_scene()
sc.add_entity(s.gs.morphs.Plane())
ent = s.add_so101(sc, (0, 0), 0, friction=args.robot_mu)
boxes = {}
for i, (name, k, m, mu) in enumerate(LIFT + CAL):
    f = sb.xml(ROOT / "outputs" / "handover" / "softbox" / f"{name}.xml", k, m, mu)
    boxes[name] = sc.add_entity(s.gs.morphs.MJCF(file=f, pos=(R, 0.4 + 0.15 * i, 0)))
sc.build(n_envs=1)
A = s.Arm(sc, ent, (0, 0), 0, 1)


def fresh(name):
    sc.reset()
    for b in boxes.values():
        sb.hold_spring(b)
    boxes[name].set_pos(np.array([[R, 0, 0]]))
    ent.set_dofs_position(np.tile(s.FOLD, (1, 1))); ent.control_dofs_position(np.tile(s.FOLD, (1, 1))); A.last = None
    for _ in range(30):
        sc.step()
    A.set_grip_tau(s.TAU_MAX)
    s.move(sc, [(A, 0.0, R - s.JAW, 0.08, s.OPEN)], 120)
    s.move(sc, [(A, 0.0, R - s.JAW, 0.02, s.OPEN)], 100)
    s.move(sc, [(A, 0.0, R - s.JAW, 0.02, s.PRE)], 40)


print(f"ロボットの摩擦 {args.robot_mu}")
print("== 持ち上がるか（300 g、硬い箱、τ で閉じて 2 cm 上げる）  理論の最小 τ ＝ m g ÷ (2 μ × 14)")
TAUS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
for name, k, m, mu in LIFT:
    row = []
    for tau in TAUS:
        fresh(name)
        A.set_grip_tau(tau); s.move(sc, [(A, 0.0, R - s.JAW, 0.02, s.CLOSE)], 250)
        s.move(sc, [(A, 0.0, R - s.JAW, 0.04, s.CLOSE)], 80); s.move(sc, [(A, 0.0, R - s.JAW, 0.04, s.CLOSE)], 40)
        z = float(boxes[name].get_pos().cpu().numpy()[0, 2])
        row.append("持てた" if z > 0.025 else "  --  ")
    print(f"  μ {mu:.1f}  " + " ".join(f"{t:.2f}:{r}" for t, r in zip(TAUS, row)) + f"   理論 {m * 9.81 / (2 * mu * 14.0):.2f}")
print("== 校正：床の上で τ 0.4 の指の角度／持ち上げて宙で τ 0.4→0.5 の指の角度")
for name, k, m, mu in CAL:
    fresh(name)
    A.set_grip_tau(0.4); s.move(sc, [(A, 0.0, R - s.JAW, 0.02, s.CLOSE)], 300)
    q_floor = float(ent.get_dofs_position().cpu().numpy()[0, 5])
    s.move(sc, [(A, 0.0, R - s.JAW, 0.04, s.CLOSE)], 80); s.move(sc, [(A, 0.0, R - s.JAW, 0.04, s.CLOSE)], 40)
    air = []
    for tau in [0.4, 0.45, 0.5]:
        A.set_grip_tau(tau); s.move(sc, [(A, 0.0, R - s.JAW, 0.04, s.CLOSE)], 120)
        air.append((tau, float(ent.get_dofs_position().cpu().numpy()[0, 5]), float(s.squeeze(A, boxes[name])[0])))
    z = float(boxes[name].get_pos().cpu().numpy()[0, 2])
    Ts, Qs, Fs = np.array(air).T
    print(f"  k {k:6.0f}  {m * 1000:3.0f} g  μ {mu}  強度 {sb.strength_of(k):5.1f}   床 τ0.4 の角度 {q_floor:.4f}   宙の傾き dq/dτ {np.polyfit(Ts, Qs, 1)[0]:+.4f}   宙 τ0.4 の角度 {Qs[0]:.4f}  箱 z {z:.3f}  沈み {sb.squash(boxes[name]) * 1000:.2f} mm")
