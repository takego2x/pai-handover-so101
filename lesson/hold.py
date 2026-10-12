"""L1: 指 2 本で持てるかは「2 μ F >= m g」で決まる。シミュレータの数字と理論を 24 マス突き合わせる。

部品はアルミの 4 cm 角（173 g、重さ m g = 1.70 N）。μ を 6 通り × 把持力 F を 4 通り = 24 の環境を並列に回す。
使い方: python lesson/hold.py [--mistake finger_mu | substeps1]
  --mistake finger_mu : 指の μ を 0.01 にしない（Genesis の既定 1.0 のまま）
  --mistake substeps1 : 1 ステップの中の細かい計算（substeps）を 10 → 1 にする
  --mistake slam      : 指をそっと閉じず、いきなり目標の力で閉じる
  --mistake slam_substeps1 : いきなり閉じる ＋ substeps=1
  --mistake jump      : 持ち上げの目標を一度に高さ 30 cm（いまの位置から 16.5 cm 上）に置く（滑らかに動かさない）
"""
import argparse

import numpy as np

import common as c

MUS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
FORCES = [2.0, 5.0, 10.0, 20.0]

ap = argparse.ArgumentParser()
ap.add_argument("--mistake", choices=["finger_mu", "substeps1", "jump", "slam", "slam_substeps1"])
args = ap.parse_args()
c.quiet()

grid = [(mu, f) for mu in MUS for f in FORCES]
N = len(grid)
mats = ["aluminium"] * N
true_mu = np.array([mu for mu, _ in grid])
force = np.array([f for _, f in grid])
XY = np.array([0.55, 0.0])

scene = c.make_scene(substeps=1 if args.mistake in ("substeps1", "slam_substeps1") else 10)
scene.add_entity(c.gs.morphs.Plane())
part = c.add_part(scene, XY)
arm = c.add_franka(scene, finger_mu=1.0 if args.mistake == "finger_mu" else 0.01)
scene.build(n_envs=N)
m = c.set_parts(part, mats, true_mu)
c.setup_franka(arm, N)

z_grip = c.SIDE / 2 + 0.115  # 手先（hand）から指先の中心までが約 0.115 m


def go(z, steps, f=None):
    c.command(arm, [*XY, z], N, force=f)
    for _ in range(steps):
        scene.step()


go(0.25, 150)              # 部品の上へ
go(z_grip, 150)            # 下ろす
if args.mistake in ("slam", "slam_substeps1"):  # いきなり目標の力で閉じる
    go(z_grip, 100, force)
else:
    go(z_grip, 50, np.minimum(force, 5.0))  # そっと閉じる（5 N まで）
    go(z_grip, 50, force)                   # それから目標の力
# ゆっくり持ち上げる。目標を一度に 16.5 cm 上に置くと、ロボットは急加速して「重さ＋加速の分」が要る
# （わざと失敗 jump で確かめる）。ここでは 2 秒かけて滑らかに動かし、静止して持つ条件だけを見る
if args.mistake == "jump":
    go(0.30, 200, force)
else:
    for k in range(200):
        s_ = (k + 1) / 200
        go(z_grip + (0.30 - z_grip) * (3 * s_**2 - 2 * s_**3), 1, force)  # 滑らかな S 字（始めと終わりの速さが 0）
go(0.30, 50, force)
z = part.get_pos()[:, 2].cpu().numpy()

held = z > 0.10
theory = 2 * true_mu * force >= m[0] * c.G
print(f"part: aluminium {m[0] * 1000:.0f} g (m g = {m[0] * c.G:.2f} N)   mistake={args.mistake}")
print("simulator (OK = held, -- = dropped)        theory 2 mu F >= m g")
print("mu \\ F  " + "".join(f"{f:>6.0f}N" for f in FORCES) + "   " + "".join(f"{f:>6.0f}N" for f in FORCES))
match = int((held == theory).sum())
for i, mu in enumerate(MUS):
    s = "".join(f"{'OK' if held[i * len(FORCES) + j] else '--':>7s}" for j in range(len(FORCES)))
    t = "".join(f"{'OK' if theory[i * len(FORCES) + j] else '--':>7s}" for j in range(len(FORCES)))
    print(f"{mu:5.2f}  {s}   {t}")
print(f"match {match}/{N}")
