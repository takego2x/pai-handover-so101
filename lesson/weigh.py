"""L2: 掴んで持ち上げ、指が部品を支える力から重さを測る（手首の力センサーの代わり）。

部品は 5 種類の材質からランダムに 10 個（同じ 4 cm 角でも、重さは 67 g〜502 g と 7 倍違う）。
持ち上げて静止させると、部品に働く外からの力は「指からの力」だけになり、その上向きの成分が m g に等しい。
使い方: python lesson/weigh.py [--mistake moving]
  --mistake moving : 持ち上げている途中（加速している最中）に測る
"""
import argparse

import numpy as np

import common as c

ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=10)
ap.add_argument("--mistake", choices=["moving"])
args = ap.parse_args()
c.quiet()

N = args.n
mats, true_mu = c.sample_parts(N)
XY = np.array([0.55, 0.0])
F_WEIGH = float(__import__("os").environ.get("F_WEIGH", 60.0))  # 重さを測るときの把持力。どの部品でも落とさないよう、上限で握る

scene = c.make_scene()
scene.add_entity(c.gs.morphs.Plane())
part = c.add_part(scene, XY)
arm = c.add_franka(scene)
scene.build(n_envs=N)
m_true = c.set_parts(part, mats, true_mu)
c.setup_franka(arm, N)
z_grip = c.SIDE / 2 + 0.115


def go(z, steps, f=None):
    c.command(arm, [*XY, z], N, force=f)
    for _ in range(steps):
        scene.step()


def contact_fz():
    """部品が受けている接触の力（外から押される力の合計）の上下の成分 [N]。形は (N,)"""
    return part.get_links_net_contact_force().cpu().numpy()[:, 0, 2]


go(0.25, 150)
go(z_grip, 150)
go(z_grip, 50, 5.0)
for k in range(50):  # 0.5 秒かけて 5 N → 60 N
    go(z_grip, 1, 5.0 + (F_WEIGH - 5.0) * (k + 1) / 50)

if args.mistake == "moving":
    # 2 cm を加速度 9 m/s^2 で持ち上げながら、加速している間に測る
    pos, T = c.trapezoid(0.02, c.A_CAP)
    readings, accs = [], []
    z_prev = part.get_pos()[:, 2].cpu().numpy()
    v_prev = np.zeros(N)
    for k in range(int(np.ceil(T / c.DT))):
        go(z_grip + pos((k + 1) * c.DT), 1, F_WEIGH)
        z = part.get_pos()[:, 2].cpu().numpy()
        v = (z - z_prev) / c.DT
        if (k + 1) * c.DT < T / 2:  # 加速している間だけ、力と「部品が実際に出した加速度」を記録する
            readings.append(contact_fz())
            accs.append((v - v_prev) / c.DT)
        z_prev, v_prev = z, v
    fz = np.mean(readings, axis=0)
    go(z_grip + 0.02, 50, F_WEIGH)  # 持ち上がったかを判定する前に、止まって待つ
    accs = np.array(accs)  # 計画は 9 m/s^2 でも、ロボットは指令に遅れてついてくるので、部品の加速度はずっと小さい
    print(f"planned a = {c.A_CAP:.1f} m/s^2, actual a of the part while measuring: "
          f"mean {accs.mean():.2f}, max {accs.max():.2f} m/s^2")
else:
    for k in range(50):  # 0.5 秒かけて滑らかに 2 cm 持ち上げる
        s = (k + 1) / 50
        go(z_grip + 0.02 * (3 * s**2 - 2 * s**3), 1, F_WEIGH)
    go(z_grip + 0.02, 50, F_WEIGH)  # 0.5 秒止めて、揺れが収まるのを待つ
    readings = []
    for _ in range(20):  # 0.2 秒の平均を取る
        go(z_grip + 0.02, 1, F_WEIGH)
        readings.append(contact_fz())
    fz = np.mean(readings, axis=0)

m_hat = fz / c.G
# 持ち上がっているか（部品の底が床から 1 cm 以上）。床に残った部品も、床が支える力で「重さ」は正しく読めてしまう
lifted = part.get_pos()[:, 2].cpu().numpy() > c.SIDE / 2 + 0.01
print(f"mistake={args.mistake}")
print(f"{'#':>2s} {'material':9s} {'true [g]':>8s} {'measured [g]':>12s} {'error':>7s} {'lifted':>6s}")
for i in range(N):
    print(f"{i:2d} {mats[i]:9s} {m_true[i] * 1000:8.1f} {m_hat[i] * 1000:12.1f} {(m_hat[i] / m_true[i] - 1) * 100:+6.1f}% "
          f"{'yes' if lifted[i] else 'NO':>6s}")
err = np.abs(m_hat / m_true - 1) * 100
print(f"max |error| = {err.max():.1f}%   lifted {int(lifted.sum())}/{N}")
