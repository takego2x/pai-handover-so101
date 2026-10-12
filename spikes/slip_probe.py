"""Spike: 握る力を緩めて滑り始めを検出し、μ/m（μ と重さの比）を推定できるか。

手順（環境ごとに並列）:
  1. 部品を F0 で掴み、PROBE_DZ だけ持ち上げて静止させる
  2. 粗い探索：握る力を F0 から指数的に緩め、部品が指に対して SLIP_MM 下がった力 F1 を記録
  3. すぐ 2*F1 に戻して止め、精密な探索：1.4*F1 → 0.5*F1 をゆっくり緩め、滑り始めの力 F2 を記録
  4. 滑り始めの条件 2 μ F = m g  ->  μ/m = g / (2 F2)。μ の推定 = m g / (2 F2)（m は正しい重さ）
使い方: python spikes/slip_probe.py <material>
"""
import sys
import time

import numpy as np
import torch
import genesis as gs

G = 9.81
SIDE = 0.04
DENSITY = {"POM": 1410.0, "aluminium": 2700.0, "steel": 7850.0}
MUS = [float(x) for x in __import__("os").environ.get("MUS", "0.05,0.1,0.2,0.3,0.5,0.7,1.0").split(",")]
F0 = 60.0          # 最初に握る力 [N]（μ=0.05 の鋼でも持てる）
F_MIN = 0.2
PROBE_DZ = 0.01    # 試しに持ち上げる高さ [m]
SLIP_MM = 1.0      # 指に対して部品がこれだけ下がったら「滑った」
COARSE_STEPS, FINE_STEPS, HOLD_STEPS = 300, 150, 30
XY = np.array([0.55, 0.0])

material = sys.argv[1]
grid = [(material, mu) for mu in MUS]
N = len(grid)
mass = np.array([DENSITY[m] * SIDE**3 for m, _ in grid])
true_mu = np.array([mu for _, mu in grid])

gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
                 rigid_options=gs.options.RigidOptions(box_box_detection=True, friction_cone=gs.friction_cone.elliptic,
                                                       impratio=10.0), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(gs.morphs.Box(size=(SIDE, SIDE, SIDE), pos=(*XY, SIDE / 2)),
                        material=gs.materials.Rigid(friction=1.0, rho=DENSITY[material]))
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
                          material=gs.materials.Rigid(friction=0.01))
scene.build(n_envs=N)
cube.set_friction_ratio(torch.tensor(true_mu, dtype=gs.tc_float, device=gs.device).unsqueeze(1), links_idx_local=[0])
assert abs(float(cube.get_mass()) - mass[0]) < 0.005 * mass[0], (cube.get_mass(), mass[0])

motors, fingers = np.arange(7), np.arange(7, 9)
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
hand = franka.get_link("hand")
quat = np.tile([0, 1, 0, 0], (N, 1))
grip_z = SIDE / 2 + 0.115
# 全環境を同じ姿勢から始める。初期の姿勢のままだと IK が環境ごとに違う解を返し、腕が回り込んで部品を払う環境が出る
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
franka.set_dofs_position(np.tile(HOME, (N, 1)))
franka.control_dofs_position(np.tile(HOME, (N, 1)))


def go(z, steps, force=None):
    q = franka.inverse_kinematics(link=hand, pos=np.tile([*XY, z], (N, 1)), quat=quat,
                                  init_qpos=franka.get_dofs_position())
    franka.control_dofs_position(q[:, :-2], motors)
    if force is None:
        franka.control_dofs_position(np.full((N, 2), 0.04), fingers)
    else:
        franka.control_dofs_force(np.column_stack([-force, -force]), fingers)
    for _ in range(steps):
        scene.step()


def rel():  # 手先から見た部品の高さ（下がると小さくなる）
    return cube.get_pos()[:, 2].cpu().numpy() - hand.get_pos()[:, 2].cpu().numpy()


t0 = time.time()
go(0.25, 150)
go(grip_z, 150)
go(grip_z, 60, np.full(N, 5.0))  # そっと閉じる（いきなり F0 で閉じると、部品をはじき出す・傾けて挟む）
for j in range(50):  # 5 N → F0 を 0.5 秒で上げる
    go(grip_z, 1, np.full(N, 5.0 + (F0 - 5.0) * (j + 1) / 50))
go(grip_z, 30, np.full(N, F0))
go(grip_z + PROBE_DZ, 60, np.full(N, F0))

import os
if os.environ.get('DIAG'):
    cp=cube.get_pos().cpu().numpy(); cq=cube.get_quat().cpu().numpy(); hp=hand.get_pos().cpu().numpy()
    fq=franka.get_dofs_position().cpu().numpy()[:,7:9]
    for i in range(N): print('diag', i, true_mu[i], 'cube', cp[i].round(4), 'quat', cq[i].round(3), 'hand_z', hp[i,2].round(4), 'fingers', fq[i].round(4))
F = np.full(N, F0)
phase = np.zeros(N, int)  # 0 粗い探索 / 1 戻して止める / 2 精密な探索 / 3 終わり / 4 検出できず
k = np.zeros(N, int)
F1 = np.full(N, np.nan)
F2 = np.full(N, np.nan)
base = rel()
coarse_r = (F_MIN / F0) ** (1 / COARSE_STEPS)
fine_r = (0.5 / 1.4) ** (1 / FINE_STEPS)
probe_steps = np.zeros(N, int)
for it in range(COARSE_STEPS + HOLD_STEPS + FINE_STEPS + 50):
    r = rel()
    slipped = (base - r) > SLIP_MM / 1000
    for i in range(N):
        if phase[i] == 0:
            if slipped[i]:
                F1[i], phase[i], k[i] = F[i], 1, 0
                F[i] = min(F0, 2 * F1[i])
            elif k[i] >= COARSE_STEPS:
                phase[i] = 4
            else:
                F[i] = F0 * coarse_r ** k[i]
        elif phase[i] == 1:
            if k[i] >= HOLD_STEPS:
                phase[i], k[i] = 2, 0
                base[i] = r[i]  # 戻した位置を新しい基準にする
                F[i] = 1.4 * F1[i]
        elif phase[i] == 2:
            if slipped[i]:
                F2[i], phase[i] = F[i], 3
                F[i] = min(F0, 2 * F2[i])
                probe_steps[i] = it
            elif k[i] >= FINE_STEPS:
                phase[i] = 4
            else:
                F[i] = 1.4 * F1[i] * fine_r ** k[i]
        k[i] += 1
    franka.control_dofs_force(np.column_stack([-F, -F]), fingers)
    scene.step()
    if (phase >= 3).all():
        break
still_held = cube.get_pos()[:, 2].cpu().numpy() > SIDE / 2 + PROBE_DZ * 0.5

print(f"{'material':9s} {'mass':>5s} {'mu':>5s} {'F_theory':>8s} {'F1':>7s} {'F2':>7s} {'mu_hat':>7s} {'err':>7s} {'t_probe':>7s} held")
for i, (m, mu) in enumerate(grid):
    f_th = mass[i] * G / (2 * mu)
    mu_hat = mass[i] * G / (2 * F2[i]) if np.isfinite(F2[i]) else np.nan
    err = (mu_hat / mu - 1) * 100
    print(f"{m:9s} {mass[i] * 1000:4.0f}g {mu:5.2f} {f_th:8.2f} {F1[i]:7.2f} {F2[i]:7.2f} {mu_hat:7.3f} {err:+6.1f}% "
          f"{probe_steps[i] * 0.01:6.2f}s {bool(still_held[i])}")
print(f"wall {time.time() - t0:.0f}s")
