"""L8: ノードの中身（dora 版と ROS 2 版で共通）。つなぎ方は lesson/l8_dora/ と lesson/l8_ros2/ にある。

  estimate_parts : 第一層。L3 と同じ手順（重さを測る → 7 回の試し掴み）で、部品ごとのデータを作る
  Carrier        : 第二層。場面を 1 回だけ作り、届いたデータで部品を 1 個ずつ運ぶ（L4 と同じ運び方）

ノードの間を流れるメッセージ（JSON の文字列）:
  part_data : {"cell", "part", "data": {重さ, μ, 運ぶ力, 加速度}, "sim_truth": {材質, 本当の重さ, 本当の μ}}
              sim_truth はシミュレータの中に同じ部品を置くためだけに使う。計画（力と加速度）には data だけを使う
  result    : {"cell", "part", "material", "F_N", "delivered", "t_layer2_s"}
"""
import math
import time

import numpy as np

import common as c

XY = np.array([0.55, 0.0])
Z_GRIP = c.SIDE / 2 + 0.115


def estimate_parts(n, seed0=1000, cell=0, trials=7, log=print):
    """第一層 1 台が、部品 n 個を試し掴みして、データの一覧を返す（n 個は並列の環境で同時に計算する）。"""
    mats, true_mu = c.sample_parts(n, seed0, stream=cell + 1)
    scene = c.make_scene()
    scene.add_entity(c.gs.morphs.Plane())
    part = c.add_part(scene, XY)
    arm = c.add_franka(scene)
    scene.build(n_envs=n)
    m_true = c.set_parts(part, mats, true_mu)
    c.setup_franka(arm, n)

    def go(z, k, f=None):
        c.command(arm, [*XY, z], n, force=f)
        for _ in range(k):
            scene.step()

    def close_gently(f):
        f0 = np.minimum(f, 5.0)
        go(Z_GRIP, 20, f0)
        for j in range(20):
            go(Z_GRIP, 1, f0 + (f - f0) * (j + 1) / 20)
        go(Z_GRIP, 20, f)

    def lift(f, dz=0.02, k=30):
        for j in range(k):
            s = (j + 1) / k
            go(Z_GRIP + dz * (3 * s**2 - 2 * s**3), 1, f)
        go(Z_GRIP + dz, 40, f)

    def z_part():
        return part.get_pos()[:, 2].cpu().numpy()

    go(0.25, 150)
    go(Z_GRIP, 150)
    full = np.full(n, c.F_MAX)
    close_gently(full)
    lift(full)
    readings = []
    for _ in range(20):
        go(Z_GRIP + 0.02, 1, full)
        readings.append(part.get_links_net_contact_force().cpu().numpy()[:, 0, 2])
    m_hat = np.mean(readings, axis=0) / c.G
    weighed = z_part() > c.SIDE / 2 + 0.01
    lo, hi = np.full(n, 0.3), np.full(n, c.F_MAX)
    for _ in range(trials):
        F = np.sqrt(lo * hi)
        go(Z_GRIP, 30)
        close_gently(F)
        lift(F)
        held = z_part() > c.SIDE / 2 + 0.01
        hi, lo = np.where(held, F, hi), np.where(held, lo, F)
    mu_hat = m_hat * c.G / (2 * hi)
    F_carry = np.where(weighed, c.carry_force(m_hat, mu_hat), c.F_MAX)
    a = np.where(weighed, c.accel_for(m_hat, mu_hat, F_carry), 0.5)
    a = np.where(a > 0, a, 0.5)
    out = []
    for i in range(n):
        out.append({"cell": cell, "part": i,
                    "data": {"ok": bool(weighed[i]), "mass_kg": round(float(m_hat[i]), 4), "mu": round(float(mu_hat[i]), 4),
                             "F_carry_N": round(float(F_carry[i]), 3), "a_mps2": round(float(a[i]), 3)},
                    "sim_truth": {"material": mats[i], "mass_kg": round(float(m_true[i]), 4), "mu": round(float(true_mu[i]), 4)}})
    log(f"[layer1 cell {cell}] estimated {n} parts")
    return out


class Carrier:
    """第二層 1 台。場面を 1 回だけ作り、carry() のたびに部品を置き直して運ぶ。"""

    START, GOAL, Z_UP = np.array([0.55, -0.20]), np.array([0.55, 0.20]), 0.35

    def __init__(self):
        self.scene = c.make_scene()
        self.scene.add_entity(c.gs.morphs.Plane())
        self.part = c.add_part(self.scene, self.START)
        self.arm = c.add_franka(self.scene)
        self.scene.build(n_envs=1)
        c.setup_franka(self.arm, 1)

    def _go(self, xy, z, k, f=None):
        c.command(self.arm, [*xy, z], 1, force=f)
        for _ in range(k):
            self.scene.step()

    def carry(self, msg):
        d, t = msg["data"], msg["sim_truth"]
        # シミュレータの中に、届いた部品と同じ部品を置き直す（材質と本当の μ はここでしか使わない）
        c.set_parts(self.part, [t["material"]], np.array([t["mu"]]))
        self.part.set_pos(np.array([[*self.START, c.SIDE / 2]]))
        self.part.set_quat(np.array([[1.0, 0.0, 0.0, 0.0]]))
        self.part.zero_all_dofs_velocity()
        self.arm.set_dofs_position(np.tile(c.HOME, (1, 1)))
        self.arm.zero_all_dofs_velocity()
        F = np.array([d["F_carry_N"]])  # 計画にはデータだけを使う
        a = d["a_mps2"]
        steps = 0
        self._go(self.START, 0.25, 150)
        self._go(self.START, Z_GRIP, 150)
        f0 = np.minimum(F, 5.0)
        self._go(self.START, Z_GRIP, 50, f0)
        for j in range(50):
            self._go(self.START, Z_GRIP, 1, f0 + (F - f0) * (j + 1) / 50)
        steps += 400
        for p0, p1 in [((*self.START, Z_GRIP), (*self.START, self.Z_UP)), ((*self.START, self.Z_UP), (*self.GOAL, self.Z_UP))]:
            p0, p1 = np.array(p0), np.array(p1)
            dist = float(np.linalg.norm(p1 - p0))
            pos, T = c.trapezoid(dist, a)
            for k in range(int(math.ceil(T / c.DT))):
                q = p0 + (p1 - p0) * pos((k + 1) * c.DT) / dist
                self._go(q[:2], q[2], 1, F)
            self._go(p1[:2], p1[2], 30, F)
            steps += int(math.ceil(T / c.DT)) + 30
        self._go(self.GOAL, self.Z_UP, 100, F)
        steps += 100
        p = self.part.get_pos().cpu().numpy()[0]
        delivered = bool(p[2] > 0.15 and np.linalg.norm(p[:2] - self.GOAL) < 0.05)
        return {"cell": msg["cell"], "part": msg["part"], "material": t["material"], "F_N": d["F_carry_N"],
                "delivered": delivered, "t_layer2_s": round(steps * c.DT, 2), "t_wall": time.time()}


def summarize(results):
    """記録ノードの集計。"""
    n = len(results)
    F = np.array([r["F_N"] for r in results])
    ok = sum(r["delivered"] for r in results)
    return (f"[monitor] delivered {ok}/{n}   grip force median {np.median(F):.1f} N (max {F.max():.1f})   "
            f"layer2 time per part {np.mean([r['t_layer2_s'] for r in results]):.1f} s (simulated)")
