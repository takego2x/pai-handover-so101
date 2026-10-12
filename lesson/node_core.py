"""L8: ノードの中身（dora 版と ROS 2 版で共通）。つなぎ方は lesson/nodes_dora/ と lesson/nodes_ros2/ にある。

  estimate_parts : 第一層。L4 と同じく、見て（VLM の候補）範囲を絞り、L3 の手順（重さ → 試し掴み）で、部品ごとのデータを作る
  Carrier        : 第二層。場面を 1 回だけ作り、届いたデータで部品を 1 個ずつ、L7 の方策で弧を描いて運ぶ

ノードの間を流れるメッセージ（JSON の文字列）:
  part_data : {"cell", "part", "data": {重さ, μ, 運ぶ力, 加速度}, "sim_truth": {材質, 本当の重さ, 本当の μ}}
              sim_truth はシミュレータの中に同じ部品を置くためだけに使う。計画（力と加速度）には data だけを使う
  result    : {"cell", "part", "material", "policy", "F_N", "delivered", "t_layer2_s"}
"""
import json
import time

import numpy as np

import common as c

XY = np.array([0.55, 0.0])
Z_GRIP = c.SIDE / 2 + 0.115


def estimate_parts(n, seed0=1000, cell=0, trials=7, prior="vlm", log=print):
    """第一層 1 台が、部品 n 個を「見て（VLM の候補）→ 触って（試し掴み）」測り、データの一覧を返す。
    prior="vlm" は L4 と同じく、Gemini の材質の候補で探す範囲を狭め、範囲の幅が 5% 以下になるまで試す。
    prior=None は L3 と同じく、全範囲を trials 回。n 個は並列の環境で同時に計算する。"""
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
    cands = [None] * n
    if prior == "vlm":  # 見る：L4 と同じ。候補の材質の文献値（1.5 倍に広げる）→ 測った重さで力の範囲へ
        rows = c.load()
        vlm = json.load(open(c.ROOT / "data" / "vlm_candidates_gemini.json"))["per_image"]
        for i in range(n):
            cands[i] = vlm[mats[i]][i % len(vlm[mats[i]])]
            r = [c.mu_range(x, rows=rows) for x in cands[i]]
            hi[i] = min(c.F_MAX, m_hat[i] * c.G / (2 * min(x[0] for x in r) / 1.5))
            lo[i] = max(0.3, m_hat[i] * c.G / (2 * max(x[1] for x in r) * 1.5))
        held_once, active, n_trials = np.zeros(n, bool), np.ones(n, bool), np.zeros(n, int)
        for _ in range(15):
            F = np.where(active, np.sqrt(lo * hi), hi)
            go(Z_GRIP, 30)
            close_gently(F)
            lift(F)
            held = z_part() > c.SIDE / 2 + 0.01
            hi, lo = np.where(active & held, F, hi), np.where(active & ~held, F, lo)
            held_once |= active & held
            n_trials += active
            done = active & (hi / lo <= 1.05)
            reopen = done & ~held_once & (hi < c.F_MAX)  # 範囲が外れていた → 上を全範囲に戻す
            hi = np.where(reopen, c.F_MAX, hi)
            active &= ~(done & ~reopen)
            if not active.any():
                break
    else:  # 触るだけ：L3 と同じ
        n_trials = np.full(n, trials)
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
                    "vlm_candidates": cands[i], "n_trials": int(n_trials[i]),
                    "data": {"ok": bool(weighed[i]), "mass_kg": round(float(m_hat[i]), 4), "mu": round(float(mu_hat[i]), 4),
                             "F_carry_N": round(float(F_carry[i]), 3), "a_mps2": round(float(a[i]), 3)},
                    "sim_truth": {"material": mats[i], "mass_kg": round(float(m_true[i]), 4), "mu": round(float(true_mu[i]), 4)}})
    log(f"[layer1 cell {cell}] estimated {n} parts (prior={prior}, trials per part {n_trials.mean():.1f})")
    return out


class Carrier:
    """第二層 1 台。L7 と同じ「弧を描いて運ぶ」場面で、届いたデータから方策が力・加速度・指の向きを決めて運ぶ。
    policy="learned" は L7 で学習した方策（data/policy_W.json）、"formula" は式の方策・接線向き。"""

    def __init__(self, policy="learned", w_path=None):
        import policy as pol  # L7 の部品
        self.pol = pol
        self.policy = policy
        wj = json.load(open(w_path or (c.ROOT / "data" / "policy_W.json")))
        self.W = wj["W"]
        pol.YAW_ONE, pol.YAW_WRAP = wj.get("yaw_one", False), wj.get("yaw_wrap", False)  # 学習したときと同じ扱い
        self.sc = pol.ArcScene(1)

    def carry(self, msg):
        d, t = msg["data"], msg["sim_truth"]
        mu_hat, m = np.array([d["mu"]]), np.array([d["mass_kg"]])  # 方策の入力はデータだけ
        if self.policy == "learned":
            kF, a, yaw = self.pol.learned(self.W, mu_hat, m)
        else:
            kF, a, yaw = self.pol.formula(1, tangent=True)
        # シミュレータの中の部品は、届いた部品と同じ材質・本当の μ にする（sim_truth はここでしか使わない）
        ok, F, tt = self.sc.episode([t["material"]], np.array([t["mu"]]), mu_hat, kF, a, yaw)
        return {"cell": msg["cell"], "part": msg["part"], "material": t["material"], "policy": self.policy,
                "F_N": round(float(F[0]), 3), "delivered": bool(ok[0]), "t_layer2_s": round(float(tt[0]), 2),
                "t_wall": time.time()}


def summarize(results):
    """記録ノードの集計。"""
    n = len(results)
    F = np.array([r["F_N"] for r in results])
    ok = sum(r["delivered"] for r in results)
    return (f"[monitor] delivered {ok}/{n}   grip force median {np.median(F):.1f} N (max {F.max():.1f})   "
            f"lift+carry time per part {np.mean([r['t_layer2_s'] for r in results]):.2f} s (simulated)")
