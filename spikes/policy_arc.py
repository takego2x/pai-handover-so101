"""Spike（2026-10-13）：運ぶ方策を学習すると、式の方策に勝てる場面があるか。

場面：第二層の腕（根元が原点）が、床の部品（4 cm 角）を根元の周りに弧を描いて運ぶ（-60° → 110°、170°）。
方策：入力 = 試し掴みのデータ（重さ m、μ。μ は本当の値より少し小さめに測れている前提で、0.85〜0.97 倍）
      出力 = 握る力の倍率 kF（式の力 × kF、上限 60 N）、加速度 a、指の向きのずれ yaw（根元から見た方向から）
  式の方策     : kF = 1、a = 9、yaw = 0（半径の向き）または 90°（接線の向き）
  学習した方策 : [kF の対数, a, yaw] = W @ [1, log μ, log m]（係数 9 個）。CEM で学ぶ
採点：成功なら 1 − 0.3·kF − 0.1·(時間 / 2 秒)、失敗なら −fail_penalty（既定 1）。
使い方: python spikes/policy_arc.py [--gens 12] [--pop 16] [--parts 8]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson"))
import common as c  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gens", type=int, default=12)
ap.add_argument("--pop", type=int, default=16)
ap.add_argument("--parts", type=int, default=8)
ap.add_argument("--test-parts", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fail-penalty", type=float, default=1.0, help="落としたときの得点（負の値の大きさ）")
ap.add_argument("--out", default=str(c.ROOT / "outputs" / "policy_arc.json"))
args = ap.parse_args()
c.quiet()

TH0, TH1, R = math.radians(-60.0), math.radians(110.0), 0.55
Z_GRIP, Z_UP = c.SIDE / 2 + 0.115, 0.35
N = max(args.pop * args.parts, args.test_parts * 3)

scene = c.make_scene()
scene.add_entity(c.gs.morphs.Plane())
part = c.add_part(scene, (R * math.cos(TH0), R * math.sin(TH0)))
arm = c.add_franka(scene)
scene.build(n_envs=N)
c.setup_franka(arm, N)


def polar(r, th):
    return np.array([r * math.cos(th), r * math.sin(th)])


def arc(t, z):  # t: (N,) 0..1。半径一定で -60° → 110°
    ang = TH0 + (TH1 - TH0) * t
    return np.column_stack([R * np.cos(ang), R * np.sin(ang), np.broadcast_to(z, ang.shape)])


def cmd(pos, yaw_off, f=None):
    az = np.arctan2(pos[:, 1], pos[:, 0])
    seed = np.tile(c.HOME, (N, 1))
    seed[:, 0] = az
    seed[:, 6] = c.HOME[6] - yaw_off
    c.command(arm, pos, N, force=f, yaw=az + yaw_off, seed=torch.tensor(seed, dtype=c.gs.tc_float, device=c.gs.device))


def episode(mats, true_mu, mu_hat, kF, a, yaw_off):
    """全環境で 1 回運ぶ。環境ごとに部品・方策の出力が違ってよい。成功・力・時間を返す。"""
    m = c.set_parts(part, mats, true_mu)
    start = np.tile([*polar(R, TH0), c.SIDE / 2], (N, 1))
    part.set_pos(start)
    part.set_quat(np.tile([math.cos(TH0 / 2), 0, 0, math.sin(TH0 / 2)], (N, 1)))  # 部品は半径の向きにそろえて置く
    part.zero_all_dofs_velocity()
    arm.set_dofs_position(np.tile(c.HOME, (N, 1)))
    arm.zero_all_dofs_velocity()
    F = np.minimum(c.F_MAX, c.carry_force(m, mu_hat) * kF)
    up0 = arc(np.zeros(N), 0.25)
    for _ in range(150):
        cmd(up0, yaw_off)
        scene.step()
    down = arc(np.zeros(N), Z_GRIP)
    for k in range(100):
        cmd(up0 + (down - up0) * (k + 1) / 100, yaw_off)
        scene.step()
    for _ in range(50):
        cmd(down, yaw_off)
        scene.step()
    f0 = np.minimum(F, 5.0)
    for k in range(50):
        cmd(down, yaw_off, f0 + (F - f0) * (k + 1) / 50)
        scene.step()
    for _ in range(20):
        cmd(down, yaw_off, F)
        scene.step()
    # 持ち上げ（台形、環境ごとの a）→ 弧（台形、環境ごとの a）
    lifts = [c.trapezoid(Z_UP - Z_GRIP, ai) for ai in a]
    L = R * (TH1 - TH0)
    moves = [c.trapezoid(L, ai) for ai in a]
    T1 = max(t for _, t in lifts)
    for k in range(int(math.ceil(T1 / c.DT)) + 20):
        z = Z_GRIP + np.array([p((k + 1) * c.DT) for p, _ in lifts])
        cmd(arc(np.zeros(N), z), yaw_off, F)
        scene.step()
    T2 = max(t for _, t in moves)
    for k in range(int(math.ceil(T2 / c.DT)) + 50):
        s = np.array([p((k + 1) * c.DT) / L for p, _ in moves])
        cmd(arc(s, Z_UP), yaw_off, F)
        scene.step()
    p = part.get_pos().cpu().numpy()
    goal = polar(R, TH1)
    ok = (p[:, 2] > 0.15) & (np.linalg.norm(p[:, :2] - goal, axis=1) < 0.05)
    t = np.array([tl for _, tl in lifts]) + np.array([tm for _, tm in moves])
    return ok, F, t


def score(ok, kF, t):
    return np.where(ok, 1 - 0.3 * kF - 0.1 * t / 2.0, -args.fail_penalty)


def features(mu_hat, m):
    return np.column_stack([np.ones_like(mu_hat), np.log(mu_hat), np.log(m)])


def policy(W, feat):  # W: (3, 3) → kF, a, yaw
    out = feat @ W.T
    kF = np.clip(np.exp(out[:, 0]), 0.5, 3.0)
    a = np.clip(out[:, 1], 0.5, c.A_CAP)
    yaw = np.clip(out[:, 2], -math.pi / 2, math.pi / 2)
    return kF, a, yaw


def sample(n, seed0, stream):
    mats, mu = c.sample_parts(n, seed0, stream=stream)
    rng = np.random.default_rng(seed0 + 7 * stream)
    mu_hat = mu * rng.uniform(0.85, 0.97, n)  # 試し掴みは μ を少し小さめに測る（L3 の実測 −3〜−15%）
    m = np.array([c.mass_of(x) for x in mats])
    return mats, mu, mu_hat, m


rng = np.random.default_rng(args.seed)
t_start = time.time()
log = []
# 方策の係数の分布（平均と広がり）。初めは「式の方策・半径向き」から出発する
mean = np.zeros((3, 3))
mean[1, 0] = c.A_CAP
std = np.array([[0.3, 0.2, 0.2], [2.0, 1.0, 1.0], [0.8, 0.3, 0.3]])
P = args.parts
for g in range(args.gens):
    mats, mu, mu_hat, m = sample(P, 5000 + 100 * g, stream=g)
    Ws = [mean] + [mean + std * rng.standard_normal((3, 3)) for _ in range(args.pop - 1)]
    idx = np.arange(N) % P                      # 環境 i → 部品 i % P
    grp = np.arange(N) // P                     # 環境 i → 方策 i // P（pop 以上は式の方策で埋める）
    kF, a, yaw = np.ones(N), np.full(N, c.A_CAP), np.zeros(N)
    for i, W in enumerate(Ws):
        sl = grp == i
        kF[sl], a[sl], yaw[sl] = policy(W, features(mu_hat[idx[sl]], m[idx[sl]]))
    ok, F, t = episode([mats[j] for j in idx], mu[idx], mu_hat[idx], kF, a, yaw)
    sc = score(ok, kF, t)[: args.pop * P].reshape(args.pop, P).mean(axis=1)
    elite = np.argsort(sc)[-max(2, args.pop // 4):]
    mean = np.mean([Ws[i] for i in elite], axis=0)
    std = np.maximum(np.std([Ws[i] for i in elite], axis=0), 0.02)
    best = elite[-1]
    sl = slice(best * P, (best + 1) * P)
    log.append({"gen": g, "score_best": round(float(sc[best]), 3), "score_mean": round(float(sc.mean()), 3),
                "best_success": round(float(ok[sl].mean()), 3), "best_kF": round(float(kF[sl].mean()), 3),
                "best_a": round(float(a[sl].mean()), 2), "best_yaw_deg": round(float(np.degrees(yaw[sl]).mean()), 1),
                "wall_s": round(time.time() - t_start, 1)})
    print(json.dumps(log[-1]), flush=True)

# 学習に使わなかった部品で、3 つの方策を比べる
T = args.test_parts
mats, mu, mu_hat, m = sample(T, 9000, stream=99)
idx, grp = np.arange(N) % T, np.arange(N) // T  # 環境 i → 部品 i % T、方策 i // T（0 式・半径、1 式・接線、2 学習）
kF, a, yaw = np.ones(N), np.full(N, c.A_CAP), np.zeros(N)
yaw[grp == 1] = math.pi / 2
kF[grp == 2], a[grp == 2], yaw[grp == 2] = policy(mean, features(mu_hat[idx[grp == 2]], m[idx[grp == 2]]))
ok, F, t = episode([mats[j] for j in idx], mu[idx], mu_hat[idx], kF, a, yaw)
res = {}
for name, sl in [("formula_radial", slice(0, T)), ("formula_tangent", slice(T, 2 * T)), ("learned", slice(2 * T, 3 * T))]:
    res[name] = {"success": f"{int(ok[sl].sum())}/{T}", "F_median_N": round(float(np.median(F[sl])), 2),
                 "kF_mean": round(float(kF[sl].mean()), 2), "a_mean": round(float(a[sl].mean()), 2),
                 "yaw_deg_mean": round(float(np.degrees(yaw[sl]).mean()), 1), "t_mean_s": round(float(t[sl].mean()), 2),
                 "score": round(float(score(ok[sl], kF[sl], t[sl]).mean()), 3)}
    print(name, json.dumps(res[name]), flush=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"log": log, "test": res, "W": mean.tolist()}, open(args.out, "w"), indent=1)
print(f"wall {time.time() - t_start:.0f} s")
