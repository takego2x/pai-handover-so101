"""L5: L3 が書いたデータ（重さ・μ）を受け取り、部品ごとに決めた力と加速度で一発で運ぶ。

運び方: 掴む → 21.5 cm 持ち上げる → 40 cm 横へ運ぶ → 1 秒保持して、まだ持っているかを見る。
持ち上げと横移動は、加速度 a の台形の速さ（common.trapezoid）。
比べるもの（--mode）:
  data       : L3 のデータ。力 = carry_force(重さ, μ)、加速度 = 上限 9 m/s^2
  nodata     : データなし。どの材質でも落とさないよう 60 N で握り、最悪（最も重く、文献で最も滑りやすい）から加速度を決める
  literature : 材質は正しく分かっているが、μ は文献値の下限、重さは密度から計算（試し掴みをしない）
使い方: python lesson/carry.py [--mode data|nodata|literature] [--mu-scale 1.0] [--mass-scale 1.0]
  --mu-scale / --mass-scale : 受け取ったデータの μ・重さを何倍かにずらす（データの精度がどこまで要るかを見る）
"""
import argparse
import json
import math

import numpy as np

import common as c

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default=str(c.ROOT / "outputs" / "lesson" / "parts.json"))
ap.add_argument("--mode", choices=["data", "nodata", "literature"], default="data")
ap.add_argument("--mu-scale", type=float, default=1.0)
ap.add_argument("--mass-scale", type=float, default=1.0)
ap.add_argument("--json")
args = ap.parse_args()
c.quiet()

P = json.load(open(args.parts))
data, truth = P["data"], P["truth"]
N = len(truth)
mats = [t["material"] for t in truth]
true_mu = np.array([t["mu"] for t in truth])
rows = c.load()

# 計画に使う重さと μ
if args.mode == "data":
    m_plan = np.array([d["mass_kg"] for d in data])
    mu_plan = np.array([d["mu"] for d in data])
    measured = np.array([d["ok"] for d in data])
elif args.mode == "nodata":
    m_plan = np.full(N, max(c.mass_of(m) for m in c.POOL))
    mu_plan = np.full(N, min(c.mu_range(m, rows=rows)[0] for m in c.POOL))
    measured = np.ones(N, bool)
else:
    m_plan = np.array([c.mass_of(m) for m in mats])
    mu_plan = np.array([c.mu_range(m, rows=rows)[0] for m in mats])
    measured = np.ones(N, bool)
m_plan, mu_plan = m_plan * args.mass_scale, mu_plan * args.mu_scale
F = np.where(measured, c.carry_force(m_plan, mu_plan), c.F_MAX) if args.mode != "nodata" else np.full(N, c.F_MAX)
a = c.accel_for(m_plan, mu_plan, F)
a = np.where(a > 0, a, 0.5)  # 計画上は持てない部品も、ゆっくり試す（落ちれば失敗として数える）

START, GOAL, Z_UP = np.array([0.55, -0.20]), np.array([0.55, 0.20]), 0.35
scene = c.make_scene()
scene.add_entity(c.gs.morphs.Plane())
part = c.add_part(scene, START)
arm = c.add_franka(scene)
scene.build(n_envs=N)
m_true = c.set_parts(part, mats, true_mu)
c.setup_franka(arm, N)
z_grip = c.SIDE / 2 + 0.115


def go(xy, z, n, f=None):
    xy = np.broadcast_to(xy, (N, 2))
    z = np.broadcast_to(z, (N,))
    c.command(arm, np.column_stack([xy, z]), N, force=f)
    for _ in range(n):
        scene.step()


def follow(path, total):  # 0.01 秒ごとに、環境ごとの目標位置へ進める。手先の位置を記録する
    trace = []
    for k in range(int(math.ceil(total.max() / c.DT))):
        xy, z = path((k + 1) * c.DT)
        go(xy, z, 1, F)
        trace.append(arm.get_link("hand").get_pos().cpu().numpy().copy())
    return np.array(trace)


go(START, 0.25, 150)
go(START, z_grip, 150)
f0 = np.minimum(F, 5.0)
go(START, z_grip, 50, f0)
for k in range(50):
    go(START, z_grip, 1, f0 + (F - f0) * (k + 1) / 50)

lifts = [c.trapezoid(Z_UP - z_grip, ai) for ai in a]
t_lift = np.array([t for _, t in lifts])
tr1 = follow(lambda t: (START, z_grip + np.array([p(t) for p, _ in lifts])), t_lift)
go(START, Z_UP, 30, F)
dist = float(np.linalg.norm(GOAL - START))
moves = [c.trapezoid(dist, ai) for ai in a]
t_move = np.array([t for _, t in moves])
u = (GOAL - START) / dist
tr2 = follow(lambda t: (START + np.outer([p(t) for p, _ in moves], u), Z_UP), t_move)
go(GOAL, Z_UP, 100, F)

p = part.get_pos().cpu().numpy()
success = (p[:, 2] > 0.15) & (np.linalg.norm(p[:, :2] - GOAL, axis=1) < 0.05)
acc = np.concatenate([np.linalg.norm(np.diff(tr, 2, axis=0) / c.DT**2, axis=2) for tr in (tr1, tr2)], axis=0)
a_meas = np.percentile(acc, 95, axis=0)
# 余裕：本当の μ・本当の重さで、使える摩擦力 2 μ F ÷ 必要な摩擦力 m (g + a) − 1（a は実際に出た加速度）
margin = 2 * true_mu * F / (m_true * (c.G + a_meas)) - 1
t = t_lift + t_move
print(f"mode={args.mode}  mu_scale={args.mu_scale}  mass_scale={args.mass_scale}")
print(f"{'#':>2s} {'material':9s} {'mu true':>7s} {'mu plan':>7s} {'F [N]':>6s} {'a plan':>6s} {'a real':>6s} {'time':>6s} {'margin':>7s} result")
for i in range(N):
    print(f"{i:2d} {mats[i]:9s} {true_mu[i]:7.3f} {mu_plan[i]:7.3f} {F[i]:6.2f} {a[i]:6.2f} {a_meas[i]:6.2f} {t[i]:5.2f}s "
          f"{margin[i] * 100:+6.0f}% {'held' if success[i] else 'DROPPED'}")
print(f"success {int(success.sum())}/{N}   force median {np.median(F):.1f} N (max {F.max():.1f})   "
      f"lift+move mean {t.mean():.2f} s   worst margin {margin.min() * 100:+.0f}%")
if args.json:
    json.dump({"mode": args.mode, "mu_scale": args.mu_scale, "mass_scale": args.mass_scale, "materials": mats,
               "F": F.round(3).tolist(), "success": success.tolist(), "t": t.round(3).tolist(),
               "margin": margin.round(4).tolist()}, open(args.json, "w"), indent=1)
