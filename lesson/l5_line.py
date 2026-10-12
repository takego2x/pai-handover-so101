"""L5・L6: 第一層（試し掴みで推定する腕）→ 置き台 → 第二層（データで一発で運ぶ腕）→ 箱。

  第一層の腕 k は、自分の部品を L3 と同じ手順で試し掴みし、データ（重さ・μ・運ぶ力）を作り、部品を置き台 k に置く。
  第二層の腕 B は、置き台を順に回り、受け取ったデータの力で掴んで、根元の周りを弧を描いて箱へ運ぶ。
  時間は工程ごとに数え、「第一層の 1 回 ÷ 第二層の 1 部品」から、第二層 1 台に第一層が何台要るかを出す。

使い方:
  python lesson/l5_line.py --cells 1            # L5: 第一層 1 台 → 第二層 1 台（受け渡しを見る）
  python lesson/l5_line.py --cells 3 [n_envs]   # L6: 第一層 3 台 → 第二層 1 台（材質はランダム）
  --nodata : 第二層がデータを使わず、60 N と最悪の加速度で運ぶ
わざと失敗（--mistake）:
  diagonal_retract : 置いた後、指を真上に抜かずに斜めに戻る
  world_yaw        : 第一層の腕の指の向きを、腕の向きではなく世界の向きに固定する
  b_seed_current   : 第二層の IK を「いまの姿勢」から解く（根元を目標に向けた姿勢から解かない）
  b_yaw_world      : 第二層が運ぶ間、指の向きを世界に対して固定する（腕と一緒に回さない。指は半径の向き）
  b_grip_radial    : 第二層の指を閉じる向きを、回る向き（接線）ではなく半径の向きにする
"""
import argparse
import json
import math
import time

import numpy as np
import torch

import common as c

ap = argparse.ArgumentParser()
ap.add_argument("n_envs", nargs="?", type=int, default=10)
ap.add_argument("--cells", type=int, choices=[1, 3], default=3)
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--trials", type=int, default=7)
ap.add_argument("--nodata", action="store_true")
ap.add_argument("--mistake", choices=["diagonal_retract", "world_yaw", "b_seed_current", "b_yaw_world", "b_grip_radial"])
ap.add_argument("--json")
args = ap.parse_args()
c.quiet()
MIS = args.mistake
N, K = args.n_envs, args.cells

# ---- 配置（第二層 B の根元が原点）。置き台は B から 0.55 m、第一層の腕は 1.15 m の所から B のほうを向く ----
TH = np.radians([-15.0]) if K == 1 else np.radians([-60.0, -15.0, 30.0])
DROP_TH = np.radians(110.0)
STAND_H, STAND_W, Z_UP = 0.06, 0.03, 0.35  # 置き台は部品（4 cm）より小さい 3 cm 角


def polar(r, th):
    return np.array([r * math.cos(th), r * math.sin(th)])


stands = [polar(0.55, t) for t in TH]
bases = [polar(1.15, t) for t in TH]
picks = [bases[k] + 0.5 * polar(1, TH[k] + math.pi) + 0.25 * polar(1, TH[k] + math.pi / 2) for k in range(K)]
tangent = polar(1, DROP_TH + math.pi / 2)
drops = [polar(0.50, DROP_TH) + ((k - (K - 1) / 2) * 0.10) * tangent for k in range(K)]
yaw_a = [0.0 if MIS == "world_yaw" else TH[k] + math.pi for k in range(K)]  # 第一層の指の向き
part_yaw = [0.0 if MIS == "world_yaw" else math.degrees(TH[k]) for k in range(K)]  # 部品はその向きに置く
B_OFF = 0.0 if MIS in ("b_grip_radial", "b_yaw_world") else math.pi / 2  # 第二層の指を、根元から見た方向から何度回すか

# ---- 部品：台ごと・環境ごとに材質と本当の μ を抽選 ----
mats, true_mu = [], []
for k in range(K):
    m_, mu_ = c.sample_parts(N, args.seed0, stream=k + 1)
    mats.append(m_)
    true_mu.append(mu_)
true_mu = np.array(true_mu)

scene = c.make_scene()
scene.add_entity(c.gs.morphs.Plane())
for s in stands:
    scene.add_entity(c.gs.morphs.Box(size=(STAND_W, STAND_W, STAND_H), pos=(*s, STAND_H / 2), fixed=True))
parts = [c.add_part(scene, picks[k], part_yaw[k]) for k in range(K)]
arms = [c.add_franka(scene, bases[k], math.degrees(TH[k]) + 180) for k in range(K)]
arm_b = c.add_franka(scene)
scene.build(n_envs=N)
m_true = np.array([c.set_parts(parts[k], mats[k], true_mu[k]) for k in range(K)])
for a in arms + [arm_b]:
    c.setup_franka(a, N)
z_pick, z_stand = c.SIDE / 2 + 0.115, STAND_H + c.SIDE / 2 + 0.115
clock = {"A": 0, "B": 0}


def step(n, who):
    for _ in range(n):
        scene.step()
    clock[who] += n


# ---- 第一層の動き（K 台を同時に動かす）----
def cmd_a(k, pos, f=None):
    c.command(arms[k], pos, N, force=f, yaw=yaw_a[k])


def glide_a(p_from, p_to, f, n=100, settle=50):
    for t in np.linspace(0, 1, n):
        for k in range(K):
            cmd_a(k, p_from[k] + (p_to[k] - p_from[k]) * t, None if f is None else f[k])
        step(1, "A")
    step(settle, "A")


def close_a(pos, f, n_soft=20, n_ramp=20, n_hold=20):
    for k in range(K):
        cmd_a(k, pos[k], np.minimum(f[k], 5.0))
    step(n_soft, "A")
    for j in range(n_ramp):
        for k in range(K):
            f0 = np.minimum(f[k], 5.0)
            cmd_a(k, pos[k], f0 + (f[k] - f0) * (j + 1) / n_ramp)
        step(1, "A")
    step(n_hold, "A")


def up(xy, z):
    return np.array([*xy, z])


# ---- 第二層の動き（B は 1 台）----
def cmd_b(pos, f=None, yaw_fixed=None):
    pos = np.broadcast_to(np.asarray(pos, float), (N, 3))
    az = np.arctan2(pos[:, 1], pos[:, 0])  # 根元から見た目標の方向
    yaw = (yaw_fixed if yaw_fixed is not None else az) + B_OFF
    if MIS == "b_seed_current":
        seed = None
    else:  # 根元を目標の方向に向けた構えの姿勢から IK を解く（肘が裏返った解に落ちない）
        seed = np.tile(c.HOME, (N, 1))
        seed[:, 0] = az
        seed[:, 6] = c.HOME[6] - B_OFF
        seed = torch.tensor(seed, dtype=c.gs.tc_float, device=c.gs.device)
    c.command(arm_b, pos, N, force=f, yaw=yaw, seed=seed)


def arc(p0, p1, t):  # 根元の周りを回る経路（半径と角度を直線的に変える）。t は (N,) か スカラー
    t = np.asarray(t, float)
    r0, r1 = np.linalg.norm(p0[:2]), np.linalg.norm(p1[:2])
    a0, a1 = math.atan2(p0[1], p0[0]), math.atan2(p1[1], p1[0])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    r, ang = r0 + (r1 - r0) * t, a0 + da * t
    return np.stack([r * np.cos(ang), r * np.sin(ang), p0[2] + (p1[2] - p0[2]) * t], axis=-1)


def glide_b(p0, p1, f=None, n=100, settle=50, yaw_fixed=None):
    for t in np.linspace(0, 1, n):
        cmd_b(arc(p0, p1, t), f, yaw_fixed)
        step(1, "B")
    step(settle, "B")


def carry_b(p0, p1, a, f, yaw_fixed=None):  # 台形の速さで弧を運ぶ。手先の加速度を測って返す
    L = float(np.linalg.norm(np.diff(arc(p0, p1, np.linspace(0, 1, 101)), axis=0), axis=1).sum())
    fns = [c.trapezoid(L, ai) for ai in a]
    T = max(t for _, t in fns)
    trace = []
    for i in range(int(math.ceil(T / c.DT)) + 30):
        s = np.array([fn((i + 1) * c.DT) / L for fn, _ in fns])
        cmd_b(arc(p0, p1, s), f, yaw_fixed)
        step(1, "B")
        trace.append(arm_b.get_link("hand").get_pos().cpu().numpy().copy())
    tr = np.array(trace)
    return np.linalg.norm(np.diff(tr, 2, axis=0) / c.DT**2, axis=2).max(axis=0)


t0 = time.time()
home_up = [up(picks[k], Z_UP) for k in range(K)]
for k in range(K):
    cmd_a(k, home_up[k])
cmd_b(up(polar(0.4, DROP_TH), Z_UP))
step(150, "A")
clock["A"] = 0

# ================= 第一層：試し掴み（L3 と同じ手順）=================
glide_a(home_up, [up(picks[k], z_pick) for k in range(K)], None)
pick_pos = [up(picks[k], z_pick) for k in range(K)]
lift_pos = [up(picks[k], z_pick + 0.02) for k in range(K)]
full = [np.full(N, c.F_MAX)] * K
close_a(pick_pos, full, n_ramp=50, n_hold=30)
glide_a(pick_pos, lift_pos, full, n=50, settle=50)
readings = []
for _ in range(20):
    step(1, "A")
    readings.append([parts[k].get_links_net_contact_force().cpu().numpy()[:, 0, 2] for k in range(K)])
m_hat = np.mean(readings, axis=0) / c.G
weighed = np.array([parts[k].get_pos()[:, 2].cpu().numpy() > c.SIDE / 2 + 0.01 for k in range(K)])
lo, hi = np.full((K, N), 0.3), np.full((K, N), c.F_MAX)
n_drops = np.zeros((K, N), int)
t_trials0 = clock["A"]
for j in range(args.trials):
    F = np.sqrt(lo * hi)
    glide_a(lift_pos, pick_pos, None, n=10, settle=20)
    close_a(pick_pos, list(F))
    glide_a(pick_pos, lift_pos, list(F), n=30, settle=40)
    held = np.array([parts[k].get_pos()[:, 2].cpu().numpy() > c.SIDE / 2 + 0.01 for k in range(K)])
    hi, lo = np.where(held, F, hi), np.where(held, lo, F)
    n_drops += ~held
t_trials = (clock["A"] - t_trials0) * c.DT
mu_hat = m_hat * c.G / (2 * hi)
F_data = np.where(weighed, c.carry_force(m_hat, mu_hat), c.F_MAX)
a_data = np.where(weighed, c.accel_for(m_hat, mu_hat, F_data), 0.5)
a_data = np.where(a_data > 0, a_data, 0.5)

# 掴み直して、自分のデータで置き台へ
glide_a(lift_pos, pick_pos, None, n=10, settle=20)
close_a(pick_pos, list(F_data))
# 第一層の運び：持ち上げ → 置き台の上へ（直線。加速度はデータから）
seg = [(pick_pos, home_up), (home_up, [up(stands[k], Z_UP) for k in range(K)])]
for p0, p1 in seg:
    fns = [[c.trapezoid(float(np.linalg.norm(p1[k] - p0[k])), ai) for ai in a_data[k]] for k in range(K)]
    T = max(t for f in fns for _, t in f)
    for i in range(int(math.ceil(T / c.DT)) + 30):
        for k in range(K):
            d = p1[k] - p0[k]
            s = np.array([fn((i + 1) * c.DT) / max(np.linalg.norm(d), 1e-9) for fn, _ in fns[k]])
            cmd_a(k, p0[k] + np.outer(s, d), F_data[k])
        step(1, "A")
stand_up = [up(stands[k], Z_UP) for k in range(K)]
stand_dn = [up(stands[k], z_stand) for k in range(K)]
glide_a(stand_up, stand_dn, list(F_data))
for k in range(K):
    cmd_a(k, stand_dn[k])  # 指を開いて置く
step(60, "A")
if MIS == "diagonal_retract":
    glide_a(stand_dn, home_up, None)
else:  # 真上に抜いてから戻る
    glide_a(stand_dn, stand_up, None)
    glide_a(stand_up, home_up, None)
cycle_a = clock["A"] * c.DT
on_stand = np.array([np.linalg.norm(parts[k].get_pos().cpu().numpy()[:, :2] - stands[k], axis=1) < 0.01 for k in range(K)])

# ================= 第二層：データで一発で運ぶ =================
if args.nodata:
    F_b = np.full((K, N), c.F_MAX)
    m_w = max(c.mass_of(m) for m in c.POOL)
    mu_w = min(c.mu_range(m, rows=c.load())[0] for m in c.POOL)
    a_b = np.full((K, N), float(c.accel_for(m_w, mu_w, c.F_MAX)))
else:
    F_b, a_b = F_data, a_data
success = np.zeros((K, N), bool)
a_peak = np.zeros((K, N))
t_b = []
b_from = up(polar(0.4, DROP_TH), Z_UP)
for k in range(K):
    c0 = clock["B"]
    yf = TH[k] if MIS == "b_yaw_world" else None  # 部品の向き（置き台の方向）に世界で固定するか
    glide_b(b_from, up(stands[k], Z_UP), yaw_fixed=yf)
    glide_b(up(stands[k], Z_UP), up(stands[k], z_stand), yaw_fixed=yf)
    cmd_b(up(stands[k], z_stand), np.minimum(F_b[k], 5.0), yf)
    step(20, "B")
    for j in range(20):
        f0 = np.minimum(F_b[k], 5.0)
        cmd_b(up(stands[k], z_stand), f0 + (F_b[k] - f0) * (j + 1) / 20, yf)
        step(1, "B")
    step(20, "B")
    a_peak[k] = np.maximum(carry_b(up(stands[k], z_stand), up(stands[k], Z_UP), a_b[k], F_b[k], yf),
                           carry_b(up(stands[k], Z_UP), up(drops[k], Z_UP), a_b[k], F_b[k], yf))
    glide_b(up(drops[k], Z_UP), up(drops[k], z_pick), F_b[k], yaw_fixed=yf)
    cmd_b(up(drops[k], z_pick), None, yf)
    step(60, "B")
    glide_b(up(drops[k], z_pick), up(drops[k], Z_UP), yaw_fixed=yf)
    b_from = up(drops[k], Z_UP)
    p = parts[k].get_pos().cpu().numpy()
    success[k] = on_stand[k] & (np.linalg.norm(p[:, :2] - drops[k], axis=1) < 0.03) & (p[:, 2] < c.SIDE)
    t_b.append((clock["B"] - c0) * c.DT)
cycle_b = float(np.mean(t_b))

# ================= 結果 =================
flat_m = np.array(mats)
print(f"cells={K} envs={N} nodata={args.nodata} mistake={MIS}  (wall {time.time() - t0:.0f} s)")
print(f"{'material':9s} {'n':>3s} {'weight err':>10s} {'mu err':>13s} {'drops':>5s} {'F layer2':>9s} {'peak a':>7s} {'stand':>6s} {'done':>6s}")
for m in c.POOL:
    sel = flat_m == m
    if not sel.any():
        continue
    w = weighed & sel
    we = np.abs(m_hat[w] / m_true[w] - 1).max() * 100 if w.any() else float("nan")
    me = (mu_hat[w] / true_mu[w] - 1) * 100 if w.any() else np.array([np.nan])
    print(f"{m:9s} {int(sel.sum()):3d} {we:9.1f}% {me.min():+5.0f}..{me.max():+4.0f}% {n_drops[sel].mean():5.1f} "
          f"{np.median(F_b[sel]):7.1f}N {a_peak[sel].max():6.1f} {int(on_stand[sel].sum()):3d}/{int(sel.sum()):<2d} "
          f"{int(success[sel].sum()):3d}/{int(sel.sum())}")
need = cycle_a / cycle_b
print(f"layer1 cycle {cycle_a:.1f} s (trials {t_trials:.1f} s)   layer2 cycle {cycle_b:.1f} s per part   "
      f"ratio {need:.2f} -> layer1 arms {math.ceil(need)}")
print(f"on stand {int(on_stand.sum())}/{K * N}   delivered {int(success.sum())}/{K * N}   "
      f"peak hand accel of layer2 {a_peak.max():.1f} m/s^2")
if args.json:
    json.dump({"cells": K, "nodata": args.nodata, "mistake": MIS, "materials": mats,
               "true_mu": true_mu.round(4).tolist(), "m_true": m_true.round(4).tolist(),
               "m_hat": m_hat.round(4).tolist(), "mu_hat": mu_hat.round(4).tolist(), "drops": n_drops.tolist(),
               "F_layer2": F_b.round(3).tolist(), "on_stand": on_stand.tolist(), "success": success.tolist(),
               "cycle_layer1_s": round(cycle_a, 2), "cycle_layer2_s": round(cycle_b, 2),
               "trials_s": round(t_trials, 2)}, open(args.json, "w"), indent=1)
