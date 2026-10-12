"""L3: 失敗してよい試し掴みで「持てる最小の力（保持力）」を絞り、重さ・μ・運ぶ力をデータとして書き出す。

1 つの部品について:
  1. 60 N で持ち上げて、重さを測る（L2）
  2. 力 F で掴んで 2 cm 持ち上げる試行を n 回。持てたら F を下げ、落ちたら上げる（落とすのは許容）
     F は「持てた力 hi」と「落ちた力 lo」の掛け算の真ん中 sqrt(lo * hi)（対数で半分ずつ絞る）
  3. 保持力 F_hold = hi。保持の境目 2 μ F = m g から  μ = m g / (2 F_hold)
  4. 運ぶ力 = 重さと μ から、加速度 9 m/s^2 で持ち上げても余裕 15% で落とさない力（common.carry_force）
書き出すファイル outputs/lesson/parts.json の "data" が、次の工程（L4）に渡すデータ。"truth" は採点用でロボットは見ない。
使い方: python lesson/l3_trials.py [n] [--trials 7] [--mistake slam | no_home]
  --mistake slam    : 重さを測るとき、指をそっと閉じず、いきなり 60 N で閉じる
  --mistake no_home : 全環境を同じ構えの姿勢から始めない（Genesis の初期の姿勢のまま IK を解く）
  --material steel --mu 0.3,0.5,0.1 : 材質と本当の μ を自分で並べる（n は μ の個数になる）
L8（E5）: --prior none|vlm|exact を付けると、探す範囲を事前の情報で狭め、必要な精度（--stop、範囲の幅の比）に
  届くまで試す（部品ごとに回数が違う）。vlm は Gemini が画像から挙げた材質の候補（data/vlm_candidates_gemini.json）、
  exact は正しい材質 1 つ（AI が完璧だった場合）、none は全範囲（0.3〜60 N）
"""
import argparse
import json
import time

import numpy as np

import common as c

ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=15)
ap.add_argument("--trials", type=int, default=7)
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--mistake", choices=["slam", "no_home"])
ap.add_argument("--out", default=str(c.ROOT / "outputs" / "lesson" / "parts.json"))
ap.add_argument("--material", help="全部品をこの材質にする（--mu と一緒に使う）")
ap.add_argument("--prior", choices=["none", "vlm", "exact"], help="事前の情報で探す範囲を狭める（L8、E5）")
ap.add_argument("--stop", type=float, default=1.05, help="--prior のとき、範囲の幅（hi / lo）がこれ以下になったら止める")
ap.add_argument("--widen", type=float, default=1.5, help="--prior のとき、文献の μ の幅をさらに何倍に広げて使うか")
ap.add_argument("--mu", help="本当の μ を環境の順にカンマ区切りで指定する（例: 0.3,0.5,0.1）。並べ替えの切り分け用")
args = ap.parse_args()
c.quiet()

if args.mu:  # 条件を自分で並べる（並べ替えて、失敗が条件についていくか、環境の番号についていくかを見る）
    true_mu = np.array([float(x) for x in args.mu.split(",")])
    N = len(true_mu)
    mats = [args.material or "steel"] * N
else:
    N = args.n
    mats, true_mu = c.sample_parts(N, args.seed0)
XY = np.array([0.55, 0.0])
F_WEIGH, F_LO = 60.0, 0.3

scene = c.make_scene()
scene.add_entity(c.gs.morphs.Plane())
part = c.add_part(scene, XY)
arm = c.add_franka(scene)
scene.build(n_envs=N)
m_true = c.set_parts(part, mats, true_mu)
c.setup_franka(arm, N, home=args.mistake != "no_home")
z_grip = c.SIDE / 2 + 0.115
steps = 0


def go(z, n, f=None):
    global steps
    c.command(arm, [*XY, z], N, force=f)
    for _ in range(n):
        scene.step()
    steps += n


def close_gently(f):  # 5 N でそっと閉じてから、0.2 秒で目標の力 f まで上げる
    f0 = np.minimum(f, 5.0)
    go(z_grip, 20, f0)
    for k in range(20):
        go(z_grip, 1, f0 + (f - f0) * (k + 1) / 20)
    go(z_grip, 20, f)


def lift(f, dz=0.02, n=30):  # 滑らかに dz 持ち上げて、0.4 秒止める
    for k in range(n):
        s = (k + 1) / n
        go(z_grip + dz * (3 * s**2 - 2 * s**3), 1, f)
    go(z_grip + dz, 40, f)


def z_part():
    return part.get_pos()[:, 2].cpu().numpy()


t0 = time.time()
go(0.25, 150)
go(z_grip, 150)
steps = 0  # ここから 1 部品あたりの時間を数える
# 1. 重さを測る
full = np.full(N, F_WEIGH)
if args.mistake == "slam":
    go(z_grip, 60, full)
else:
    close_gently(full)
lift(full)
readings = []
for _ in range(20):
    go(z_grip + 0.02, 1, full)
    readings.append(part.get_links_net_contact_force().cpu().numpy()[:, 0, 2])
m_hat = np.mean(readings, axis=0) / c.G
weighed = z_part() > c.SIDE / 2 + 0.01  # 60 N で持ち上がったか（持ち上がらなければ、重さも測れていない）

# 2. 失敗してよい試行
lo, hi = np.full(N, F_LO), np.full(N, F_WEIGH)
drops = np.zeros(N, int)
n_trials = np.full(N, args.trials)
cands = [None] * N
if args.prior:
    # 事前の情報：材質の候補 → 文献の μ の幅（さらに --widen 倍に広げる）→ 測った重さで、探す力の範囲に直す
    rows = c.load()
    vlm = json.load(open(c.ROOT / "data" / "vlm_candidates_gemini.json"))["per_image"]
    for i in range(N):
        if args.prior == "vlm":
            cands[i] = vlm[mats[i]][i % len(vlm[mats[i]])]  # Gemini がその材質の部品に挙げた候補（画像 10 枚を順に）
        elif args.prior == "exact":
            cands[i] = [mats[i]]
        if cands[i]:
            r = [c.mu_range(m, rows=rows) for m in cands[i]]
            mu_lo, mu_hi = min(x[0] for x in r) / args.widen, max(x[1] for x in r) * args.widen
            hi[i] = min(F_WEIGH, m_hat[i] * c.G / (2 * mu_lo))  # μ が最も小さいときに要る力（ここなら持てるはず）
            lo[i] = max(F_LO, m_hat[i] * c.G / (2 * mu_hi))     # μ が最も大きいときに要る力（ここでは落ちるはず）
    held_once = np.zeros(N, bool)  # 範囲の中で一度でも持てたか（持てないまま終わったら、範囲が外れていた）
    active = np.ones(N, bool)
    n_trials = np.zeros(N, int)
    for j in range(15):
        F = np.where(active, np.sqrt(lo * hi), hi)  # 終わった部品は、持てる力で掴むだけ（数に入れない）
        go(z_grip, 30)
        close_gently(F)
        lift(F)
        held = z_part() > c.SIDE / 2 + 0.01
        hi = np.where(active & held, F, hi)
        lo = np.where(active & ~held, F, lo)
        held_once |= active & held
        drops += active & ~held
        n_trials += active
        done = active & (hi / lo <= args.stop)
        # 範囲が外れていた（一度も持てずに幅が縮んだ）→ 上を全範囲に戻して続ける
        reopen = done & ~held_once & (hi < F_WEIGH)
        hi = np.where(reopen, F_WEIGH, hi)
        active = active & ~(done & ~reopen)
        if not active.any():
            break
else:
    for j in range(args.trials):
        F = np.sqrt(lo * hi)
        go(z_grip, 30)  # 下ろして指を開く（前の試行で落ちた部品は、もう床の上）
        close_gently(F)
        lift(F)
        held = z_part() > c.SIDE / 2 + 0.01
        hi, lo = np.where(held, F, hi), np.where(held, lo, F)
        drops += ~held
go(z_grip, 30)  # 下ろして放す
t_part = steps * c.DT
TRIAL_S = 1.6  # 試行 1 回の時間（下ろす 0.3 + 閉じる 0.6 + 持ち上げて止める 0.7 秒）
t_each = t_part - (n_trials.max() - n_trials) * TRIAL_S if args.prior else np.full(N, t_part)  # 部品ごとの時間
# 3. データにする
F_hold = hi
mu_hat = m_hat * c.G / (2 * F_hold)
F_carry = c.carry_force(m_hat, mu_hat)
ok = weighed
print(f"mistake={args.mistake}  trials={args.trials}  time per part {t_part:.1f} s  (wall {time.time() - t0:.0f} s)")
print(f"{'#':>2s} {'material':9s} {'weight':>14s} {'mu true->est':>16s} {'drops':>5s} {'F_hold':>7s} {'F_carry':>7s}")
for i in range(N):
    w = f"{m_hat[i] * 1000:6.1f} g {(m_hat[i] / m_true[i] - 1) * 100:+5.1f}%" if ok[i] else "  not lifted  "
    mu = f"{true_mu[i]:.3f}->{mu_hat[i]:.3f} {(mu_hat[i] / true_mu[i] - 1) * 100:+5.0f}%" if ok[i] else "-"
    print(f"{i:2d} {mats[i]:9s} {w:>14s} {mu:>16s} {drops[i]:5d} {F_hold[i]:6.2f}N {F_carry[i]:6.2f}N")
err = (mu_hat / true_mu - 1)[ok] * 100
print(f"weighed {int(ok.sum())}/{N}   mu error {err.min():+.0f}% .. {err.max():+.0f}%   drops per part {drops.mean():.1f}")
if args.prior:
    for i in range(N):
        print(f"   #{i:2d} {mats[i]:9s} prior={cands[i]}  trials={n_trials[i]}")
    # 第一層の 1 回 = 試行以外の時間（L6 の実測 22.2 - 11.2 = 11.0 秒）+ 試行の回数 × 1.6 秒。第二層の 1 部品は 9.0 秒
    cyc = 11.0 + n_trials.mean() * TRIAL_S
    print(f"prior={args.prior}  trials per part mean {n_trials.mean():.1f} (max {n_trials.max()})   "
          f"time per part mean {t_each.mean():.1f} s   |mu error| median {np.median(np.abs(err)):.1f}%   "
          f"-> layer1 cycle ~{cyc:.1f} s, arms {cyc / 9.0:.2f} -> {int(np.ceil(cyc / 9.0))}")

out = {"trials": args.trials, "time_per_part_s": round(t_part, 2), "seed0": args.seed0, "prior": args.prior,
       "n_trials": n_trials.tolist(), "candidates": cands,
       "data": [{"ok": bool(ok[i]), "mass_kg": round(float(m_hat[i]), 4), "mu": round(float(mu_hat[i]), 4),
                 "F_hold_N": round(float(F_hold[i]), 3), "F_carry_N": round(float(F_carry[i]), 3)} for i in range(N)],
       "truth": [{"material": mats[i], "mass_kg": round(float(m_true[i]), 4), "mu": round(float(true_mu[i]), 4),
                  "drops": int(drops[i])} for i in range(N)]}
c.ROOT.joinpath("outputs", "lesson").mkdir(parents=True, exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print("wrote", args.out)
