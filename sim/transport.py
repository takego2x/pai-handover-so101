"""S8: 掴んで運ぶ。運ぶ加速度は「見積もった μ と重さ」から決め、本当の μ は文献値の幅の外も含めて抽選する。

持ち上げ（上向きに加速度 a）の保持の条件（指 2 本）:  m * (g + a) <= 2 * mu * F
  -> 許される加速度  a_max(mu, m) = 2 mu F / m - g
横に運ぶときの条件 m*sqrt(g^2+a^2) <= 2 mu F はこれより緩いので、同じ a を使えば両方を満たす。

部品はどれも同じ大きさ（SIDE の立方体）で、重さは材質の密度で決まる（POM 90 g、アルミ 173 g、鋼 502 g）。
ロボットは μ と重さの「見積もり」で計画し、評価は「本当の値」で行う。見積もりと本当の値は別の情報源から来る:
  本当の μ  : 文献値の幅を TRUE_WIDEN 倍に広げた範囲から、seed ごとに抽選（文献値を信じても外れる場合を作る）
  見積もり  : 戦略 A/B/C は文献値の表から、戦略 O は本当の値そのもの（＝完全に知っている場合）
  --mu-scale / --mass-scale で見積もりを何倍かにずらす（推定のずれの実験 E1・E2）

使い方: python sim/transport.py <material> <strategy> [n_seeds] [--mu-scale 1.2] [--mass-scale 0.8] [--record out.mp4]
  material: POM / aluminium / steel
  strategy: A (一律・代表値) / B (一律・最悪値) / C (材質別・下限。--candidates で VLM の候補) / O (本当の値を知っている)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mu_table import load, mu_range, safe_mu  # noqa: E402

G = 9.81
FORCE = float(os.environ.get("FORCE", 15.0))  # 指 1 本の把持力 [N]。仮の値（組み替え方針 §3 で決める）
SIDE = 0.04          # 部品の一辺 [m]。全材質で同じ大きさ
# 密度 [kg/m^3]。POM・アルミ・鋼は一般的な値。ABS・NBR は候補に出たときの重さの見積もり用（代表値）
DENSITY = {"POM": 1410.0, "ABS": 1050.0, "aluminium": 2700.0, "steel": 7850.0, "NBR": 1200.0}
SIMULATED = ["POM", "aluminium", "steel"]
A_CAP = float(os.environ.get("A_CAP", 9.0))  # ロボットが実際に出せる加速度 [m/s^2]（実測で決める）
V_MAX = 1.5          # 運ぶ速さの上限 [m/s]
SAFETY = 0.85        # 使える摩擦力のうち計画に使う割合（最低 15% の余裕を残す）
F_MAX = 60.0         # グリッパが出せる把持力の上限 [N]（--min-force のとき）
TRUE_WIDEN = (0.7, 1.3)  # 本当の μ の抽選範囲＝（文献の下限 × 0.7）〜（文献の上限 × 1.3）
START, GOAL = np.array([0.55, -0.20]), np.array([0.55, 0.20])  # 運ぶ区間（x, y）[m]
Z_CARRY = 0.35


def mass_of(material):
    return DENSITY[material] * SIDE**3


MASS = mass_of("POM")  # 旧版との互換（sim/line.py が使う）


def a_max(mu, mass=MASS, force=FORCE):
    return max(0.0, 2 * mu * force / mass - G)


def planned_accel(mu_assumed, mass_assumed=MASS, force=FORCE):
    # m (g + a) <= SAFETY * 2 mu F  ->  a = SAFETY * 2 mu F / m - g（安全率は加速度ではなく摩擦力にかける）
    # 0 なら「この見積もりでは静止して持つこともできない」＝運ばない
    return min(A_CAP, max(0.0, SAFETY * 2 * mu_assumed * force / mass_assumed - G))


def min_force(mu_assumed, mass_assumed, a_target=None):
    """見積もりで、加速度 a_target（既定は A_CAP）を余裕 15% で出せる最小の把持力。上限 F_MAX で頭打ち。"""
    a = A_CAP if a_target is None else a_target
    return min(F_MAX, mass_assumed * (G + a) / (SAFETY * 2 * mu_assumed))


def assumed_mu(strategy, material, rows, candidates=None):
    ranges = {m: mu_range(m, rows=rows) for m in ["POM", "ABS", "aluminium", "steel", "NBR"]}
    if strategy == "A":  # 一律・代表値：材質ごとの幅の中点の中央値を、全部品に使う
        return float(np.median([(lo + hi) / 2 for lo, hi, _ in ranges.values()]))
    if strategy == "B":  # 一律・最悪値：全材質の下限の最小を、全部品に使う
        return min(lo for lo, _, _ in ranges.values())
    if strategy == "C":  # 材質別・下限。candidates（VLM が挙げた材質の候補）があれば、その中の最小の下限
        if candidates:
            mu = safe_mu(candidates, rows=rows)
            return mu if mu is not None else assumed_mu("B", material, rows)  # 値のない候補 → 最悪値に倒す
        return ranges[material][0]
    raise ValueError(strategy)


def assumed_mass(strategy, material, candidates=None):
    """重さの見積もり。迷ったら重いほうに倒す（重く見積もれば遅くなるだけで、落とさない）。"""
    if strategy == "A":  # 一律・代表値：材質ごとの重さの中央値
        return float(np.median([mass_of(m) for m in SIMULATED]))
    if strategy == "B":  # 一律・最悪値：いちばん重い材質
        return max(mass_of(m) for m in DENSITY)
    if strategy == "C":
        if candidates:
            if any(c not in DENSITY for c in candidates):
                return assumed_mass("B", material)  # 密度の分からない候補 → 最悪値に倒す
            return max(mass_of(c) for c in candidates)
        return mass_of(material)
    raise ValueError(strategy)


def trapezoid(dist, a, v):
    """距離 dist を加速度 a・最高速度 v の台形で動くときの時刻→位置の関数と所要時間。"""
    if a <= 0:
        return None, math.inf
    t_acc = v / a
    if a * t_acc**2 >= dist:  # 最高速度に届かない（三角形）
        t_acc = math.sqrt(dist / a)
        v = a * t_acc
        t_flat = 0.0
    else:
        t_flat = (dist - a * t_acc**2) / v
    total = 2 * t_acc + t_flat

    def pos(t):
        t = min(max(t, 0.0), total)
        if t < t_acc:
            return 0.5 * a * t * t
        if t < t_acc + t_flat:
            return 0.5 * a * t_acc**2 + v * (t - t_acc)
        td = total - t
        return dist - 0.5 * a * td * td

    return pos, total


def trapezoids(dist, accels, v):
    """環境ごとに加速度が違う台形。pos(t) は (N,) を返す。運ばない環境（a=0）は 0 のまま。"""
    fns = [trapezoid(dist, a, v) for a in accels]
    totals = np.array([t for _, t in fns])
    return (lambda t: np.array([f(t) if f else 0.0 for f, _ in fns])), totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("material", choices=SIMULATED)
    ap.add_argument("strategy", choices=["A", "B", "C", "O"])
    ap.add_argument("n_seeds", nargs="?", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--force", type=float, default=FORCE, help="指 1 本の把持力 [N]")
    ap.add_argument("--mu-scale", type=float, default=1.0, help="見積もった μ を何倍にずらすか（E1）")
    ap.add_argument("--mass-scale", type=float, default=1.0, help="見積もった重さを何倍にずらすか（E2）")
    ap.add_argument("--min-force", action="store_true",
                    help="部品ごとに、見積もりから必要最小の把持力を決める（上限 F_MAX）。省略時は --force で固定")
    ap.add_argument("--true-mu", type=float, help="本当の μ を全環境でこの値に固定する（抽選しない。理論との突き合わせ用）")
    ap.add_argument("--record", help="env 0 の様子を mp4 に保存する")
    ap.add_argument("--cam", default="1.5,0.0,0.7", help="録画のカメラ位置 x,y,z")
    ap.add_argument("--lookat", default="0.5,0.0,0.2", help="録画のカメラの注視点 x,y,z")
    ap.add_argument("--json", help="結果を JSON で保存する")
    ap.add_argument("--mu-assumed", type=float, help="戦略 A の代表値を上書きする（感度の分析用）")
    ap.add_argument("--candidates", help="戦略 C に渡す材質の候補（カンマ区切り。例: POM,ABS）。省略時は正解の材質")
    args = ap.parse_args()

    rows = load()
    lo, hi, sources = mu_range(args.material, rows=rows)
    N = args.n_seeds
    F = args.force
    seeds = list(range(args.seed0, args.seed0 + N))
    if args.true_mu is not None:
        true_mu = np.full(N, args.true_mu)
    else:  # 本当の μ は、文献値の幅を広げた範囲から抽選（文献の外にある部品も混じる）
        true_mu = np.array([np.random.default_rng(s).uniform(lo * TRUE_WIDEN[0], hi * TRUE_WIDEN[1]) for s in seeds])
    mass_true = mass_of(args.material)
    candidates = args.candidates.split(",") if args.candidates else None

    if args.strategy == "O":  # 本当の値を知っている（ずれは --mu-scale / --mass-scale だけ）
        mu_hat = true_mu.copy()
        m_hat = mass_true
    else:
        mu_hat = np.full(N, assumed_mu(args.strategy, args.material, rows, candidates))
        m_hat = assumed_mass(args.strategy, args.material, candidates)
    if args.mu_assumed is not None:
        if args.strategy != "A":
            sys.exit("--mu-assumed は戦略 A（一律）だけで使う")
        mu_hat = np.full(N, args.mu_assumed)
    mu_hat = mu_hat * args.mu_scale
    m_hat = m_hat * args.mass_scale
    if args.min_force:  # 見積もりが正しければ、最大の加速度を最小の力で出せる
        Fv = np.array([min_force(float(m), m_hat) for m in mu_hat])
    else:
        Fv = np.full(N, F)
    a_plan = np.array([planned_accel(float(m), m_hat, f) for m, f in zip(mu_hat, Fv)])
    refused = a_plan <= 0  # 見積もりでは静止して持つこともできない → 運ばない（失敗として数える）

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
                     rigid_options=gs.options.RigidOptions(box_box_detection=True, friction_cone=gs.friction_cone.elliptic, impratio=10.0), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(gs.morphs.Box(size=(SIDE, SIDE, SIDE), pos=(*START, SIDE / 2)),
                            material=gs.materials.Rigid(friction=1.0, rho=DENSITY[args.material]))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
                              material=gs.materials.Rigid(friction=0.01))
    cam = None
    if args.record:
        cam = scene.add_camera(res=(960, 540), pos=tuple(map(float, args.cam.split(","))),
                               lookat=tuple(map(float, args.lookat.split(","))), fov=40, GUI=False)
    scene.build(n_envs=N)
    cube.set_friction_ratio(torch.tensor(true_mu, dtype=gs.tc_float, device=gs.device).unsqueeze(1),
                            links_idx_local=[0])
    mass_sim = float(cube.get_mass())
    if abs(mass_sim - mass_true) > 0.005 * mass_true:  # シミュレータの中の重さが、計算した重さと合っているか
        sys.exit(f"mass mismatch: sim {mass_sim:.4f} kg vs density x volume {mass_true:.4f} kg")

    motors, fingers = np.arange(7), np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                                np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
    hand = franka.get_link("hand")
    quat = np.tile([0, 1, 0, 0], (N, 1))
    grip_z = SIDE / 2 + 0.115  # 手先（hand リンク）から指先の中心までが約 0.11 m
    # 全環境を同じ姿勢から始め、IK もいまの姿勢から解く。初期の姿勢のままだと IK が環境ごとに違う解を返し、
    # 腕が回り込んで部品を払う環境が出る（2026-10-12、spikes/slip_probe.py で 2 番目の環境だけ部品が飛んだ）
    HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
    franka.set_dofs_position(np.tile(HOME, (N, 1)))
    franka.control_dofs_position(np.tile(HOME, (N, 1)))

    def ik(pos):
        return franka.inverse_kinematics(link=hand, pos=pos, quat=quat, init_qpos=franka.get_dofs_position())

    def grip(force):
        franka.control_dofs_force(np.column_stack([-force, -force]), fingers)
    if cam:
        cam.start_recording(save_to_filename=args.record, fps=50)  # 動画の 1 秒 = シミュレーションの 1 秒

    def step(n):  # 録画中は、カメラが step ごとに自分で描画する
        for _ in range(n):
            scene.step()

    def go(xy, z, steps, gripping, force=None):
        q = ik(np.tile([*xy, z], (N, 1)))
        franka.control_dofs_position(q[:, :-2], motors)
        if gripping:
            grip(Fv if force is None else force)
        else:
            franka.control_dofs_position(np.full((N, 2), 0.04), fingers)
        step(steps)

    def follow(path_fn, n):  # 0.01 s ごとに手先の目標を進める。path_fn(t) は (N, 3)
        trace = []
        for k in range(n):
            q = ik(path_fn((k + 1) * 0.01))
            franka.control_dofs_position(q[:, :-2], motors)
            grip(Fv)
            step(1)
            trace.append(hand.get_pos().cpu().numpy().copy())
        return np.array(trace).reshape(len(trace), N, 3)

    def accel_p95(tr):  # 手先の位置の 2 階差分 → 環境ごとの加速度の 95 パーセンタイル
        if len(tr) < 3:
            return np.zeros(N)
        return np.percentile(np.linalg.norm(np.diff(tr, 2, axis=0) / 0.01**2, axis=2), 95, axis=0)

    def steps_for(totals):
        fin = totals[np.isfinite(totals)]
        return int(math.ceil(fin.max() / 0.01)) if len(fin) else 0

    t0 = time.time()
    go(START, 0.25, 150, False)
    go(START, grip_z, 150, False)
    go(START, grip_z, 50, True, np.minimum(Fv, 5.0))  # そっと閉じてから、0.5 秒で目標の力まで上げる（いきなり強く閉じるとはじき出す）
    for j in range(50):
        go(START, grip_z, 1, True, np.minimum(Fv, 5.0) + (Fv - np.minimum(Fv, 5.0)) * (j + 1) / 50)

    lift_fn, t_lift = trapezoids(Z_CARRY - grip_z, a_plan, V_MAX)
    tr_lift = follow(lambda t: np.column_stack([np.tile(START, (N, 1)), grip_z + lift_fn(t)]), steps_for(t_lift))
    z_end = np.where(refused, grip_z, Z_CARRY)  # 運ばない環境は、掴んだ高さのまま
    q = ik(np.column_stack([np.tile(START, (N, 1)), z_end]))
    franka.control_dofs_position(q[:, :-2], motors)
    step(30)
    lifted = cube.get_pos()[:, 2].cpu().numpy() > 0.15

    # 横に運ぶ：環境ごとの台形の速度で、0.01 s ごとに手先の目標を進める
    dist = float(np.linalg.norm(GOAL - START))
    direction = (GOAL - START) / dist
    move_fn, t_move = trapezoids(dist, a_plan, V_MAX)
    tr_move = follow(lambda t: np.column_stack([START + np.outer(move_fn(t), direction), z_end]), steps_for(t_move))
    xy_end = np.where(refused[:, None], START, GOAL)
    q = ik(np.column_stack([xy_end, z_end]))
    franka.control_dofs_position(q[:, :-2], motors)
    step(100)  # 1 秒保持して、まだ持っているかを見る
    wall = time.time() - t0
    if cam:
        cam.stop_recording()

    p = cube.get_pos().cpu().numpy()
    a95 = np.maximum(accel_p95(tr_lift), accel_p95(tr_move))
    held = (p[:, 2] > 0.15) & (np.linalg.norm(p[:, :2] - GOAL, axis=1) < 0.05)
    success = lifted & held & ~refused
    t_each = t_lift + t_move
    ok_t = t_each[np.isfinite(t_each)]
    rnd = lambda xs, k=4: [round(float(x), k) if np.isfinite(x) else None for x in xs]  # noqa: E731
    result = {
        "material": args.material, "strategy": args.strategy, "candidates": candidates, "seeds": seeds,
        "force_N": rnd(Fv, 3), "min_force": args.min_force, "side_m": SIDE, "mass_kg": round(mass_true, 4), "mass_assumed": round(m_hat, 4),
        "mu_scale": args.mu_scale, "mass_scale": args.mass_scale,
        "mu_range": [lo, hi], "mu_sources": sources, "true_widen": list(TRUE_WIDEN),
        "mu_assumed": round(float(mu_hat[0]), 4) if np.ptp(mu_hat) == 0 else None,  # 全環境で同じときだけ
        "mu_hat": rnd(mu_hat), "true_mu": rnd(true_mu),
        "a_plan": rnd(a_plan, 3), "a_measured_p95": rnd(a95, 3),
        "a_max_true": [round(a_max(float(m), mass_true, f), 3) for m, f in zip(true_mu, Fv)],
        "refused": [bool(r) for r in refused],
        "t_each": rnd(t_each, 3),
        # 平均は運んだ環境だけで取る（旧版の t_lift / t_move を使う呼び出し側との互換）
        "t_lift": round(float(np.mean(t_lift[np.isfinite(t_lift)])), 3) if len(ok_t) else None,
        "t_move": round(float(np.mean(t_move[np.isfinite(t_move)])), 3) if len(ok_t) else None,
        "success": [bool(s) for s in success], "success_rate": float(success.mean()),
        # 摩擦の余裕：使える摩擦力 2μF ÷ 必要な摩擦力 m(g + a) − 1（本当の μ・本当の重さと、実測した加速度の p95 で計算）
        "margin": [round(2 * float(m) * f / (mass_true * (G + float(a))) - 1, 4) for m, f, a in zip(true_mu, Fv, a95)],
        "wall_s": round(wall, 1),
    }
    t_txt = f"{np.mean(ok_t):.2f}s" if len(ok_t) else "-"
    print(f"{args.material:9s} {args.strategy}  F={np.median(Fv):.1f}N  m={mass_true * 1000:.0f}g (hat {m_hat * 1000:.0f}g)  "
          f"mu_hat={np.median(mu_hat):.3f}  a_plan={np.median(a_plan):5.2f}  a_meas95={np.median(a95):5.2f}  "
          f"t_lift+move={t_txt}  refused={int(refused.sum())}  success={int(success.sum())}/{N}  (wall {wall:.0f}s)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
