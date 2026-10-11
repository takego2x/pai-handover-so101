"""S8: 掴んで横に運ぶ。運ぶ加速度は「想定した μ」から決め、実際の μ は文献値の幅から抽選する。

持ち上げ（上向きに加速度 a）の保持の条件（指 2 本）:  m * (g + a) <= 2 * mu * F
  -> 許される加速度  a_max(mu) = 2 mu F / m - g
横に運ぶときの条件 m*sqrt(g^2+a^2) <= 2 mu F はこれより緩いので、同じ a を使えば両方を満たす。

使い方: python sim/transport.py <material> <strategy> [n_seeds] [--record out.mp4]
  material: POM / aluminium / steel
  strategy: A (一律・代表値) / B (一律・最悪値) / C (材質別・下限)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mu_table import load, mu_range, safe_mu  # noqa: E402

G = 9.81
FORCE = 8.0          # 指 1 本の把持力 [N]。全戦略・全材質で固定（グリッパの上限）
MASS = 0.090         # 部品の質量 [kg]。材質ごとに大きさを変えて約 90 g にそろえる
DENSITY = {"POM": 1410.0, "aluminium": 2700.0, "steel": 7850.0}  # [kg/m^3]
A_CAP = float(__import__("os").environ.get("A_CAP", 9.0))  # ロボットが実際に出せる加速度 [m/s^2]（実測で決める）
V_MAX = 1.5          # 運ぶ速さの上限 [m/s]
SAFETY = 0.85        # 使える摩擦力のうち計画に使う割合（最低 15% の余裕を残す）
START, GOAL = np.array([0.55, -0.20]), np.array([0.55, 0.20])  # 運ぶ区間（x, y）[m]
Z_CARRY = 0.35


def a_max(mu):
    return max(0.0, 2 * mu * FORCE / MASS - G)


def planned_accel(mu_assumed):
    # m (g + a) <= SAFETY * 2 mu F  ->  a = SAFETY * 2 mu F / m - g（安全率は加速度ではなく摩擦力にかける）
    return min(A_CAP, max(0.0, SAFETY * 2 * mu_assumed * FORCE / MASS - G))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("material", choices=list(DENSITY))
    ap.add_argument("strategy", choices=["A", "B", "C"])
    ap.add_argument("n_seeds", nargs="?", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1000)
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
    seeds = list(range(args.seed0, args.seed0 + N))
    true_mu = np.array([np.random.default_rng(s).uniform(lo, hi) for s in seeds])  # 実際の μ は幅の中から抽選
    candidates = args.candidates.split(",") if args.candidates else None
    mu_hat = assumed_mu(args.strategy, args.material, rows, candidates)
    if args.mu_assumed is not None:
        if args.strategy != "A":
            sys.exit("--mu-assumed は戦略 A（一律）だけで使う")
        mu_hat = args.mu_assumed
    a_plan = planned_accel(mu_hat)
    dist = float(np.linalg.norm(GOAL - START))
    pos_fn, t_move = trapezoid(dist, a_plan, V_MAX)

    side = (MASS / DENSITY[args.material]) ** (1 / 3)  # 約 90 g になる立方体の一辺
    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
                     rigid_options=gs.options.RigidOptions(box_box_detection=True, friction_cone=gs.friction_cone.elliptic, impratio=10.0), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(gs.morphs.Box(size=(side, side, side), pos=(*START, side / 2)),
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

    motors, fingers = np.arange(7), np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                                np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
    hand = franka.get_link("hand")
    quat = np.tile([0, 1, 0, 0], (N, 1))
    grip_z = side / 2 + 0.115  # 手先（hand リンク）から指先の中心までが約 0.11 m
    if cam:
        cam.start_recording(save_to_filename=args.record, fps=50)  # 動画の 1 秒 = シミュレーションの 1 秒

    def step(n):  # 録画中は、カメラが step ごとに自分で描画する
        for _ in range(n):
            scene.step()

    def go(xy, z, steps, grip):
        q = franka.inverse_kinematics(link=hand, pos=np.tile([*xy, z], (N, 1)), quat=quat)
        franka.control_dofs_position(q[:, :-2], motors)
        if grip:
            franka.control_dofs_force(np.full((N, 2), -FORCE), fingers)
        else:
            franka.control_dofs_position(np.full((N, 2), 0.04), fingers)
        step(steps)

    t0 = time.time()
    go(START, 0.25, 150, False)
    go(START, grip_z, 150, False)
    go(START, grip_z, 100, True)
    trace = []  # env 0 の手先位置（実際に出た加速度を測るため）

    def follow(path_fn, n):  # 0.01 s ごとに手先の目標を進める
        for k in range(n):
            q = franka.inverse_kinematics(link=hand, pos=np.tile(path_fn((k + 1) * 0.01), (N, 1)), quat=quat)
            franka.control_dofs_position(q[:, :-2], motors)
            franka.control_dofs_force(np.full((N, 2), -FORCE), fingers)
            step(1)
            trace.append(hand.get_pos().cpu().numpy()[0].copy())

    lift_fn, t_lift = trapezoid(Z_CARRY - grip_z, a_plan, V_MAX)
    follow(lambda t: np.array([*START, grip_z + lift_fn(t)]), int(math.ceil(t_lift / 0.01)))
    go(START, Z_CARRY, 30, True)
    lifted = cube.get_pos()[:, 2].cpu().numpy() > 0.15

    # 横に運ぶ：台形の速度で、0.01 s ごとに手先の目標を進める
    direction = (GOAL - START) / dist
    follow(lambda t: np.array([*(START + direction * pos_fn(t)), Z_CARRY]), int(math.ceil(t_move / 0.01)))
    go(GOAL, Z_CARRY, 100, True)  # 1 秒保持して、まだ持っているかを見る
    wall = time.time() - t0
    if cam:
        cam.stop_recording()

    p = cube.get_pos().cpu().numpy()
    tr = np.array(trace)
    acc = np.linalg.norm(np.diff(tr, 2, axis=0) / 0.01**2, axis=1) if len(tr) > 2 else np.zeros(1)
    held = (p[:, 2] > 0.15) & (np.linalg.norm(p[:, :2] - GOAL, axis=1) < 0.05)
    success = lifted & held
    result = {
        "material": args.material, "strategy": args.strategy, "candidates": candidates, "seeds": seeds,
        "mu_range": [lo, hi], "mu_sources": sources, "mu_assumed": round(mu_hat, 4),
        "a_planned": round(a_plan, 3), "a_measured_p95": round(float(np.percentile(acc, 95)), 3),
        "t_lift": round(t_lift, 3), "t_move": round(t_move, 3),
        "true_mu": [round(float(m), 4) for m in true_mu],
        "a_max_true": [round(a_max(float(m)), 3) for m in true_mu],
        "success": [bool(s) for s in success], "success_rate": float(success.mean()),
        # 摩擦の余裕：使える摩擦力 2μF ÷ 必要な摩擦力 m(g + a) − 1（実際の μ と、実測した加速度の p95 で計算）
        "margin": [round(2 * float(m) * FORCE / (MASS * (G + float(np.percentile(acc, 95)))) - 1, 4) for m in true_mu],
        "force_N": FORCE, "mass_kg": MASS, "wall_s": round(wall, 1),
    }
    print(f"{args.material:9s} {args.strategy}  mu_assumed={mu_hat:.3f}  a_plan={a_plan:5.2f} m/s2  "
          f"a_meas95={np.percentile(acc, 95):5.2f}  t_lift+move={t_lift + t_move:.2f}s  success={success.sum()}/{N}  (wall {wall:.0f}s)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
