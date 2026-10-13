"""応用編 X1：2 台の直列ライン（A が置き台へ → B が箱へ）に、戦略 A/B/C の加速度を入れる。

置き台があるので、2 台は別々の部品を同時に扱える（A が次の部品を運ぶ間に、B が前の部品を運ぶ）。
そのためラインのタクトタイムは「遅いほうの工程の時間」で決まる。両方の工程とも、持ち上げと移動に
同じ加速度の計画（sim/transport.py の planned_accel）を使う。実際の μ は文献値の幅から seed ごとに抽選。

使い方: python sim/line.py <material> <strategy> [n_seeds] [--candidates POM,ABS] [--json out.json]
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
from mu_table import load, mu_range  # noqa: E402
from transport import DENSITY, FORCE, G, MASS, V_MAX, assumed_mu, planned_accel, trapezoid  # noqa: E402

PICK, STAND, PLACE = np.array([0.55, -0.25]), np.array([0.55, 0.00]), np.array([0.55, 0.25])
STAND_H, OFFSET, Z_UP = 0.06, 0.115, 0.35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("material", choices=list(DENSITY))
    ap.add_argument("strategy", choices=["A", "B", "C"])
    ap.add_argument("n_seeds", nargs="?", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--candidates")
    ap.add_argument("--json")
    args = ap.parse_args()

    rows = load()
    lo, hi, _ = mu_range(args.material, rows=rows)
    N = args.n_seeds
    seeds = list(range(args.seed0, args.seed0 + N))
    true_mu = np.array([np.random.default_rng(s).uniform(lo, hi) for s in seeds])
    cands = args.candidates.split(",") if args.candidates else None
    mu_hat = assumed_mu(args.strategy, args.material, rows, cands)
    a = planned_accel(mu_hat)
    side = (MASS / DENSITY[args.material]) ** (1 / 3)

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
                     rigid_options=gs.options.RigidOptions(box_box_detection=True,
                                                           friction_cone=gs.friction_cone.elliptic, impratio=10.0),
                     show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.12, 0.12, STAND_H), pos=(*STAND, STAND_H / 2), fixed=True))
    cube = scene.add_entity(gs.morphs.Box(size=(side,) * 3, pos=(*PICK, side / 2)),
                            material=gs.materials.Rigid(friction=1.0, rho=DENSITY[args.material]))
    arm_a = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 0, 0)),
                             material=gs.materials.Rigid(friction=0.01))
    arm_b = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(1.10, 0, 0), euler=(0, 0, 180)),
                             material=gs.materials.Rigid(friction=0.01))
    scene.build(n_envs=N)
    cube.set_friction_ratio(torch.tensor(true_mu, dtype=gs.tc_float, device=gs.device).unsqueeze(1), links_idx_local=[0])
    for arm in (arm_a, arm_b):
        arm.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        arm.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        arm.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                                 np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
    motors, fingers, quat = np.arange(7), np.arange(7, 9), np.tile([0, 1, 0, 0], (N, 1))

    def cmd(arm, p, grip):
        q = arm.inverse_kinematics(link=arm.get_link("hand"), pos=np.tile(p, (N, 1)), quat=quat)
        arm.control_dofs_position(q[:, :-2], motors)
        if grip:
            arm.control_dofs_force(np.full((N, 2), -FORCE), fingers)
        else:
            arm.control_dofs_position(np.full((N, 2), 0.04), fingers)

    def hold(n):
        for _ in range(n):
            scene.step()

    def glide(arm, p0, p1, grip, n=100, settle=50):  # 掴む前後の、ゆっくりした動き（時間は工程の固定分）
        for t in np.linspace(0, 1, n):
            cmd(arm, p0 + (p1 - p0) * t, grip)
            scene.step()
        hold(settle)

    def planned(arm, p0, p1):  # 部品を持ったまま、戦略の加速度 a で台形に動く
        d = float(np.linalg.norm(p1 - p0))
        pos, T = trapezoid(d, a, V_MAX)
        for k in range(int(math.ceil(T / 0.01))):
            cmd(arm, p0 + (p1 - p0) / d * pos((k + 1) * 0.01), True)
            scene.step()
        hold(30)
        return T

    t0 = time.time()
    z_pick, z_stand = side / 2 + OFFSET, STAND_H + side / 2 + OFFSET
    cmd(arm_a, np.array([0.40, -0.25, Z_UP]), False)
    cmd(arm_b, np.array([0.70, 0.25, Z_UP]), False)
    hold(150)
    # 工程 1（アーム A）：トレイ → 置き台
    glide(arm_a, np.array([0.40, -0.25, Z_UP]), np.array([*PICK, Z_UP]), False)
    glide(arm_a, np.array([*PICK, Z_UP]), np.array([*PICK, z_pick]), False)
    cmd(arm_a, np.array([*PICK, z_pick]), True)
    hold(80)
    t1 = planned(arm_a, np.array([*PICK, z_pick]), np.array([*PICK, Z_UP]))
    t1 += planned(arm_a, np.array([*PICK, Z_UP]), np.array([*STAND, Z_UP]))
    on_stand_a = cube.get_pos()[:, 2].cpu().numpy() > 0.15  # 運んでいる途中で落としていないか
    glide(arm_a, np.array([*STAND, Z_UP]), np.array([*STAND, z_stand]), True)
    cmd(arm_a, np.array([*STAND, z_stand]), False)
    hold(60)
    glide(arm_a, np.array([*STAND, z_stand]), np.array([0.40, -0.25, Z_UP]), False)
    # 工程 2（アーム B）：置き台 → 箱
    glide(arm_b, np.array([0.70, 0.25, Z_UP]), np.array([*STAND, Z_UP]), False)
    glide(arm_b, np.array([*STAND, Z_UP]), np.array([*STAND, z_stand]), False)
    cmd(arm_b, np.array([*STAND, z_stand]), True)
    hold(80)
    t2 = planned(arm_b, np.array([*STAND, z_stand]), np.array([*STAND, Z_UP]))
    t2 += planned(arm_b, np.array([*STAND, Z_UP]), np.array([*PLACE, Z_UP]))
    glide(arm_b, np.array([*PLACE, Z_UP]), np.array([*PLACE, z_pick]), True)
    cmd(arm_b, np.array([*PLACE, z_pick]), False)
    hold(60)
    p = cube.get_pos().cpu().numpy()
    success = on_stand_a & (np.linalg.norm(p[:, :2] - PLACE, axis=1) < 0.03) & (p[:, 2] < 0.05)
    res = {"material": args.material, "strategy": args.strategy, "candidates": cands, "mu_assumed": round(mu_hat, 4),
           "a_plan": round(a, 3), "seeds": seeds, "true_mu": [round(float(m), 4) for m in true_mu],
           "success": [bool(s) for s in success], "success_rate": float(success.mean()),
           "t_stage1_planned": round(t1, 3), "t_stage2_planned": round(t2, 3),
           "bottleneck_s": round(max(t1, t2), 3), "wall_s": round(time.time() - t0, 1)}
    print(f"{args.material:9s} {args.strategy}  a_plan={a:5.2f}  stage1={t1:.2f}s stage2={t2:.2f}s  "
          f"success={int(success.sum())}/{N}  (wall {res['wall_s']:.0f}s)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
