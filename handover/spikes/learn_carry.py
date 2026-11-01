"""試し：運び方（握る力の倍率 kF・運ぶ時間）を、並列の環境で試して記録する（学習の素材）。

1 つの箱（重さ・硬さ・摩擦）につき、場面を 1 回作り、環境ごとに違う候補を同時に試す。
  ① 力加減：環境ごとに τ を変えて持ち上げ、持ち上がった最小の τ ＝ 力加減 found
  ② 運ぶ：環境ごとに (kF, 運ぶ step 数) を変え、握る力 found × kF で持って弧を描いて運ぶ → 潰れ・落下・時間
使い方（リポジトリ直下）:
  docker compose run --rm sim python handover/spikes/learn_carry.py --boxes 1 --seed0 1000
出力：outputs/learn/carry_<seed0>.jsonl（1 行 1 試行）
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "handover"))
import so101 as s  # noqa: E402
import softbox as sb  # noqa: E402
import vision as vi  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--boxes", type=int, default=1)
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--debug", action="store_true")
args = ap.parse_args()

R_BOX, ZG, Z_CARRY = 0.20, 0.015, 0.08
CLEAR_DOWN, PRESS = 0.004, -0.003
TAUS = np.round(np.arange(0.15, 0.80 + 1e-9, 0.025), 3)          # ① 力加減を探す τ（環境ごと）
KF = np.array([1.2, 1.5, 2.0, 2.5, 3.0, 4.0])                     # ② 握る力の倍率
NC = np.array([60, 100, 150, 250])                                # ② 運ぶ step 数（0.01 秒／step）
N_ENV = max(len(TAUS), len(KF) * len(NC))


def move(arm, sc, r, z, g, n, az=0.0):
    """全部の環境を、方位 az（数値か (環境,)）・半径 r・高さ z へ。n は数値か (環境,)：環境ごとに違う速さ"""
    az = np.broadcast_to(np.asarray(az, float), (arm.n,))
    n = np.broadcast_to(np.asarray(n, int), (arm.n,))
    tgt = arm.world(az, np.full(arm.n, r), np.full(arm.n, z))
    start = arm.last if arm.last is not None else arm.tip()
    g = np.broadcast_to(np.asarray(g, float), (arm.n,))
    for k in range(int(n.max())):
        f = np.minimum(1.0, (k + 1) / (0.7 * n))
        f = 3 * f ** 2 - 2 * f ** 3
        if k % 5 == 0 or k == n.max() - 1:
            cur = start + (tgt - start) * f[:, None]
            # 途中の方位：弧を描く（半径一定で方位を f で進める）
            a0 = np.arctan2(start[:, 1], start[:, 0]) - arm.yaw
            a = a0 + ((az - a0 + math.pi) % (2 * math.pi) - math.pi) * f
            rr = np.linalg.norm(start[:, :2], axis=1) * (1 - f) + r * f
            cur[:, 0], cur[:, 1] = rr * np.cos(a), rr * np.sin(a)
            pos, quat = np.zeros((arm.n, 3)), np.zeros((arm.n, 4))
            for i in range(arm.n):
                Rd = arm.Rd(a[i])
                pos[i] = cur[i] - Rd @ s.OFF
                quat[i] = s.quat_of(Rd)
            q = arm.e.inverse_kinematics(link=arm.g, pos=pos, quat=quat, init_qpos=arm.e.get_dofs_position(),
                                         max_solver_iters=50).cpu().numpy()
            q[:, 5] = g
            arm.e.control_dofs_position(q)
        sc.step()
        yield
    arm.last = tgt


def squash_all(ent):
    return -ent.get_dofs_position().cpu().numpy()[:, -2:].min(axis=1)


def run_box(spec):
    t0 = time.time()
    sc = s.make_scene()
    sc.add_entity(s.gs.morphs.Plane())
    ent_a = s.add_so101(sc, (0, 0), 0, friction=0.1, model=vi.camera_model(s.MODEL))
    f = sb.xml(ROOT / "outputs" / "learn" / "xml" / f"{spec['name']}.xml", spec["k"], spec["mass"], spec["mu"], spec["look"])
    box = sc.add_entity(s.gs.morphs.MJCF(file=f, pos=(R_BOX, 0, 0)))
    sc.build(n_envs=N_ENV)
    t_build = time.time() - t0

    def reset():
        sc.reset()
        ent_a.set_dofs_kp(np.full(6, 998.22))
        arm = s.Arm(sc, ent_a, (0, 0), 0, N_ENV)
        box.control_dofs_position(np.zeros((N_ENV, 2)), [box.n_dofs - 2, box.n_dofs - 1])
        for _ in range(30):
            sc.step()
        return arm

    held = {}

    def grasp(arm, tau):
        yield from move(arm, sc, 0.15, 0.10, s.CLOSE, 100)       # 箱の手前の上で構えてから、箱の位置を見る（カメラの役目）
        bx = float(np.median(box.get_pos().cpu().numpy()[:, 0]))
        r_down, r_press = bx - s.JAW - CLEAR_DOWN, bx - s.JAW - PRESS
        held["r"] = r_press
        def dbg(tag):
            if args.debug:
                print(f"DEBUG {tag}: 目印 {np.round(arm.tip()[0], 3).tolist()} 箱 {np.round(box.get_pos().cpu().numpy()[0], 3).tolist()} "
                      f"指 {ent_a.get_dofs_position().cpu().numpy()[0, 5]:.2f}", flush=True)
        yield from move(arm, sc, r_down, 0.08, s.OPEN, 120)
        dbg("上")
        yield from move(arm, sc, r_down, ZG, s.OPEN, 60)
        dbg("下ろした")
        yield from move(arm, sc, r_down, ZG, s.PRE, 40)
        dbg("PRE")
        yield from move(arm, sc, r_press, ZG, s.PRE, 40)
        dbg("当てた")
        arm.set_grip_tau(tau)
        yield from move(arm, sc, r_press, ZG, s.CLOSE, 200)
        dbg("閉じた")

    def drive(gen, worst):
        for _ in gen:
            np.maximum(worst, squash_all(box), out=worst)

    # ① 力加減
    arm = reset()
    worst = np.zeros(N_ENV)
    tau = np.resize(TAUS, N_ENV)
    drive(grasp(arm, tau), worst)
    z0 = box.get_pos().cpu().numpy()[:, 2].copy()
    drive(move(arm, sc, held["r"], ZG + 0.02, s.CLOSE, 60), worst)
    drive(move(arm, sc, held["r"], ZG + 0.02, s.CLOSE, 60), worst)
    rise = box.get_pos().cpu().numpy()[:, 2] - z0
    if args.debug:
        print("DEBUG 目印", np.round(arm.tip()[:3], 3).tolist(), "箱", np.round(box.get_pos().cpu().numpy()[:3], 3).tolist(),
              "指", np.round(ent_a.get_dofs_position().cpu().numpy()[:, 5], 3).tolist(), "上がった mm", np.round(rise * 1000, 1).tolist(),
              "沈み mm", np.round(worst * 1000, 1).tolist(), flush=True)
    ok = (rise > 0.015) & (worst < sb.TRAVEL * 0.9)
    found = float(tau[ok].min()) if ok.any() else None
    t_found = time.time() - t0 - t_build
    rows = []
    if found is not None:
        # ② 運ぶ
        arm = reset()
        cand = [(kf, nc) for kf in KF for nc in NC]
        cand += [cand[-1]] * (N_ENV - len(cand))
        kf_e, nc_e = np.array([c[0] for c in cand]), np.array([c[1] for c in cand])
        worst = np.zeros(N_ENV)
        drive(grasp(arm, np.minimum(found * kf_e, s.TAU_MAX)), worst)
        r_c = held["r"]
        drive(move(arm, sc, r_c, Z_CARRY, s.CLOSE, 60), worst)
        drive(move(arm, sc, r_c, Z_CARRY, s.CLOSE, nc_e, az=math.radians(90)), worst)
        drive(move(arm, sc, r_c, Z_CARRY, s.CLOSE, 50, az=math.radians(90)), worst)
        zf = box.get_pos().cpu().numpy()[:, 2]
        for e in range(len(KF) * len(NC)):
            rows.append({**spec, "found": found, "kF": float(kf_e[e]), "n_carry": int(nc_e[e]),
                         "crushed": bool(worst[e] > sb.TRAVEL * 0.9), "dropped": bool(zf[e] < Z_CARRY - 0.03)})
    print(f"{spec['name']} m {spec['mass']*1000:.0f} g k {spec['k']} 強度 {spec['strength']} N μ {spec['mu']}  力加減 {found}  "
          f"作る {t_build:.0f}s ① {t_found:.0f}s 合計 {time.time() - t0:.0f}s  "
          f"成功 {sum(not r['crushed'] and not r['dropped'] for r in rows)}/{len(rows)}", flush=True)
    return rows


out = ROOT / "outputs" / "learn" / f"carry_{args.seed0}.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
specs = sb.random_boxes(args.boxes, seed=args.seed0)
with out.open("a") as fp:
    for sp in specs:
        for row in run_box(sp):
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"→ {out}")
