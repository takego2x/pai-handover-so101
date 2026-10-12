"""Spike（2026-10-12）：第一層 3 台が試し掴みで推定 → データと部品を置き台へ → 第二層 1 台がデータで一発輸送。

第一層（アーム A1〜A3、並行）：部品を掴んで 1 cm 持ち上げ、握る力を緩めて滑り始めの力 F_slip を探る（2 回）。
    推定した F_slip を「データ」として、部品を自分の置き台へ運ぶ。
第二層（アーム B、順番に）：置き台の部品を、データから決めた力で掴み、加速度の上限で一発で箱へ運ぶ。
    運ぶ力  F = F_slip * (g + a) / (SAFETY * g)
    （滑り始め 2 μ F_slip = m g と、保持の条件 m (g + a) <= SAFETY * 2 μ F から。μ と重さを別々に知る必要がない）
時間は工程ごとに数え、第二層 1 台に第一層が何台要るか（＝第一層の 1 回の時間 ÷ 第二層の 1 回の時間）を出す。
使い方: python spikes/line3x1.py [n_envs] [--nodata] [--json out.json]
  --nodata: 第二層がデータを使わず、F_MAX と文献の最悪値の加速度で運ぶ（比べる相手）
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
from mu_table import load, mu_range  # noqa: E402
from transport import (A_CAP, DENSITY, F_MAX, G, SAFETY, SIDE, TRUE_WIDEN, V_MAX, assumed_mu, mass_of,  # noqa: E402
                       planned_accel, trapezoid)

MATERIALS = ["POM", "aluminium", "steel"]  # 第一層の 1〜3 台目が扱う部品（--random でないとき）
POOL = ["POM", "ABS", "aluminium", "steel", "NBR"]  # --random のとき、部品はこの中からランダムに来る
TH = np.radians([-60.0, -15.0, 30.0])      # 置き台の方向（第二層 B の根元から見た角度）
R_STAND, R_A = 0.55, 1.15
DROP_TH, R_DROP = np.radians(110.0), 0.50
STAND_H, STAND_W, OFFSET, Z_UP = 0.06, 0.03, 0.115, 0.35
F0, F_MIN, PROBE_DZ, SLIP_MM = 60.0, 0.2, 0.01, 1.0
COARSE_STEPS, FINE_STEPS, HOLD_STEPS = 300, 150, 30
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
DT = 0.01
B_YAW_OFFSET = float(__import__("os").environ.get("B_YAW_OFFSET", math.pi / 2))


def polar(r, th):
    return np.array([r * math.cos(th), r * math.sin(th)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_envs", nargs="?", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--nodata", action="store_true")
    ap.add_argument("--layer1", choices=["trials", "ramp"], default="trials",
                    help="trials: 落としてよい試行で保持できる最小の力を絞る／ramp: 握ったまま力を緩めて滑り始めを探る")
    ap.add_argument("--n-trials", type=int, default=7, help="trials のときの試行の回数（力を半分ずつ絞る）")
    ap.add_argument("--random", action="store_true", help="第一層の各台に、材質がランダムに来る（POOL から抽選）")
    ap.add_argument("--json")
    args = ap.parse_args()
    N = args.n_envs
    rows = load()

    stands = [polar(R_STAND, t) for t in TH]
    a_bases = [polar(R_A, t) for t in TH]
    picks = [a_bases[k] + 0.5 * polar(1, TH[k] + math.pi) + 0.25 * polar(1, TH[k] + math.pi / 2) for k in range(3)]
    tangent = polar(1, DROP_TH + math.pi / 2)
    drops = [polar(R_DROP, DROP_TH) + (k - 1) * 0.10 * tangent for k in range(3)]

    # 本当の μ：部品（台）ごと・環境ごとに、文献値の幅を広げた範囲から抽選
    # 材質：台ごとに固定、または --random で台ごと・環境ごとにランダム
    if args.random:
        mats = [[POOL[np.random.default_rng(7000 * (k + 1) + s).integers(len(POOL))] for s in range(args.seed0, args.seed0 + N)]
                for k in range(3)]
    else:
        mats = [[m] * N for m in MATERIALS]
    true_mu = np.zeros((3, N))
    for k in range(3):
        for i, s in enumerate(range(args.seed0, args.seed0 + N)):
            lo, hi, _ = mu_range(mats[k][i], rows=rows)
            true_mu[k, i] = np.random.default_rng(1000 * (k + 1) + s).uniform(lo * TRUE_WIDEN[0], hi * TRUE_WIDEN[1])
    mass2 = np.array([[mass_of(m) for m in row] for row in mats])  # (3, N)

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, substeps=10),
                     rigid_options=gs.options.RigidOptions(box_box_detection=True, batch_links_info=True,
                                                           friction_cone=gs.friction_cone.elliptic, impratio=10.0),
                     show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    for s in stands:
        scene.add_entity(gs.morphs.Box(size=(STAND_W, STAND_W, STAND_H), pos=(*s, STAND_H / 2), fixed=True))
    # 部品は、扱う腕の向きに合わせて回して置く（指を腕の向きに合わせると、手首が関節の限界から離れる）
    cubes = [scene.add_entity(gs.morphs.Box(size=(SIDE,) * 3, pos=(*picks[k], SIDE / 2), euler=(0, 0, math.degrees(TH[k]))),
                              material=gs.materials.Rigid(friction=1.0, rho=1000.0)) for k in range(3)]
    arms_a = [scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(*a_bases[k], 0),
                                              euler=(0, 0, math.degrees(TH[k]) + 180)),
                               material=gs.materials.Rigid(friction=0.01)) for k in range(3)]
    arm_b = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 0, 0)),
                             material=gs.materials.Rigid(friction=0.01))
    scene.build(n_envs=N)
    for k in range(3):
        cubes[k].set_friction_ratio(torch.tensor(true_mu[k], dtype=gs.tc_float, device=gs.device).unsqueeze(1),
                                    links_idx_local=[0])
        # 重さ（と慣性）は環境ごとに材質から設定する（同じ形の部品を、環境ごとに違う材質にする）
        m_k = torch.tensor(mass2[k], dtype=gs.tc_float, device=gs.device)
        cubes[k].set_mass(m_k)  # 部品全体の重さ。慣性も同じ比で変わる（基本形の部品はリンク単位では書けない）
        got = cubes[k].get_mass().cpu().numpy()
        assert np.allclose(got, mass2[k], rtol=0.005), (got, mass2[k])
    for arm in arms_a + [arm_b]:
        arm.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        arm.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        arm.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                                 np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
        arm.set_dofs_position(np.tile(HOME, (N, 1)))
        arm.control_dofs_position(np.tile(HOME, (N, 1)))
    motors, fingers = np.arange(7), np.arange(7, 9)
    # 指の向き（手先の回転）は腕ごとに決める。世界の向きに固定すると、回して置いた腕では手首（第 7 関節）が
    # 限界 166° の近くまで回り、環境によって IK が別の解に飛んだ（2026-10-12）
    yaw = {id(arms_a[k]): TH[k] + math.pi for k in range(3)}
    yaw[id(arm_b)] = 0.0

    def quat_of(arm, pos=None):
        if arm is arm_b:  # B の指の向きは、根元から見て常に同じ（腕が回ると部品も一緒に回る。手首が限界に近づかない）
            # さらに 90° 回し、指を閉じる向きを回る向き（接線）に揃える。接線の加速を指の押す力で受け、
            # 部品が指を軸に振り子のように回らないようにする（2026-10-12、弧の途中で部品が回って落ちた）
            y = np.arctan2(pos[:, 1], pos[:, 0]) + B_YAW_OFFSET
            return np.column_stack([np.zeros(N), np.cos(y / 2), np.sin(y / 2), np.zeros(N)])
        y = yaw[id(arm)]
        return np.tile([0.0, math.cos(y / 2), math.sin(y / 2), 0.0], (N, 1))

    def wrap90(a):  # 正方形の部品は 90° ごとに同じ形
        return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4

    def path(arm, p0, p1, t):  # t: (N,) か スカラー。B は根元の周りを弧で回る（直線だと根元の近くを通って暴れる）
        t = np.asarray(t, float)[..., None]
        if arm is not arm_b:
            return p0 + (p1 - p0) * t
        r0, r1 = np.linalg.norm(p0[:2]), np.linalg.norm(p1[:2])
        a0, a1 = math.atan2(p0[1], p0[0]), math.atan2(p1[1], p1[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        r, a = r0 + (r1 - r0) * t[..., 0], a0 + da * t[..., 0]
        return np.stack([r * np.cos(a), r * np.sin(a), p0[2] + (p1[2] - p0[2]) * t[..., 0]], axis=-1)

    def path_len(arm, p0, p1):
        pts = path(arm, p0, p1, np.linspace(0, 1, 101))
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    z_pick, z_stand = SIDE / 2 + OFFSET, STAND_H + SIDE / 2 + OFFSET
    clock = {"A": 0, "B": 0}  # 工程ごとの経過ステップ（第一層 3 台は並行なので 1 本の時計）

    def cmd(arm, pos, force):  # pos: (3,) か (N, 3)。force: None なら指を開く、(N,) なら力で握る
        pos = np.broadcast_to(pos, (N, 3))
        if arm is arm_b:  # B は大きく回るので、毎回「根元を目標の方向に向けた自然な姿勢」から解く（肘が裏返った解に落ちない）
            seed = np.tile(HOME, (N, 1))
            seed[:, 0] = np.arctan2(pos[:, 1], pos[:, 0])
            seed[:, 6] = HOME[6] - B_YAW_OFFSET
            seed = torch.tensor(seed, dtype=gs.tc_float, device=gs.device)
        else:
            seed = arm.get_dofs_position()
        q = arm.inverse_kinematics(link=arm.get_link("hand"), pos=pos, quat=quat_of(arm, pos), init_qpos=seed)
        arm.control_dofs_position(q[:, :-2], motors)
        if force is None:
            arm.control_dofs_position(np.full((N, 2), 0.04), fingers)
        else:
            arm.control_dofs_force(np.column_stack([-force, -force]), fingers)

    def step(n, who):
        for _ in range(n):
            scene.step()
        clock[who] += n

    def glide_all(targets_from, targets_to, forces, who, n=100, settle=50, arms=None):
        arms = arms_a if arms is None else arms
        for t in np.linspace(0, 1, n):
            for k, arm in enumerate(arms):
                cmd(arm, path(arm, targets_from[k], targets_to[k], t), forces[k])
            step(1, who)
        step(settle, who)

    def close_all(targets, forces, who, arms=None):  # そっと閉じてから、0.5 秒で目標の力まで上げる
        arms = arms_a if arms is None else arms
        f0 = [np.minimum(f, 5.0) for f in forces]
        for k, arm in enumerate(arms):
            cmd(arm, targets[k], f0[k])
        step(50, who)
        for j in range(50):
            for k, arm in enumerate(arms):
                cmd(arm, targets[k], f0[k] + (forces[k] - f0[k]) * (j + 1) / 50)
            step(1, who)
        step(30, who)

    def planned_all(p0s, p1s, accels, forces, who, arms=None):  # 台形の速度で動く。環境ごとに加速度が違ってよい
        arms = arms_a if arms is None else arms
        fns = []
        for k in range(len(arms)):
            d = path_len(arms[k], p0s[k], p1s[k])
            fns.append((d, [trapezoid(d, float(a), V_MAX) for a in np.broadcast_to(accels[k], (N,))]))
        T = max(t for _, f in fns for _, t in f)
        trace = []
        for i in range(int(math.ceil(T / DT))):
            for k, arm in enumerate(arms):
                d, fk = fns[k]
                s = np.array([fn((i + 1) * DT) / max(d, 1e-9) for fn, _ in fk])
                cmd(arm, path(arm, p0s[k], p1s[k], s), forces[k])
            step(1, who)
            trace.append(arms[0].get_link("hand").get_pos().cpu().numpy().copy())
        for _ in range(30):
            step(1, who)
            trace.append(arms[0].get_link("hand").get_pos().cpu().numpy().copy())
        if __import__("os").environ.get("DIAG") and len(trace) > 2:
            tr = np.array(trace)
            acc = np.linalg.norm(np.diff(tr, 2, axis=0) / DT**2, axis=2)
            print(f"diag accel who={who} planned_a={np.round(np.broadcast_to(accels[0], (N,)), 2).tolist()} "
                  f"p95={np.round(np.percentile(acc, 95, axis=0), 1).tolist()} max={np.round(acc.max(axis=0), 1).tolist()} T={T:.2f}s")
        return T

    def where(tag):
        if not __import__("os").environ.get("DIAG"):
            return
        for k in range(3):
            c = cubes[k].get_pos().cpu().numpy()
            q = cubes[k].get_quat().cpu().numpy()
            print("diag", tag, k, np.round(c[:, :3] * 1000).astype(int).tolist(), "qw", np.round(q[:, 0], 2).tolist(), flush=True)

    def up(p, z):
        return np.array([*p, z])

    t_wall = time.time()
    # ---------------- 第一層：3 台が並行に、試し掴み → 置き台へ ----------------
    home_up = [up(picks[k], Z_UP) for k in range(3)]
    for k in range(3):
        cmd(arms_a[k], home_up[k], None)
    cmd(arm_b, up(polar(0.4, DROP_TH), Z_UP), None)
    step(150, "A")
    clock["A"] = 0  # ここから第一層の 1 回分を数える
    glide_all(home_up, [up(picks[k], z_pick) for k in range(3)], [None] * 3, "A")
    full = [np.full(N, F0)] * 3
    where('grasp_pos_reached')
    close_all([up(picks[k], z_pick) for k in range(3)], full, "A")
    glide_all([up(picks[k], z_pick) for k in range(3)], [up(picks[k], z_pick + PROBE_DZ) for k in range(3)], full, "A",
              n=30, settle=30)

    # 重さを測る：60 N で持ち上げて静止した状態で、指が部品を支える上向きの力（手首の力センサーに相当）
    def lift_force_z(n=20):
        acc = np.zeros((3, N))
        for _ in range(n):
            step(1, "A")
            for k in range(3):
                acc[k] += cubes[k].get_links_net_contact_force().cpu().numpy()[:, 0, 2]
        return acc / n

    where('probe_lifted')
    m_hat = lift_force_z() / G  # (3, N)
    n_drops = np.zeros((3, N), int)
    probe_start = clock["A"]
    if args.layer1 == "trials":
        # 失敗してよい試行：力 F で掴んで 2 cm 持ち上げ、持てたら F を下げ、落ちたら上げる（対数で半分ずつ絞る）
        lo, hi = np.full((3, N), 0.3), np.full((3, N), F0)
        lifted_at_f0 = np.array([cubes[k].get_pos()[:, 2].cpu().numpy() > SIDE / 2 + PROBE_DZ * 0.5 for k in range(3)])
        for j in range(args.n_trials):
            Fj = np.sqrt(lo * hi)
            for k in range(3):  # 下ろして、指を開く（前の試行で落ちた部品は、もう台の上）
                cmd(arms_a[k], up(picks[k], z_pick), None)
            step(30, "A")
            for k in range(3):
                cmd(arms_a[k], up(picks[k], z_pick), np.minimum(Fj[k], 5.0))
            step(20, "A")
            for r_ in range(20):
                for k in range(3):
                    f0 = np.minimum(Fj[k], 5.0)
                    cmd(arms_a[k], up(picks[k], z_pick), f0 + (Fj[k] - f0) * (r_ + 1) / 20)
                step(1, "A")
            step(20, "A")
            glide_all([up(picks[k], z_pick) for k in range(3)], [up(picks[k], z_pick + 0.02) for k in range(3)],
                      list(Fj), "A", n=30, settle=40)
            held = np.array([cubes[k].get_pos()[:, 2].cpu().numpy() > SIDE / 2 + 0.01 for k in range(3)])
            hi = np.where(held, Fj, hi)
            lo = np.where(held, lo, Fj)
            n_drops += ~held
        estimated = lifted_at_f0
        data = np.where(estimated, hi, np.nan)  # 持てた最小の力（2 μ F >= m g の境目の、上側）
        t_probe = np.full((3, N), (clock["A"] - probe_start) * DT)
    else:
        # 握る力を緩めて、滑り始めの力を探る（粗い探索 → 戻す → 精密な探索）。台ごと・環境ごとの状態機械
        where('probe_lifted')
        hands = [arm.get_link("hand") for arm in arms_a]

        def rel():
            return np.array([cubes[k].get_pos()[:, 2].cpu().numpy() - hands[k].get_pos()[:, 2].cpu().numpy()
                             for k in range(3)])

        F = np.full((3, N), F0)
        phase = np.zeros((3, N), int)
        kk = np.zeros((3, N), int)
        F1, F2 = np.full((3, N), np.nan), np.full((3, N), np.nan)
        t_probe = np.zeros((3, N))
        base = rel()
        cr, fr = (F_MIN / F0) ** (1 / COARSE_STEPS), (0.5 / 1.4) ** (1 / FINE_STEPS)
        probe_start = clock["A"]
        for it in range(COARSE_STEPS + HOLD_STEPS + FINE_STEPS + 50):
            r = rel()
            slipped = (base - r) > SLIP_MM / 1000
            for k in range(3):
                for i in range(N):
                    ph = phase[k, i]
                    if ph == 0:
                        if slipped[k, i]:
                            F1[k, i], phase[k, i], kk[k, i] = F[k, i], 1, 0
                            F[k, i] = min(F0, 2 * F1[k, i])
                        elif kk[k, i] >= COARSE_STEPS:
                            phase[k, i] = 4
                        else:
                            F[k, i] = F0 * cr ** kk[k, i]
                    elif ph == 1 and kk[k, i] >= HOLD_STEPS:
                        phase[k, i], kk[k, i] = 2, 0
                        base[k, i] = r[k, i]
                        F[k, i] = 1.4 * F1[k, i]
                    elif ph == 2:
                        if slipped[k, i]:
                            F2[k, i], phase[k, i] = F[k, i], 3
                            F[k, i] = min(F0, 2 * F2[k, i])
                            t_probe[k, i] = (it + 1) * DT
                        elif kk[k, i] >= FINE_STEPS:
                            phase[k, i] = 4
                        else:
                            F[k, i] = 1.4 * F1[k, i] * fr ** kk[k, i]
                    kk[k, i] += 1
            for k in range(3):
                arms_a[k].control_dofs_force(np.column_stack([-F[k], -F[k]]), fingers)
            step(1, "A")
            if (phase >= 3).all():
                break
        estimated = phase == 3
        data = np.where(estimated, F2, np.nan)
    probe_steps_all = clock["A"] - probe_start
    where('probe_done')
    # データ：保持の境目の力（見つからなければ、最悪に備えて F_MAX）。重さ・μ・運ぶ力にして第二層へ部品と一緒に渡す
    f_carry = np.where(estimated, np.minimum(F_MAX, data * (G + A_CAP) / (SAFETY * G)), F_MAX)
    a_carry = np.where(estimated & (data * (G + A_CAP) / (SAFETY * G) <= F_MAX), A_CAP,
                       np.where(estimated, np.maximum(0.0, SAFETY * G * F_MAX / np.nan_to_num(data, nan=1.0) - G), 0.0))
    a_carry = np.minimum(A_CAP, a_carry)
    mu_hat = m_hat * G / (2 * data)  # 測った重さから μ を出す

    if args.layer1 == "trials":  # 最後の試行で持ち上げたままの部品もある
        glide_all([up(picks[k], z_pick + 0.02) for k in range(3)], [up(picks[k], z_pick + PROBE_DZ) for k in range(3)],
                  list(np.sqrt(lo * hi)), "A", n=10, settle=10)
    # 掴み直して（試し掴みで台に落ちた部品もある）、自分のデータで置き台へ運ぶ
    glide_all([up(picks[k], z_pick + PROBE_DZ) for k in range(3)], [up(picks[k], z_pick) for k in range(3)],
              [None] * 3, "A", n=20, settle=40)
    close_all([up(picks[k], z_pick) for k in range(3)], list(f_carry), "A")
    where('regrasped')
    a1 = [np.where(a_carry[k] > 0, a_carry[k], 0.5) for k in range(3)]  # 推定できなかった部品は、ゆっくり
    planned_all([up(picks[k], z_pick) for k in range(3)], home_up, a1, list(f_carry), "A")
    planned_all(home_up, [up(stands[k], Z_UP) for k in range(3)], a1, list(f_carry), "A")
    where('above_stand')
    glide_all([up(stands[k], Z_UP) for k in range(3)], [up(stands[k], z_stand) for k in range(3)], list(f_carry), "A")
    for k in range(3):
        cmd(arms_a[k], up(stands[k], z_stand), None)
    step(60, "A")
    glide_all([up(stands[k], z_stand) for k in range(3)], [up(stands[k], Z_UP) for k in range(3)], [None] * 3, "A")
    glide_all([up(stands[k], Z_UP) for k in range(3)], home_up, [None] * 3, "A")
    where('layer1_done')
    cycle_a_fixed = (clock["A"] - probe_steps_all) * DT  # 試し掴み以外の時間（台・環境で共通）
    cycle_a = cycle_a_fixed + t_probe  # 台・環境ごとの第一層の 1 回の時間
    on_stand = np.array([np.linalg.norm(cubes[k].get_pos().cpu().numpy()[:, :2] - stands[k], axis=1) < 0.01
                         for k in range(3)])

    # ---------------- 第二層：B が置き台を順に回り、データで一発で箱へ ----------------
    if args.nodata:
        m_worst = max(mass_of(m) for m in DENSITY)
        fB = np.full((3, N), F_MAX)
        aB = np.full((3, N), planned_accel(assumed_mu("B", None, rows), m_worst, F_MAX))
    else:
        fB, aB = f_carry, np.where(a_carry > 0, a_carry, 0.5)
    success = np.zeros((3, N), bool)
    t_b = []
    b_from = up(polar(0.4, DROP_TH), Z_UP)
    for k in range(3):
        c0 = clock["B"]
        # 置き台の部品は置き台の方向（TH[k]）を向いているので、根元から見て同じ向きの指で掴める
        glide_all([b_from], [up(stands[k], Z_UP)], [None], "B", arms=[arm_b])
        glide_all([up(stands[k], Z_UP)], [up(stands[k], z_stand)], [None], "B", arms=[arm_b])
        close_all([up(stands[k], z_stand)], [fB[k]], "B", arms=[arm_b])
        where(f"B{k}_closed")
        if __import__("os").environ.get("DIAG"):
            E = int(__import__("os").environ.get("DIAG_ENV", 0)); print("diag B_at_close env", E, "hand", np.round(arm_b.get_link("hand").get_pos().cpu().numpy()[E] * 1000).astype(int).tolist(),
                  "target", np.round(up(stands[k], z_stand) * 1000).astype(int).tolist(),
                  "fingers", np.round(arm_b.get_dofs_position().cpu().numpy()[E, 7:9] * 1000, 1).tolist(),
                  "q", np.round(arm_b.get_dofs_position().cpu().numpy()[E, :7], 2).tolist(), "F", round(float(fB[k][E]), 2))
        planned_all([up(stands[k], z_stand)], [up(stands[k], Z_UP)], [aB[k]], [fB[k]], "B", arms=[arm_b])
        where(f"B{k}_lifted")
        planned_all([up(stands[k], Z_UP)], [up(drops[k], Z_UP)], [aB[k]], [fB[k]], "B", arms=[arm_b])
        where(f"B{k}_moved")
        glide_all([up(drops[k], Z_UP)], [up(drops[k], z_pick)], [fB[k]], "B", arms=[arm_b])
        cmd(arm_b, up(drops[k], z_pick), None)
        step(60, "B")
        glide_all([up(drops[k], z_pick)], [up(drops[k], Z_UP)], [None], "B", arms=[arm_b])
        b_from = up(drops[k], Z_UP)
        where(f"B{k}_released")
        if __import__("os").environ.get("DIAG"):
            print("diag hand_b", np.round(arm_b.get_link("hand").get_pos().cpu().numpy()[0] * 1000).astype(int).tolist(), "drop", np.round(drops[k] * 1000).astype(int).tolist())
        p = cubes[k].get_pos().cpu().numpy()
        success[k] = on_stand[k] & (np.linalg.norm(p[:, :2] - drops[k], axis=1) < 0.03) & (p[:, 2] < SIDE)
        t_b.append((clock["B"] - c0) * DT)
    cycle_b = float(np.mean(t_b))

    print(f"{'material':9s} {'n':>3s} {'true_mu':>11s} {'mu_hat err':>15s} {'m_hat err':>15s} {'drops':>5s} {'F_layer2':>16s} {'probe_s':>8s} on_stand success")
    flat = np.array(mats)
    for m in (POOL if args.random else MATERIALS):
        sel = flat == m
        if not sel.any():
            continue
        err = (mu_hat[sel] / true_mu[sel] - 1) * 100
        print(f"{m:9s} {int(sel.sum()):3d} {true_mu[sel].min():5.2f}-{true_mu[sel].max():5.2f}  "
              f"{np.nanmin(err):+6.1f}..{np.nanmax(err):+6.1f}%  "
              f"{np.min(m_hat[sel] / mass2[sel] - 1) * 100:+6.1f}..{np.max(m_hat[sel] / mass2[sel] - 1) * 100:+6.1f}%  "
              f"{n_drops[sel].mean():5.1f}  {np.median(fB[sel]):5.1f}N(max {fB[sel].max():4.1f})  "
              f"{np.median(t_probe[sel]):5.2f}s  {int(on_stand[sel].sum())}/{int(sel.sum())}  {int(success[sel].sum())}/{int(sel.sum())}")
    need = cycle_a.max() / cycle_b
    print(f"layer1 cycle: fixed {cycle_a_fixed:.2f}s + probe {t_probe.min():.2f}-{t_probe.max():.2f}s "
          f"= {cycle_a.min():.2f}-{cycle_a.max():.2f}s   layer2 cycle per part: {cycle_b:.2f}s "
          f"({', '.join(f'{t:.2f}' for t in t_b)})")
    print(f"layer1 arms needed per layer2 arm = max layer1 cycle / layer2 cycle = {need:.2f} -> {math.ceil(need)}"
          f"   success {int(success.sum())}/{3 * N}  nodata={args.nodata}  (wall {time.time() - t_wall:.0f}s)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"materials": mats, "random": args.random, "nodata": args.nodata, "true_mu": true_mu.round(4).tolist(),
                   "mu_hat": np.round(mu_hat, 4).tolist(), "m_true": mass2.round(4).tolist(),
                   "m_hat": np.round(m_hat, 4).tolist(), "drops": n_drops.tolist(), "layer1": args.layer1, "F_slip": np.round(data, 3).tolist(),
                   "F_layer2": np.round(fB, 3).tolist(), "a_layer2": np.round(aB, 3).tolist(),
                   "t_probe": t_probe.round(3).tolist(), "cycle_layer1_fixed_s": round(cycle_a_fixed, 3),
                   "cycle_layer2_s": [round(t, 3) for t in t_b], "arms_needed": round(need, 3),
                   "on_stand": on_stand.tolist(), "success": success.tolist()}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
