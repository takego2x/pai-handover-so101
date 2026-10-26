"""応用：その 1 を 3 台にして、情報を集める側を増やす。その 2（1 台）は、受け取った情報で、試さずに 1 回で持って運ぶ。

  その 2 は真ん中。台座は その 2 から見て −100°・0°・+100°（22 cm）、大箱は +50°。
  SO-101 の根元の旋回は ±110° しか回らない。最初に台座を 0°・90°・180°、大箱を 270° に置いたら、その 2 は 180° の
  台座に手が届かず、そこに置かれた 2 箱を大箱に入れられなかった。そこで、台座 3 つと大箱を ±110° の中に収めた。
  その 1 の 3 台は、それぞれ台座の向こう側（真ん中から 44 cm）から真ん中を向き、自分の脇（隣の台や大箱とぶつからない側）の箱を
  試し掴み → 重さ → 台座に置く（line.py と同じ手順）。3 台は同時に動く。
  その 2 は、箱が置かれた台座から順に、力加減 × 2（潰れの 7 割まで）で持ち、重さに合わせた速さで大箱へ。

  共通部品の move は「動き終わるまで戻らない」ので、4 台を別々の手順で同時に動かせない。
  ここでは、腕ごとの手順を「1 step ずつ進む手順（generator）」として書き、毎 step、全部の手順を 1 つずつ進める。

使い方（答え合わせ用リポジトリの直下で）:
  docker compose run --rm sim python handover/line3.py [--record outputs/handover/line3.mp4] [--cam top,low]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101 as s  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent / "spikes"))
from vlm_boxes import cube_obj, dented_obj, crushed_texture, cardboard_texture, foam_texture, metal_texture  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--record")
ap.add_argument("--cam", default="top")
ap.add_argument("--max-steps", type=int, default=30000)
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
DEBUG = bool(__import__("os").environ.get("LINE3_DEBUG"))
PROBE = bool(__import__("os").environ.get("LINE3_PROBE"))

CRUSH = {"cardboard": 15.0, "foam": 6.0, "metal": float("inf")}
GENTLE = [0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8]
QUICK = [0.65, 0.8, 1.2, 1.6]
MARGIN, SQ_PER_TAU = 2.0, 14.5
D, R, SIDE, STAND_H = 0.44, 0.22, 0.03, 0.02
Z_CARRY_A, Z_CARRY_B, LIFT_DZ, zg = 0.06, 0.07, 0.02, 0.020
zs = STAND_H + 0.020
HALF = 0.6
# Gemini の答え（line.py で保存した答えを、見た目ごとに使う）：段ボール＝壊れやすい・自信 low、発泡スチロール＝「石」・壊れにくい・自信 medium、金属＝壊れにくい・自信 high
ANSWER = {"cardboard": (True, "low"), "foam": (False, "medium"), "metal": (False, "high")}
STATIONS = [0, 100, -100]                     # 台座の方位（その 2 から見て、度）。その 2 の旋回の範囲 ±110° の中
SIDE_OF = [1, -1, 1]                          # その 1 ごとに、箱を置く側（+1：左手側、−1：右手側）。隣の台と大箱を避ける
BIG_AZ = 50                                   # 大箱の方位（その 2 から見て、度）
BOXES = [  # その 1 ごとの箱（名前, 見た目, 重さ g）。左手側に、方位 +65° と +95° に置く
    [("cardboard_30", "cardboard", 30), ("metal_400", "metal", 400)],
    [("foam_10", "foam", 10), ("cardboard_300", "cardboard", 300)],
    [("cardboard_150", "cardboard", 150), ("cardboard_80", "cardboard", 80)],
]
BOX_AZ = [55, 85]                             # 箱を置く方位（その 1 から見て、度。SIDE_OF の向きに）

import os as _os
if _os.environ.get("LINE3_NOSHADOW"):   # 試し：影を描かない（GPU なしの描画で、あるコマから先へ進まなくなったため）
    if not getattr(s.gs, "_initialized", False):
        s.gs.init(backend=s.gs.cpu, logging_level="warning")
    sc = s.gs.Scene(sim_options=s.gs.options.SimOptions(dt=0.01, substeps=10),
                    rigid_options=s.gs.options.RigidOptions(box_box_detection=True, friction_cone=s.gs.friction_cone.elliptic,
                                                            impratio=10.0, batch_links_info=True, batch_dofs_info=True),
                    vis_options=s.gs.options.VisOptions(rendered_envs_idx=[0], shadow=False), show_viewer=False)
else:
    sc = s.make_scene()
sc.add_entity(s.gs.morphs.Plane())
entB = s.add_so101(sc, (0, 0), 0)
entA = [s.add_so101(sc, (D * math.cos(math.radians(a)), D * math.sin(math.radians(a))), a + 180) for a in STATIONS]
for a in STATIONS:
    sc.add_entity(s.gs.morphs.Box(size=(0.05, 0.05, STAND_H), pos=(R * math.cos(math.radians(a)), R * math.sin(math.radians(a)), STAND_H / 2),
                                  fixed=True), surface=s.gs.surfaces.Default(color=(0.6, 0.6, 0.65)))
bx, by = R * math.cos(math.radians(BIG_AZ)), R * math.sin(math.radians(BIG_AZ))   # 大箱（内寸 14 cm）
for dx, dy, sx, sy in [(0.07, 0, 0.004, 0.144), (-0.07, 0, 0.004, 0.144), (0, 0.07, 0.144, 0.004), (0, -0.07, 0.144, 0.004)]:
    sc.add_entity(s.gs.morphs.Box(size=(sx, sy, 0.04), pos=(bx + dx, by + dy, 0.02), fixed=True),
                  surface=s.gs.surfaces.Default(color=(0.75, 0.6, 0.4)))
obj = cube_obj(ROOT / "outputs" / "handover" / "cube_uv.obj")
dent = dented_obj(ROOT / "outputs" / "handover" / "cube_dent.obj")
TEX = {"cardboard": lambda: s.gs.surfaces.Default(diffuse_texture=s.gs.textures.ImageTexture(image_array=cardboard_texture()), roughness=0.9),
       "foam": lambda: s.gs.surfaces.Default(diffuse_texture=s.gs.textures.ImageTexture(image_array=foam_texture()), roughness=1.0),
       "metal": lambda: s.gs.surfaces.Default(diffuse_texture=s.gs.textures.ImageTexture(image_array=metal_texture()),
                                              roughness=0.25, metallic=0.9)}
CRUSHED_TEX = {"cardboard": cardboard_texture, "foam": foam_texture}
boxes = []   # dict: name, look, grams, ent, shell, owner, cur
n_far = 0
for i, a in enumerate(STATIONS):
    base, yaw = np.array([D * math.cos(math.radians(a)), D * math.sin(math.radians(a))]), math.radians(a + 180)
    for (name, look, grams), baz in zip(BOXES[i], BOX_AZ):
        w = yaw + SIDE_OF[i] * math.radians(baz)
        ent = sc.add_entity(s.gs.morphs.Mesh(file=obj, pos=(base[0] + R * math.cos(w), base[1] + R * math.sin(w), SIDE / 2),
                                             euler=(0, 0, math.degrees(w))),
                            material=s.gs.materials.Rigid(friction=1.0, rho=1000.0), surface=TEX[look]())
        shell = None
        if look in CRUSHED_TEX:
            n_far += 1
            shell = sc.add_entity(s.gs.morphs.Mesh(file=dent, pos=(1.5 + 0.1 * n_far, 1.5, SIDE / 2)),
                                  material=s.gs.materials.Rigid(friction=1.0, rho=1000.0),
                                  surface=s.gs.surfaces.Default(diffuse_texture=s.gs.textures.ImageTexture(
                                      image_array=crushed_texture(CRUSHED_TEX[look])), roughness=0.95))
        boxes.append({"name": name, "look": look, "grams": grams, "ent": ent, "shell": shell, "owner": i, "cur": ent,
                      "far": (1.5 + 0.1 * n_far, 1.5, SIDE / 2)})
# top は (0.0, -0.80, 0.85) だと、40.0 秒過ぎの 1 コマを、GPU なしの描画（OSMesa）が何時間たっても描き終えなかった
# （絵を読み出す所で止まる。物の位置は正常。影を切っても止まる）。3 cm ずらすと描ける
CAMS = {"top": ((0.03, -0.78, 0.86), (0.0, 0.08, 0.0), 55),
        "low": ((0.0, -0.75, 0.35), (0.0, 0.05, 0.03), 60),
        "near": ((0.12, -0.62, 0.62), (0.10, 0.0, 0.0), 55),
        "side": ((0.75, -0.55, 0.55), (0.05, 0.0, 0.02), 50),
        "over": ((0.08, -0.34, 1.05), (0.08, -0.03, 0.0), 55),     # 提出動画に使う（真上寄り、4 台が同じ大きさ）
        "near2": ((0.15, -0.80, 0.70), (0.10, 0.0, 0.0), 50)}
cams = {c: sc.add_camera(res=(1280, 720), pos=CAMS[c][0], lookat=CAMS[c][1], fov=CAMS[c][2], GUI=False)
        for c in args.cam.split(",")} if args.record else {}
RENDER_TEST = bool(__import__("os").environ.get("LINE3_RENDER_TEST"))
_rpos = tuple(float(x) for x in _os.environ.get("LINE3_CAMPOS", "").split(",")) if _os.environ.get("LINE3_CAMPOS") else CAMS["top"][0]
rcam = sc.add_camera(res=(1280, 720), pos=_rpos, lookat=CAMS["top"][1], fov=CAMS["top"][2], GUI=False) if RENDER_TEST else None
sc.build(n_envs=1)
for b in boxes:
    b["ent"].set_mass(torch.tensor([b["grams"] / 1000.0], dtype=s.gs.tc_float))
    if b["shell"] is not None:
        b["shell"].set_mass(torch.tensor([b["grams"] / 1000.0], dtype=s.gs.tc_float))
B = s.Arm(sc, entB, (0, 0), 0, 1)
A = [s.Arm(sc, e, (D * math.cos(math.radians(a)), D * math.sin(math.radians(a))), a + 180, 1) for e, a in zip(entA, STATIONS)]
for _ in range(30):
    sc.step()
rec = Path(args.record) if args.record else None
if cams:
    rec.parent.mkdir(parents=True, exist_ok=True)
    for c, cam in cams.items():
        cam.start_recording(save_to_filename=str(rec.with_name(f"{rec.stem}_{c}.mp4")), fps=25)
t = [0]          # いまの step
marks = []       # (step, 腕, 箱, 出来事)


def mark(who, name, what):
    marks.append({"step": t[0], "who": who, "box": name, "what": what})
    print(f"[{t[0] / 100:6.1f} s] {who:4s} {name:14s} {what}", flush=True)


# ---- 1 step ずつ進む動き ----
def move(arm, az, r, z, g, n):
    """共通部品の move を、1 step ずつ進む形にしたもの（S 字、5 step ごとに IK を解き直す）"""
    az = np.broadcast_to(np.asarray(az, float), (1,))
    tgt = arm.world(az, np.broadcast_to(r, (1,)), np.broadcast_to(z, (1,)))
    start = arm.last if arm.last is not None else arm.tip()
    for k in range(n):
        f = min(1.0, (k + 1) / (0.7 * n))
        f = 3 * f ** 2 - 2 * f ** 3
        if k % 5 == 0 or k == n - 1:
            cur = start + (tgt - start) * f
            Rd = arm.Rd(az[0], arm.dyaw[0])
            q = arm.e.inverse_kinematics(link=arm.g, pos=(cur[0] - Rd @ s.OFF)[None], quat=s.quat_of(Rd)[None],
                                         init_qpos=arm.e.get_dofs_position(), max_solver_iters=50).cpu().numpy()
            q[:, 5] = g
            arm.e.control_dofs_position(q)
        yield
    arm.last = tgt


def close_gently(arm, az, r, z, tau):
    arm.set_grip_tau(s.TAU_MAX)
    yield from move(arm, az, r, z, s.PRE, 40)
    arm.set_grip_tau(tau)
    yield from move(arm, az, r, z, s.CLOSE, 200)


def wait(n):
    for _ in range(n):
        yield


def aim(arm, box):
    """箱の実際の位置と向きから、目印を置く (方位, 半径) と指を回す角度（line.py と同じ）"""
    cp = box.get_pos().cpu().numpy()
    q = box.get_quat().cpu().numpy()
    yaw = math.atan2(2 * (q[0, 0] * q[0, 3] + q[0, 1] * q[0, 2]), 1 - 2 * (q[0, 2] ** 2 + q[0, 3] ** 2))
    tip = cp[0, :2].copy()
    for _ in range(3):
        rel = tip - arm.base
        az = math.atan2(rel[1], rel[0]) - arm.yaw
        dyaw = (yaw - (arm.yaw + az) + math.pi / 4) % (math.pi / 2) - math.pi / 4
        c = arm.yaw + az + dyaw
        tip = cp[0, :2] - s.JAW * np.array([math.cos(c), math.sin(c)])
    rel = tip - arm.base
    az = (math.atan2(rel[1], rel[0]) - arm.yaw + math.pi) % (2 * math.pi) - math.pi
    return az, float(np.linalg.norm(rel)), dyaw


def z_of(e):
    return float(e.get_pos().cpu().numpy()[0, 2])


def tilt_of(e):
    q = e.get_quat().cpu().numpy()[0]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))


# ---- 潰れの見張り：いま腕が扱っている箱だけ（全部の組を毎 step 調べると遅い） ----
holding = {}     # 箱の名前 → 扱っている腕
recent = {}


def watch():
    for b in boxes:
        arm = holding.get(b["name"])
        if arm is None or b.get("crushed"):
            continue
        f = float(s.squeeze(arm, b["cur"])[0])
        recent[b["name"]] = (recent.get(b["name"], []) + [f])[-5:]
        fm = float(np.mean(recent[b["name"]]))
        b["peak"] = max(b.get("peak", 0.0), fm)
        if fm > CRUSH[b["look"]]:
            b["crushed"] = True
            mark("!", b["name"], f"潰れた（{fm:.1f} N）")
            if b["shell"] is not None:
                pos, quat = b["ent"].get_pos(), b["ent"].get_quat()
                b["shell"].set_pos(pos); b["shell"].set_quat(quat); b["shell"].zero_all_dofs_velocity()
                b["ent"].set_pos(torch.tensor([b["far"]], dtype=s.gs.tc_float)); b["ent"].zero_all_dofs_velocity()
                b["cur"] = b["shell"]


# ---- 台座：その 1 が置いて情報を書き、その 2 が持ち上げたら空く ----
stands = [{"box": None, "info": None, "ready_at": None, "busy": False} for _ in STATIONS]


def first(i):
    """その 1（i 台目）の手順：自分の箱を順に、試し掴み → 重さ → 台座が空いたら置く"""
    arm = A[i]
    for b in [x for x in boxes if x["owner"] == i]:
        fragile, conf = ANSWER[b["look"]]
        steps = QUICK if (not fragile and conf == "high") else GENTLE
        name, limit = b["name"], CRUSH[b["look"]]
        mark(f"1-{i}", name, f"{'強めから' if steps is QUICK else '弱い順に'}試す（Gemini：{'壊れやすい' if fragile else '壊れにくい'}・自信 {conf}）")
        az, r, arm.dyaw[:] = aim(arm, b["cur"])
        yield from move(arm, az, r, Z_CARRY_A, s.CLOSE, 150)
        v = []
        for _ in range(100):
            yield
            v.append(arm.e.get_dofs_control_force().cpu().numpy()[0, 1])
        tau_empty = float(np.mean(v))
        found, tries = None, 0
        holding[name] = arm
        for tau in steps:
            if b.get("crushed"):
                break
            az, r, arm.dyaw[:] = aim(arm, b["cur"])
            yield from move(arm, az, r, 0.08, s.OPEN, 100)
            yield from move(arm, az, r, zg, s.OPEN, 80)
            yield from close_gently(arm, az, r, zg, tau)
            yield from move(arm, az, r, zg + LIFT_DZ, s.CLOSE, 60)
            yield from move(arm, az, r, zg + LIFT_DZ, s.CLOSE, 30)
            tries += 1
            if z_of(b["cur"]) > SIDE / 2 + LIFT_DZ * 0.5 and tilt_of(b["cur"]) < 10.0:
                found = tau
                break
            arm.set_grip_tau(s.TAU_MAX)
            yield from move(arm, az, r, zg + LIFT_DZ, s.OPEN, 50)
        if found is None or b.get("crushed"):
            mark(f"1-{i}", name, "持てない／潰れた → この箱は渡さない")
            holding.pop(name, None)
            yield from move(arm, az, r, 0.12, s.OPEN, 80)
            continue
        hold = min(found * MARGIN, max(0.7 * limit / SQ_PER_TAU, found)) if math.isfinite(limit) else found * MARGIN
        arm.set_grip_tau(hold)
        yield from move(arm, az, r, Z_CARRY_A, s.CLOSE, 100)
        taus = []
        for _ in range(100):
            yield
            taus.append(arm.e.get_dofs_control_force().cpu().numpy()[0, 1])
        lever = float(np.linalg.norm(b["cur"].get_pos().cpu().numpy()[0, :2] - arm.e.get_link("upper_arm").get_pos().cpu().numpy()[0, :2]))
        if DEBUG:
            print(f"      DEBUG 1-{i} {name}: 肩の力 空 {tau_empty:+.3f} 持つ {float(np.mean(taus)):+.3f}  てこ {lever:.3f}  箱 z {z_of(b['cur']):.3f}  q {np.round(arm.e.get_dofs_position().cpu().numpy()[0], 2).tolist()}")
        m_hat = max((tau_empty - float(np.mean(taus))) / (9.81 * lever), 0.0)
        mark(f"1-{i}", name, f"力加減 {found} N·m（{tries} 回目）、重さ {m_hat * 1000:.0f} g")
        while stands[i]["box"] is not None:          # 台座が空くまで、持ったまま待つ
            yield
        n_carry = int(150 * (1 + m_hat / 0.15))
        yield from move(arm, 0.0, D / 2 - s.JAW, Z_CARRY_A, s.CLOSE, n_carry)
        yield from move(arm, 0.0, D / 2 - s.JAW, zs + 0.002, s.CLOSE, 120)
        yield from move(arm, 0.0, D / 2 - s.JAW, zs + 0.002, s.OPEN, 60)
        holding.pop(name, None)
        yield from move(arm, 0.0, D / 2 - s.JAW, 0.15, s.OPEN, 80)
        # 台座から、箱のある側へ退く。箱のない側へ退くと、次の箱へ向かうときに台座の上を横切り、下りてくるその 2 の腕とぶつかった
        yield from move(arm, 1.2 * SIDE_OF[i], 0.18, 0.15, HALF, 100)
        stands[i].update(box=b, info=(found, m_hat), ready_at=t[0])
        mark(f"1-{i}", name, f"台座 {i} に置いた → 情報を渡す")
    mark(f"1-{i}", "-", "自分の箱を全部終えた")


def second():
    """その 2 の手順：箱の置かれた台座から順に、情報で 1 回で持ち、重さに合わせた速さで大箱へ"""
    done, k = 0, 0
    # たたんだ姿勢のその 2 は、指先が台座 0 のすぐ手前にあり、置きに来るその 1 の開いた指とぶつかる（H4 と同じ）。
    # 初めに、大箱の側の高い所へ退いて待つ
    yield from move(B, math.radians(BIG_AZ), 0.15, 0.15, s.CLOSE, 100)
    while True:
        ready = [j for j, st in enumerate(stands) if st["box"] is not None and not st["busy"]]
        if not ready:
            if all(g is None for g in gens[1:]) and not any(st["box"] for st in stands):
                return
            yield
            continue
        j = min(ready, key=lambda j: stands[j]["ready_at"])
        st = stands[j]
        st["busy"] = True
        b, (found, m_hat) = st["box"], st["info"]
        limit = CRUSH[b["look"]]
        tau_B = min(found * MARGIN, max(0.7 * limit / SQ_PER_TAU, found)) if math.isfinite(limit) else found * MARGIN
        az_s = math.radians(STATIONS[j])
        mark("2", b["name"], f"台座 {j} から、受け取った情報で持つ（{tau_B:.2f} N·m、重さ {m_hat * 1000:.0f} g）")
        B.set_grip_tau(s.TAU_MAX)
        B.dyaw[:] = 0.0
        yield from move(B, az_s, 0.15, 0.18, s.CLOSE, 100)
        az_b, r_b, B.dyaw[:] = aim(B, b["cur"])
        yield from move(B, az_b, r_b, 0.15, HALF, 80)
        yield from move(B, az_b, r_b, 0.15, s.OPEN, 30)
        yield from move(B, az_b, r_b, zs, s.OPEN, 100)
        holding[b["name"]] = B
        yield from close_gently(B, az_b, r_b, zs, tau_B)
        if DEBUG:
            print(f"      DEBUG 2 {b['name']}: 閉じた後 挟む力 {float(s.squeeze(B, b['cur'])[0]):.2f} N  指 {float(B.e.get_dofs_position()[0, 5]):.2f}  目印 {np.round(B.tip()[0], 3).tolist()}  箱 {np.round(b['cur'].get_pos().cpu().numpy()[0], 3).tolist()}  az {math.degrees(az_b):.0f}° r {r_b:.3f}")
        yield from move(B, az_b, r_b, Z_CARRY_B, s.CLOSE, 80)
        st.update(box=None, info=None, ready_at=None, busy=False)      # 持ち上げたら台座は空く
        off = [(-0.03, -0.03), (0.03, -0.03), (-0.03, 0.03), (0.03, 0.03), (0.0, 0.0), (0.0, -0.035)][k % 6]
        k += 1
        tgt = np.array([bx + off[0], by + off[1]]) - B.base
        az_d = (math.atan2(tgt[1], tgt[0]) - B.yaw + math.pi) % (2 * math.pi) - math.pi
        B.dyaw[:] = 0.0
        n_carry = int(150 * (1 + m_hat / 0.15))
        yield from move(B, az_d, float(np.linalg.norm(tgt)) - s.JAW, Z_CARRY_B, s.CLOSE, n_carry)
        yield from move(B, az_d, float(np.linalg.norm(tgt)) - s.JAW, Z_CARRY_B, s.OPEN, 50)
        holding.pop(b["name"], None)
        p = b["cur"].get_pos().cpu().numpy()[0]
        inside = abs(p[0] - bx) < 0.068 and abs(p[1] - by) < 0.068 and p[2] < 0.06
        b["inside"] = inside
        done += 1
        mark("2", b["name"], f"大箱へ（{'入った' if inside else '外れた ' + str(np.round(p, 3).tolist())}）")
        yield from move(B, az_d, 0.15, 0.15, s.CLOSE, 60)


gens = [second()] + [first(i) for i in range(len(A))]
while any(g is not None for g in gens) and t[0] < args.max_steps:
    for i, g in enumerate(gens):
        if g is None:
            continue
        try:
            next(g)
        except StopIteration:
            gens[i] = None
    sc.step()
    watch()
    t[0] += 1
    if RENDER_TEST and 3880 <= t[0] <= 4150 and t[0] % 4 == 0:   # 1 コマずつ時間を測って描く（どのコマで止まるか）
        import time as _time
        _t0 = _time.time(); rcam.render(rgb=True); _dt = _time.time() - _t0
        if _dt > 1.0 or t[0] % 40 == 0 or t[0] >= 4000:
            print(f"RENDER step {t[0]} {_dt:.2f} s", flush=True)
    if PROBE and t[0] >= 3800 and t[0] % 5 == 0:   # 描く前に、全部の物の位置が有限で、場面の近くにあるかを確かめる
        for ent in sc.entities:
            P = ent.get_links_pos().cpu().numpy() if hasattr(ent, "get_links_pos") else None
            if P is None:
                continue
            bad = (~np.isfinite(P)).any() or np.abs(P).max() > 3.0
            if bad:
                print(f"PROBE step {t[0]}: entity {ent.idx} {type(ent.morph).__name__} max|pos| {np.nanmax(np.abs(P)):.3g} finite {np.isfinite(P).all()}", flush=True)

if cams:
    for cam in cams.values():
        cam.stop_recording()
    rec.with_suffix(".json").write_text(json.dumps({"fps": 25, "steps": t[0], "marks": marks}, ensure_ascii=False, indent=1))
print(f"steps {t[0]} ({t[0] / 100:.1f} s)  {'時間切れ' if t[0] >= args.max_steps else ''}")
print(" box             true    peak squeeze  crushed  result")
for b in boxes:
    res = "CRUSHED" if b.get("crushed") else ("OK" if b.get("inside") else ("DROPPED / MISSED" if "inside" in b else "NOT DELIVERED"))
    print(f" {b['name']:14s} {b['grams']:4d} g   {b.get('peak', 0):6.1f} N     {'YES' if b.get('crushed') else 'no '}    {res}")
