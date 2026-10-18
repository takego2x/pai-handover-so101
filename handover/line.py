"""作品の本体（5 つの箱を順番に）：見た目で予想し（Gemini）、持って確かめ（その 1）、台座で渡し、大箱へ（その 2）。

  0. Gemini：箱の写真から、材質・重さ・壊れやすさ・自信を答える（handover/data/boxes.gemini.txt。--live でその場で呼ぶ）
  1. その 1：壊れやすい箱（または自信が「あり」でない箱）は弱い力から少しずつ強めて、壊れにくい箱は強めの力から試し掴み
     → 持ち上がる最小の力（力加減）。その力で持って止め、肩の関節の力から重さを出す
  2. その 1：台座に置く
  3. その 2：力加減 × 2（潰れの 7 割まで）で台座から持つ
  4. その 2：重さに合わせた速さで運び、大箱に入れる
  潰れは、2 本の指で挟む力の 0.05 秒の平均が、箱の強さを超えたら（段ボール 15 N、発泡スチロール 6 N、金属は潰れない）。

使い方（答え合わせ用リポジトリの直下で）:
  docker compose run --rm -v <SO-ARM100>:/m:ro sim python handover/line.py [--only 0,1] [--record outputs/handover/line.mp4]
  --mistake trust_ai : Gemini の「壊れにくい」を、自信によらず信じる（発泡スチロールを石と見誤った答えで、強く握って潰す）
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
from vlm_boxes import BOXES, cube_obj, cardboard_texture, foam_texture, metal_texture  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", help="処理する箱の番号（カンマ区切り。0 から、左の箱から順に）")
ap.add_argument("--record")
ap.add_argument("--mistake", choices=["trust_ai"])
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]

CRUSH = {"cardboard": 15.0, "foam": 6.0, "metal": float("inf")}   # 箱の強さ：挟む力 [N] の上限
GENTLE = [0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8]                    # 壊れやすい箱：弱い順
QUICK = [0.65, 0.8, 1.2, 1.6]                                     # 壊れにくい箱：強めから
MARGIN, SQ_PER_TAU = 2.0, 14.5
D, R, SIDE, STAND_H = 0.44, 0.22, 0.03, 0.02
Z_CARRY_A, Z_CARRY_B, LIFT_DZ, zg = 0.06, 0.07, 0.02, 0.020
zs = STAND_H + 0.020
AZ = np.radians([-100, -80, -60, -40, -20])                      # 箱を置く方位（その 1 から見て。台座は 0°）
B_WAIT = (math.radians(60), 0.20, 0.15)                           # その 2 の待つ場所（その 2 から見て +60°）。台座の真上で待つと、
                                                                  # 台座に近い箱を測るその 1 の腕とぶつかり、重さを 1606 g と読んだ

# ---- 0. Gemini の答え（箱の写真の左から順に、BOXES と同じ並び） ----
text = (ROOT / "handover" / "data" / "boxes.gemini.txt").read_text()
answers = sorted(json.loads(text.split("```json")[-1].split("```")[0] if "```" in text else text), key=lambda t: t["point"][1])
assert len(answers) == len(BOXES), f"Gemini の答えの数 {len(answers)} が箱の数 {len(BOXES)} と合わない"

sc = s.make_scene()
sc.add_entity(s.gs.morphs.Plane())
entA = s.add_so101(sc, (0, 0), 0)
entB = s.add_so101(sc, (D, 0), 180)
sc.add_entity(s.gs.morphs.Box(size=(0.05, 0.05, STAND_H), pos=(D / 2, 0, STAND_H / 2), fixed=True),
              surface=s.gs.surfaces.Default(color=(0.6, 0.6, 0.65)))
# 大箱（その 2 から見て +90°、内寸 14 cm）。+45° に置くと、壁がその 1 の作業範囲にかかり、外へ開く指が壁に当たって
# 手が下りきれなかった（金属と 300 g の段ボールに、指が一度も触れなかった）
bx, by = D + R * math.cos(math.radians(270)), R * math.sin(math.radians(270))
for dx, dy, sx, sy in [(0.07, 0, 0.004, 0.144), (-0.07, 0, 0.004, 0.144), (0, 0.07, 0.144, 0.004), (0, -0.07, 0.144, 0.004)]:
    sc.add_entity(s.gs.morphs.Box(size=(sx, sy, 0.04), pos=(bx + dx, by + dy, 0.02), fixed=True),
                  surface=s.gs.surfaces.Default(color=(0.75, 0.6, 0.4)))
obj = cube_obj(ROOT / "outputs" / "handover" / "cube_uv.obj")
TEX = {"cardboard": lambda: s.gs.surfaces.Default(diffuse_texture=s.gs.textures.ImageTexture(image_array=cardboard_texture()), roughness=0.9),
       "foam": lambda: s.gs.surfaces.Default(diffuse_texture=s.gs.textures.ImageTexture(image_array=foam_texture()), roughness=1.0),
       "metal": lambda: s.gs.surfaces.Default(diffuse_texture=s.gs.textures.ImageTexture(image_array=metal_texture()),
                                              roughness=0.25, metallic=0.9)}
boxes = []
for k, (name, look, grams, _) in enumerate(BOXES):
    az = AZ[k]
    ent = sc.add_entity(s.gs.morphs.Mesh(file=obj, pos=(R * math.cos(az), R * math.sin(az), SIDE / 2), euler=(0, 0, math.degrees(az))),
                        material=s.gs.materials.Rigid(friction=1.0, rho=1000.0), surface=TEX[look]())
    boxes.append(ent)
cam = s.gs and sc.add_camera(res=(1280, 720), pos=(0.15, -0.55, 0.32), lookat=(0.15, -0.08, 0.03), fov=55, GUI=False) if args.record else None
sc.build(n_envs=1)  # 環境 1 つでも、並列と同じ配列の形（環境の数 × …）にそろえる
for ent, (name, look, grams, _) in zip(boxes, BOXES):
    ent.set_mass(torch.tensor([grams / 1000.0], dtype=s.gs.tc_float))
A = s.Arm(sc, entA, (0, 0), 0, 1)
B = s.Arm(sc, entB, (D, 0), 180, 1)
for _ in range(30):
    sc.step()
if cam:
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    cam.start_recording(save_to_filename=args.record, fps=25)


def aim(arm, box):
    """箱の実際の位置と向きを見て、目印を置く (方位, 半径) と指を回す角度（作品ではカメラの役目、ここではシミュレータの値）。"""
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


def z_of(box):
    return float(box.get_pos().cpu().numpy()[0, 2])


def tilt_of(box):
    q = box.get_quat().cpu().numpy()[0]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))


results = []
only = [int(x) for x in args.only.split(",")] if args.only else range(len(BOXES))
for k in only:
    name, look, grams, _ = BOXES[k]
    box, ans, limit = boxes[k], answers[k], CRUSH[look]
    st = {"peak": 0.0, "crushed": False, "recent": [], "where": ""}

    def watch(i, st=st, box=box, limit=limit):
        f = max(float(s.squeeze(A, box)[0]), float(s.squeeze(B, box)[0]))
        st["recent"] = (st["recent"] + [f])[-5:]
        fm = float(np.mean(st["recent"]))
        st["peak"] = max(st["peak"], fm)
        if fm > limit and not st["crushed"]:
            st["crushed"], st["where"] = True, f"{st.get('phase', '')} ({fm:.1f} N)"

    # 0. Gemini の答えで探し方を決める。「壊れにくい」を信じるのは自信ありのときだけ（--mistake trust_ai では自信によらず）
    sturdy = (not ans.get("fragile", True)) and (ans.get("confidence") == "high" or args.mistake == "trust_ai")
    steps = QUICK if sturdy else GENTLE
    print(f"[{k}] {name} ({look}, {grams} g): Gemini says {ans.get('looks_like')} {ans.get('weight_g')} g, "
          f"fragile={ans.get('fragile')}, confidence={ans.get('confidence')} -> {'QUICK' if sturdy else 'GENTLE'} search")
    # 重さの基準：箱の真上の、持って止める姿勢で、何も持たないときの肩の関節の力を測る（姿勢が変わると腕の重さの分も変わる）
    az, r, A.dyaw[:] = aim(A, box)

    s.move(sc, [(A, az, r, Z_CARRY_A, s.CLOSE)], 150)
    v = []
    for _ in range(100):
        sc.step()
        v.append(entA.get_dofs_control_force().cpu().numpy()[0, 1])
    tau_empty, sd_empty = float(np.mean(v)), float(np.std(v))
    # 1. 試し掴み
    st["phase"] = "1-search"
    found, tries = None, 0
    for tau in steps:
        if st["crushed"]:
            break
        az, r, A.dyaw[:] = aim(A, box)
        s.move(sc, [(A, az, r, 0.08, s.OPEN)], 100, watch)
        s.move(sc, [(A, az, r, zg, s.OPEN)], 80, watch)
        s.close_gently(sc, A, az, r, zg, tau, watch)
        s.move(sc, [(A, az, r, zg + LIFT_DZ, s.CLOSE)], 60, watch)
        s.move(sc, [(A, az, r, zg + LIFT_DZ, s.CLOSE)], 30, watch)
        tries += 1
        if z_of(box) > SIDE / 2 + LIFT_DZ * 0.5 and tilt_of(box) < 10.0:
            found = tau
            break
        A.set_grip_tau(s.TAU_MAX)
        s.move(sc, [(A, az, r, zg + LIFT_DZ, s.OPEN)], 50, watch)   # その場で開いてから、次の力で
    if found is None or st["crushed"]:
        results.append((name, look, grams, found, tries, None, st["peak"], st["crushed"], False, st["where"]))
        s.move(sc, [(A, az, r, 0.12, s.OPEN)], 80, watch)
        continue
    hold = min(found * MARGIN, max(0.7 * limit / SQ_PER_TAU, found)) if math.isfinite(limit) else found * MARGIN
    A.set_grip_tau(hold)
    st["phase"] = "1-weigh"
    s.move(sc, [(A, az, r, Z_CARRY_A, s.CLOSE)], 100, watch)
    taus = []
    for _ in range(100):
        sc.step()
        watch(0)
        taus.append(entA.get_dofs_control_force().cpu().numpy()[0, 1])
    lever = float(np.linalg.norm(box.get_pos().cpu().numpy()[0, :2] - entA.get_link("upper_arm").get_pos().cpu().numpy()[0, :2]))
    m_hat = (tau_empty - float(np.mean(taus))) / (9.81 * lever)   # 持っているとき − 持っていないとき（同じ姿勢）
    print(f"    weigh: shoulder tau empty {tau_empty:+.3f} (sd {sd_empty:.3f})  holding {np.mean(taus):+.3f} (sd {np.std(taus):.3f})  lever {lever:.3f} m"
          f"  box z {z_of(box):.3f} tilt {tilt_of(box):.1f}")
    n_carry = int(150 * (1 + max(m_hat, 0.0) / 0.15))
    # 2. 台座に置く
    st["phase"] = "2-place"
    s.move(sc, [(A, 0.0, D / 2 - s.JAW, Z_CARRY_A, s.CLOSE)], n_carry, watch)
    s.move(sc, [(A, 0.0, D / 2 - s.JAW, zs + 0.002, s.CLOSE)], 120, watch)
    s.move(sc, [(A, 0.0, D / 2 - s.JAW, zs + 0.002, s.OPEN)], 60, watch)
    s.move(sc, [(A, 0.0, D / 2 - s.JAW, 0.15, s.OPEN)], 80, watch)
    s.move(sc, [(A, -1.2, 0.18, 0.15, 0.6)], 100, watch)
    # 3. その 2 が持つ
    st["phase"] = "3-B pick"
    B.set_grip_tau(s.TAU_MAX)
    az_b, r_b, B.dyaw[:] = aim(B, box)
    s.move(sc, [(B, 0.0, 0.26, 0.20, s.CLOSE)], 120, watch)
    s.move(sc, [(B, az_b, r_b, 0.15, 0.6)], 100, watch)
    s.move(sc, [(B, az_b, r_b, 0.15, s.OPEN)], 40, watch)
    s.move(sc, [(B, az_b, r_b, zs, s.OPEN)], 120, watch)
    s.close_gently(sc, B, az_b, r_b, zs, hold, watch)
    s.move(sc, [(B, az_b, r_b, Z_CARRY_B, s.CLOSE)], 100, watch)
    # 4. 大箱へ（入れる場所を箱ごとに少しずらす）
    st["phase"] = "4-B carry"
    off = [(-0.03, -0.03), (0.03, -0.03), (-0.03, 0.03), (0.03, 0.03), (0.0, 0.0)][k % 5]
    tgt = np.array([bx + off[0], by + off[1]]) - B.base
    az_d = math.atan2(tgt[1], tgt[0]) - B.yaw
    az_d = (az_d + math.pi) % (2 * math.pi) - math.pi
    B.dyaw[:] = 0.0
    s.move(sc, [(B, az_d, float(np.linalg.norm(tgt)) - s.JAW, Z_CARRY_B, s.CLOSE)], n_carry, watch)
    s.move(sc, [(B, az_d, float(np.linalg.norm(tgt)) - s.JAW, Z_CARRY_B, s.OPEN)], 60, watch)
    s.move(sc, [(B, *B_WAIT, s.CLOSE)], 120, watch)
    p = box.get_pos().cpu().numpy()[0]
    inside = abs(p[0] - bx) < 0.068 and abs(p[1] - by) < 0.068 and p[2] < 0.06
    results.append((name, look, grams, found, tries, m_hat, st["peak"], st["crushed"], inside, st["where"]))
    if not inside:
        print(f"    {name} ended at {np.round(p, 3).tolist()} (big box center {bx:.3f}, {by:.3f})")

if cam:
    cam.stop_recording()
print(f"mistake={args.mistake}")
print(" box          look       true    grip     tries  measured  peak squeeze  crushed  result")
for name, look, grams, found, tries, m_hat, peak, crushed, inside, where in results:
    res = "CRUSHED " + where if crushed else ("OK" if inside else ("NOT LIFTED" if found is None else "DROPPED / MISSED"))
    print(f" {name:12s} {look:9s} {grams:4d} g  {('%.2f' % found) if found else '  - '} N·m  {tries:3d}   "
          f"{('%4.0f g' % (m_hat * 1000)) if m_hat is not None else '   -  '}   {peak:6.1f} N     {'YES' if crushed else 'no '}    {res}")
