"""試し：手首のカメラ（色の画像だけ。公式キットの 32 mm 角 UVC）と肩のトルクで、「持てたか」を見分けられるか。

  実機の手首カメラには深度がない。そこで次の 2 つを測る：
    画像：持ち上げる前後で、指のあいだの範囲（vision.BOX_ROI）の見た目が変わらないか（正規化相互相関 NCC、平均の差 MAD）
    肩のトルク：持ち上げた高さで、空の手のときと比べて肩の力が増えたか（→ 手に掛かっている重さ [g]）
  場面：見た目 3 種（段ボール・発泡・金属）× 置き場所（床・台座）× 握る力（弱い＝持てない／ぎりぎり＝滑る／十分＝持てた）
  あわせて、目印の狙いと実際の位置のずれ（手首に 35 g 載ったことの影響）も出す。答え合わせに、箱の本当の高さを出す。

  docker compose run --rm sim python handover/spikes/wrist_rgb.py
  → outputs/handover/wrist_rgb/result.json と画像
"""
import json
import sys
from pathlib import Path

import imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "handover"))
sys.path.insert(0, str(ROOT / "handover" / "spikes"))
import so101 as s  # noqa: E402
import softbox as sb  # noqa: E402
import vision as vi  # noqa: E402
from vlm_softboxes import surface  # noqa: E402

OUT = ROOT / "outputs" / "handover" / "wrist_rgb"
OUT.mkdir(parents=True, exist_ok=True)
STAND_H, LIFT = 0.02, 0.02
LOOKS = ["cardboard", "foam", "metal"]
TAUS = [0.10, 0.20, 0.40]            # 200 g・摩擦 0.7 の箱：0.10 は持てない、0.20 は滑る、0.40 は持てる（の見込み）
MASS, MU, K = 0.20, 0.7, 8000

sc = s.make_scene()
sc.add_entity(s.gs.morphs.Plane())
e = s.add_so101(sc, (0, 0), 0, friction=0.1, model=vi.camera_model(s.MODEL))
sc.add_entity(s.gs.morphs.Box(size=(0.05, 0.05, STAND_H), pos=(0.22, -0.20, STAND_H / 2), fixed=True),
              surface=s.gs.surfaces.Default(color=(0.6, 0.6, 0.65)))
# 場面ごとに箱を 1 つずつ使う（見た目 3 × 置き場所 2 × 力 3 = 18 個）。使わない箱は遠くに置いておく
cases, boxes = [], []
for li, look in enumerate(LOOKS):
    for place in ("floor", "stand"):
        for tau in TAUS:
            f = sb.xml(OUT / f"box_{look}.xml", K, MASS, MU, look)
            park = (1.5 + 0.1 * len(cases), 1.5, 0.0)
            boxes.append(sc.add_entity(s.gs.morphs.MJCF(file=f, pos=park), surface=surface(s.gs, look)))
            cases.append({"look": look, "place": place, "tau": tau})
hand = vi.add_hand_camera(sc)
sc.build(n_envs=1)
for b in boxes:
    sb.hold_spring(b)
arm = s.Arm(sc, e, (0, 0), 0, 1)
arm.hand = vi.attach_hand(hand, e)


def go(az, r, z, g, n):
    tgt = arm.world(np.array([az]), np.array([r]), np.array([z]))
    start = arm.last if arm.last is not None else arm.tip()
    for k in range(n):
        u = min(1.0, (k + 1) / (0.7 * n))
        u = 3 * u ** 2 - 2 * u ** 3
        if k % 5 == 0 or k == n - 1:
            cur = start + (tgt - start) * u
            Rd = arm.Rd(az, 0.0)
            q = e.inverse_kinematics(link=arm.g, pos=(cur[0] - Rd @ s.OFF)[None], quat=s.quat_of(Rd)[None],
                                     init_qpos=e.get_dofs_position(), max_solver_iters=50).cpu().numpy()
            q[:, 5] = g
            e.control_dofs_position(q)
        sc.step()
    arm.last = tgt


def shoulder(n=60):
    v = []
    for _ in range(n):
        sc.step()
        v.append(float(e.get_dofs_control_force().cpu().numpy()[0, 1]))
    return float(np.mean(v[-30:]))


def roi(img):
    h, w = img.shape[:2]
    r0, r1, c0, c1 = vi.BOX_ROI
    return img[int(r0 * h):int(r1 * h), int(c0 * w):int(c1 * w)].astype(float)


def ncc(a, b):
    a, b = a.mean(axis=2).ravel(), b.mean(axis=2).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


res = []
for i, (c, box) in enumerate(zip(cases, boxes)):
    az, r = (0.0, 0.20) if c["place"] == "floor" else (float(np.arctan2(-0.20, 0.22 - s.JAW * 0.22 / 0.297)), 0.297 - s.JAW)
    base_z = 0.0 if c["place"] == "floor" else STAND_H
    center = arm.world(np.array([az]), np.array([r + s.JAW]), np.array([0.0]))[0]
    box.set_pos(np.array([[center[0], center[1], base_z + 0.001]]))
    box.set_quat(np.array([[np.cos(az / 2), 0, 0, np.sin(az / 2)]]))
    zg = base_z + 0.020
    arm.set_grip_tau(s.TAU_MAX)
    go(az, r, zg + LIFT + 0.05, s.OPEN, 120)
    go(az, r, zg + LIFT, s.OPEN, 60)
    tau_empty = shoulder()                                # 空の手、持ち上げた高さで
    go(az, r, zg, s.OPEN, 60)
    go(az, r, zg, s.PRE, 40)
    aim_err = float(np.linalg.norm(arm.tip()[0] - arm.world(np.array([az]), np.array([r]), np.array([zg]))[0]) * 1000)
    arm.set_grip_tau(c["tau"])
    go(az, r, zg, s.CLOSE, 200)
    before = vi.hand_rgb(arm.hand)
    q_grip = float(e.get_dofs_position().cpu().numpy()[0, 5])
    z0 = float(box.get_pos().cpu().numpy()[0, 2])
    go(az, r, zg + LIFT, s.CLOSE, 60)
    tau_held = shoulder()
    after = vi.hand_rgb(arm.hand)
    z1 = float(box.get_pos().cpu().numpy()[0, 2])
    a, b = roi(before), roi(after)
    lever = 0.20
    row = {**c, "aim_err_mm": round(aim_err, 1), "q_grip": round(q_grip, 3), "box_dz_mm": round((z1 - z0) * 1000, 1),
           "ncc": round(ncc(a, b), 3), "mad": round(float(np.abs(a - b).mean()), 1),
           "m_hand_g": round((tau_empty - tau_held) / (9.81 * lever) * 1000, 1)}
    res.append(row)
    print(row, flush=True)
    imageio.imwrite(OUT / f"{i:02d}_{c['look']}_{c['place']}_{c['tau']}_before.png", before)
    imageio.imwrite(OUT / f"{i:02d}_{c['look']}_{c['place']}_{c['tau']}_after.png", after)
    arm.set_grip_tau(s.TAU_MAX)
    go(az, r, zg + LIFT, s.OPEN, 40)
    box.set_pos(np.array([[1.5 + 0.1 * i, 1.5, 0.0]]))
    go(az, r, 0.15, s.OPEN, 60)
(OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
