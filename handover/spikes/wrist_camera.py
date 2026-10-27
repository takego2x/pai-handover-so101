"""試し：手首のカメラを、公式モデル（so101_new_calib_camera.xml、SO-ARM100 a758567）の取り付け位置に付ける。

  公式のカメラ部品（wrist_camera_so101_v1.stl）は 32 mm 角。手を真下に向けたとき、指先の方を向くのは部品の +z（実測：指先への向きと cos 0.981）。
  Genesis のカメラは自分の −z を見るので、x 軸のまわりに 180° 回して付ける。
  1 台で、箱の上に手を下ろし → 閉じる → 2 cm 持ち上げる。各場面で、手首のカメラの画像と深度、外からの画を保存。
  「持てたか」の判断（持ち上げる前後で、箱までの距離が変わらないか）に使える画素の範囲を探す。

  docker compose run --rm sim python handover/spikes/wrist_camera.py
  → outputs/handover/wrist_camera/*.png
"""
import math
import sys
from pathlib import Path

import imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "handover"))
import so101 as s  # noqa: E402
import softbox as sb  # noqa: E402

OUT = ROOT / "outputs" / "handover" / "wrist_camera"
OUT.mkdir(parents=True, exist_ok=True)
CAM_MODEL = str(Path(s.MODEL).with_name("so101_new_calib_camera.xml"))
LENS = (0.0, 0.0, 0.0275)           # 部品の +z の面（0.0263）の少し前

sc = s.make_scene()
sc.add_entity(s.gs.morphs.Plane())
e = sc.add_entity(s.gs.morphs.MJCF(file=CAM_MODEL, pos=(0, 0, 0), euler=(0, 0, 0), decompose_robot_error_threshold=0.0),
                  material=s.gs.materials.Rigid(friction=0.1))
BOX = (0.20 + s.JAW, 0.0)
f = sb.xml(OUT / "box.xml", 8000, 0.1, 0.8, "cardboard")
box = sc.add_entity(s.gs.morphs.MJCF(file=f, pos=(BOX[0], BOX[1], 0.0)))
hand = sc.add_camera(res=(320, 240), fov=70, near=0.005, GUI=False)
ext = sc.add_camera(res=(960, 540), pos=(0.30, -0.16, 0.12), lookat=(0.21, 0.0, 0.05), fov=45, GUI=False)
sc.build(n_envs=1)
sb.hold_spring(box)
arm = s.Arm(sc, e, (0, 0), 0, 1)
T = np.diag([1.0, -1.0, -1.0, 1.0])  # x 軸のまわりに 180°：カメラの −z ＝ 部品の +z
T[:3, 3] = LENS
hand.attach(e.get_link("wrist_camera"), T)


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


def shot(tag):
    hand.move_to_attach()
    rgb, depth, _, _ = hand.render(rgb=True, depth=True)
    d = np.asarray(depth)
    imageio.imwrite(OUT / f"hand_{tag}.png", np.asarray(rgb))
    dv = np.clip((d - 0.0) / 0.25, 0, 1)
    imageio.imwrite(OUT / f"depth_{tag}.png", (255 * (1 - dv)).astype(np.uint8))
    imageio.imwrite(OUT / f"ext_{tag}.png", np.asarray(ext.render(rgb=True)[0]))
    np.save(OUT / f"depth_{tag}.npy", d)
    h, w = d.shape
    print(f"{tag:8s} 中央 {np.median(d[h//2-8:h//2+8, w//2-8:w//2+8]):.4f}  最小 {d.min():.4f}  "
          f"箱 z {float(box.get_pos().cpu().numpy()[0, 2]):.3f}  指 {float(e.get_dofs_position().cpu().numpy()[0, 5]):.2f}", flush=True)


arm.set_grip_tau(s.TAU_MAX)
go(0.0, 0.2, 0.08, s.OPEN, 150)
shot("above")
go(0.0, 0.2, 0.020, s.OPEN, 100)
go(0.0, 0.2, 0.020, s.PRE, 40)
shot("open")
arm.set_grip_tau(0.4)
go(0.0, 0.2, 0.020, s.CLOSE, 200)
shot("closed")
go(0.0, 0.2, 0.040, s.CLOSE, 90)
shot("lifted")
# 箱を離して、空の手で同じ高さ（持てなかったときの見え方）
arm.set_grip_tau(s.TAU_MAX)
go(0.0, 0.2, 0.040, s.OPEN, 60)
for _ in range(60):
    sc.step()
shot("dropped")
