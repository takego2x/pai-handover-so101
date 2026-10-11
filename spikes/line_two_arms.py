"""Lv1 の試作：アーム A がトレイ側から掴んで置き台に置き、向かい合うアーム B が置き台から取って箱側に置く。
動きが成立するか（受け渡しで落とさないか、2 台がぶつからないか）だけを見る。μ と加速度の戦略はまだ入れない。
使い方: python spikes/line_two_arms.py [--record out.mp4]
"""
import argparse

import numpy as np
import genesis as gs

ap = argparse.ArgumentParser()
ap.add_argument("--record")
args = ap.parse_args()

SIDE = 0.04
PICK = np.array([0.55, -0.25])    # トレイ側（A の受け持ち）
STAND = np.array([0.55, 0.00])    # 置き台（A と B の受け渡し）
PLACE = np.array([0.55, 0.25])    # 箱側（B の受け持ち）
STAND_H = 0.06                    # 置き台の高さ
OFFSET = 0.115                    # 手先リンクから指先の中心まで

gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
                 rigid_options=gs.options.RigidOptions(box_box_detection=True, friction_cone=gs.friction_cone.elliptic, impratio=10.0), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Box(size=(0.12, 0.12, STAND_H), pos=(*STAND, STAND_H / 2), fixed=True),
                 surface=gs.surfaces.Plastic(color=(0.6, 0.6, 0.6)))
cube = scene.add_entity(gs.morphs.Box(size=(SIDE,) * 3, pos=(*PICK, SIDE / 2)),
                        material=gs.materials.Rigid(friction=0.5, rho=1400.0),
                        surface=gs.surfaces.Plastic(color=(0.9, 0.3, 0.1)))
arm_a = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, 0.0)),
                         material=gs.materials.Rigid(friction=0.01))
arm_b = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(1.10, 0.0, 0.0),
                                        euler=(0, 0, 180)),  # A と向かい合わせ
                         material=gs.materials.Rigid(friction=0.01))
cam = scene.add_camera(res=(960, 540), pos=(0.55, -1.6, 0.9), lookat=(0.55, 0.0, 0.1), fov=45, GUI=False) if args.record else None
scene.build()

kp = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
kv = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
for arm in (arm_a, arm_b):
    arm.set_dofs_kp(kp)
    arm.set_dofs_kv(kv)
motors, fingers, down = np.arange(7), np.arange(7, 9), np.array([0, 1, 0, 0])
if cam:
    cam.start_recording(save_to_filename=args.record, fps=50)


def cmd(arm, xy, z, grip):
    q = arm.inverse_kinematics(link=arm.get_link("hand"), pos=np.array([*xy, z]), quat=down)
    arm.control_dofs_position(q[:-2], motors)
    if grip:
        arm.control_dofs_force(np.array([-10.0, -10.0]), fingers)
    else:
        arm.control_dofs_position(np.array([0.04, 0.04]), fingers)


def hold(n):
    for _ in range(n):
        scene.step()


def move(arm, xy0, xy1, z0, z1, grip, n=100, settle=50):
    for t in np.linspace(0, 1, n):
        cmd(arm, xy0 + (xy1 - xy0) * t, z0 + (z1 - z0) * t, grip)
        scene.step()
    hold(settle)  # 目標に届くまで待つ（指令を出しただけでは届いていない：S3 と同じ）


HOME_A, HOME_B = np.array([0.40, -0.25]), np.array([0.70, 0.25])
z_pick, z_stand, z_up = SIDE / 2 + OFFSET, STAND_H + SIDE / 2 + OFFSET, 0.35
log = []

cmd(arm_a, HOME_A, z_up, False); cmd(arm_b, HOME_B, z_up, False); hold(150)
# A：トレイ側で掴む → 置き台へ → 離す → 退く
move(arm_a, HOME_A, PICK, z_up, z_up, False); move(arm_a, PICK, PICK, z_up, z_pick, False)
cmd(arm_a, PICK, z_pick, True); hold(80)
move(arm_a, PICK, PICK, z_pick, z_up, True); move(arm_a, PICK, STAND, z_up, z_up, True)
move(arm_a, STAND, STAND, z_up, z_stand, True)
cmd(arm_a, STAND, z_stand, False); hold(60)
log.append(("A placed on stand", cube.get_pos().cpu().numpy().round(3)))
move(arm_a, STAND, STAND, z_stand, z_up, False); move(arm_a, STAND, HOME_A, z_up, z_up, False)
# B：置き台で掴む → 箱側へ → 置く
move(arm_b, HOME_B, STAND, z_up, z_up, False); move(arm_b, STAND, STAND, z_up, z_stand, False)
cmd(arm_b, STAND, z_stand, True); hold(80)
move(arm_b, STAND, STAND, z_stand, z_up, True)
log.append(("B lifted from stand", cube.get_pos().cpu().numpy().round(3)))
move(arm_b, STAND, PLACE, z_up, z_up, True); move(arm_b, PLACE, PLACE, z_up, z_pick, True)
cmd(arm_b, PLACE, z_pick, False); hold(60)
move(arm_b, PLACE, PLACE, z_pick, z_up, False)
final = cube.get_pos().cpu().numpy()
log.append(("B placed", final.round(3)))
if cam:
    cam.stop_recording()

for k, v in log:
    print(f"{k:22s} cube at {v}")
ok = np.linalg.norm(final[:2] - PLACE) < 0.03 and final[2] < 0.05
print("LINE OK" if ok else "LINE FAILED", f"(distance to target {np.linalg.norm(final[:2] - PLACE) * 1000:.0f} mm)")
