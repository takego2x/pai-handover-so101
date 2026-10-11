"""S4a: μ と把持力を 1 組だけ決めて、掴んで持ち上げる。
使い方: python sim/s4_grasp_one.py <mu> <force_N> <substeps>
"""
import sys

import numpy as np
import genesis as gs

MU, FORCE, SUBSTEPS = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])

gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=SUBSTEPS),
                 rigid_options=gs.options.RigidOptions(box_box_detection=True, friction_cone=gs.friction_cone.elliptic, impratio=10.0), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)),
                        material=gs.materials.Rigid(friction=MU, rho=4000.0))  # 約 0.26 kg
# 接触の μ は「2 物体の大きいほう」で決まるので、指は最小値 0.01 にして箱の μ を効かせる
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
                          material=gs.materials.Rigid(friction=0.01))
scene.build()

motors, fingers = np.arange(7), np.arange(7, 9)
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
hand, down = franka.get_link("hand"), np.array([0, 1, 0, 0])


def go(z, steps, grip):
    q = franka.inverse_kinematics(link=hand, pos=np.array([0.65, 0.0, z]), quat=down)
    franka.control_dofs_position(q[:-2], motors)
    if grip:
        franka.control_dofs_force(np.array([-FORCE, -FORCE]), fingers)  # 指を閉じる力
    else:
        franka.control_dofs_position(np.array([0.04, 0.04]), fingers)  # 指を開く
    for _ in range(steps):
        scene.step()


go(0.25, 150, False)   # 箱の上へ
go(0.135, 120, False)  # 下ろす
go(0.135, 100, True)   # 掴む
for z in np.linspace(0.135, 0.30, 40):  # ゆっくり持ち上げる
    go(float(z), 5, True)
go(0.30, 100, True)    # 1 秒保持

z = float(cube.get_pos()[2])
m_g = 0.04 ** 3 * 4000 * 9.81
print(f"mu={MU} F={FORCE}N substeps={SUBSTEPS} -> cube z={z:.3f} m  {'HELD' if z > 0.10 else 'DROPPED'}"
      f"   (theory: 2*mu*F={2 * MU * FORCE:.2f}N vs mg={m_g:.2f}N -> {'HELD' if 2 * MU * FORCE >= m_g else 'DROPPED'})")
