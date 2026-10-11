"""S3: Franka の手先を目標位置へ動かす（逆運動学 IK → 関節の位置制御）。"""
import numpy as np
import genesis as gs

gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.build()

motors = np.arange(7)  # 腕の 7 関節（7, 8 番は指）
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
hand = franka.get_link("hand")
down = np.array([0, 1, 0, 0])  # 手先を真下に向ける姿勢（四元数 w,x,y,z）

for target in ([0.65, 0.0, 0.25], [0.65, 0.0, 0.135], [0.45, 0.2, 0.30]):
    q = franka.inverse_kinematics(link=hand, pos=np.array(target), quat=down)
    franka.control_dofs_position(q[:-2], motors)
    for _ in range(300):
        scene.step()
    err = np.linalg.norm(hand.get_pos().cpu().numpy() - np.array(target))
    print(f"target={target}  reached={np.round(hand.get_pos().cpu().numpy(), 3)}  error={err * 1000:.1f} mm")
