"""Spike: two Franka arms in one Genesis scene, each grasps and lifts its own cube simultaneously."""
import time
import numpy as np
import genesis as gs

gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
                 rigid_options=gs.options.RigidOptions(box_box_detection=True), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
arms, cubes, targets = [], [], []
for y in (-0.5, 0.5):  # two arms facing +x, 1.0 m apart
    arms.append(scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, y, 0)),
                                 material=gs.materials.Rigid(friction=0.01)))
    cubes.append(scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.6, y, 0.02)),
                                  material=gs.materials.Rigid(friction=0.5, rho=4000.0)))
    targets.append((0.6, y))
scene.build()
m, f = np.arange(7), np.arange(7, 9)
for a in arms:
    a.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    a.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))

def step_all(z, n, close):
    for a, (x, y) in zip(arms, targets):
        q = a.inverse_kinematics(link=a.get_link("hand"), pos=np.array([x, y, z]), quat=np.array([0, 1, 0, 0]))
        a.control_dofs_position(q[:-2], m)
        a.control_dofs_force(np.array([-10.0, -10.0]), f) if close else a.control_dofs_position(np.array([0.04, 0.04]), f)
    for _ in range(n):
        scene.step()

t0 = time.time()
step_all(0.25, 150, False); step_all(0.135, 120, False); step_all(0.135, 100, True)
for z in np.linspace(0.135, 0.30, 40):
    step_all(float(z), 5, True)
step_all(0.30, 100, True)
print(f"wall={time.time()-t0:.1f}s", ["lifted z=%.2f" % c.get_pos()[2] for c in cubes])
