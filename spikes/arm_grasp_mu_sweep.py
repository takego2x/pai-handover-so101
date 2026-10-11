"""Spike: does object friction (mu) change grasp-and-lift success in Genesis on M5?

Grid of envs: mu x gripper force. Finger friction is set to the floor (0.01) so the
contact pair friction = max(finger, cube) = cube's mu (Genesis rule, contact.py:455).
Success = cube is still lifted (z > 0.10 m) after the lift phase.
"""
import sys
import time

import numpy as np
import torch
import genesis as gs

BACKEND = sys.argv[1] if len(sys.argv) > 1 else "cpu"
MUS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
FORCES = [2.0, 5.0, 10.0, 20.0]  # N per finger
GRID = [(m, f) for m in MUS for f in FORCES]
N = len(GRID)

gs.init(backend=gs.gpu if BACKEND == "gpu" else gs.cpu, logging_level="warning")
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
    rigid_options=gs.options.RigidOptions(box_box_detection=True),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(
    gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)),
    material=gs.materials.Rigid(friction=1.0, rho=4000.0),  # ~0.26 kg
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    material=gs.materials.Rigid(friction=0.01),
)
scene.build(n_envs=N)

motors = np.arange(7)
fingers = np.arange(7, 9)
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

# cube mu per env: base friction 1.0 x ratio
mu = torch.tensor([m for m, _ in GRID], dtype=gs.tc_float, device=gs.device)
cube.set_friction_ratio(mu.unsqueeze(1), links_idx_local=[0])

ee = franka.get_link("hand")
quat = np.tile(np.array([0, 1, 0, 0]), (N, 1))


def move_to(z, steps, finger_cmd):
    pos = np.tile(np.array([0.65, 0.0, z]), (N, 1))
    q = franka.inverse_kinematics(link=ee, pos=pos, quat=quat)
    franka.control_dofs_position(q[:, :-2], motors)
    finger_cmd()
    for _ in range(steps):
        scene.step()


def open_fingers():
    franka.control_dofs_position(np.tile([0.04, 0.04], (N, 1)), fingers)


forces = np.array([[-f, -f] for _, f in GRID])


def close_fingers():
    franka.control_dofs_force(forces, fingers)


t0 = time.time()
move_to(0.25, 150, open_fingers)   # pre-grasp
move_to(0.135, 120, open_fingers)  # reach
move_to(0.135, 100, close_fingers)  # grasp
for z_ in np.linspace(0.135, 0.30, 40):  # gradual lift
    move_to(float(z_), 5, close_fingers)
move_to(0.30, 100, close_fingers)
elapsed = time.time() - t0

z = cube.get_pos()[:, 2].cpu().numpy()
print(f"backend={BACKEND} envs={N} wall={elapsed:.1f}s")
print("mu\\F  " + "  ".join(f"{f:>5.0f}N" for f in FORCES))
for i, m in enumerate(MUS):
    row = z[i * len(FORCES):(i + 1) * len(FORCES)]
    print(f"{m:<5} " + "  ".join(("  OK " if v > 0.10 else "drop ") + f"{v:.2f}"[1:] for v in row))
# physics sanity: hold condition 2*mu*F >= m*g  (m ~ 0.256 kg -> 2.51 N)
print("theory hold (2*mu*F >= 2.51N):")
for m in MUS:
    print(f"{m:<5} " + "  ".join("  OK     " if 2 * m * f >= 2.51 else "drop     " for f in FORCES))
