"""G0 spike: Microduck head test stand — can the beak hold (咥える) an object, and does mu decide it?

The official Microduck MJCF has no jaw joint (the head is one rigid body). Here the lower jaw
(jaw_soft + jaw meshes) is split off onto a slide joint along head-x (parallel-jaw approximation
of the real linkage). The head is mounted beak-down on a world z-slider so it can lift.
Pads friction = 0.01, so pair friction = object's mu (Genesis uses max of the pair).
Success = object still held after lifting (z > 0.05 m).
"""
import sys
import time

import numpy as np
import torch
import genesis as gs

ASSETS = __import__("os").path.abspath(sys.argv[1])  # .../robot/microduck/assets (meshdir resolves relative to the XML, so make it absolute)
MUS = [0.1, 0.2, 0.3, 0.5, 0.8]
FORCES = [0.25, 0.5, 1.0, 2.0]  # N, beak closing force
GRID = [(m, f) for m in MUS for f in FORCES]
N = len(GRID)
OBJ = 0.012  # 12 mm cube, ~20 g

Q = 'quat="0.5 0.5 0.5 0.5"'
VIS = 'contype="0" conaffinity="0"'
XML = f"""
<mujoco model="beak_stand">
  <compiler angle="radian" meshdir="{ASSETS}"/>
  <asset>
    <mesh name="soft_mouth_top" file="soft_mouth_top.stl"/>
    <mesh name="jaw_soft" file="jaw_soft.stl"/>
    <mesh name="jaw" file="jaw.stl"/>
    <mesh name="top_head_shell" file="top_head_shell.stl"/>
    <mesh name="face_part" file="face_part.stl"/>
  </asset>
  <worldbody>
    <body name="lift" pos="0 0 0.20">
      <joint name="lift_z" type="slide" axis="0 0 1" range="-0.2 0.2"/>
      <inertial pos="0 0 0" mass="0.3" diaginertia="1e-3 1e-3 1e-3"/>
      <body name="head">
        <inertial pos="0 0 -0.03" mass="0.19" diaginertia="3e-4 3e-4 1.5e-4"/>
        <geom type="mesh" mesh="soft_mouth_top" pos="-0.0042 0 -0.0181" {Q} friction="0.01" rgba="0.85 0.76 0.87 1"/>
        <geom type="mesh" mesh="top_head_shell" pos="-0.0045 0 -0.018" {Q} {VIS} rgba="1 0.85 0.2 1"/>
        <geom type="mesh" mesh="face_part" pos="-0.0045 0 -0.0178" {Q} {VIS} rgba="0.2 0.2 0.2 1"/>
        <body name="lower_jaw">
          <joint name="jaw_open" type="slide" axis="-1 0 0" range="0 0.03"/>
          <inertial pos="-0.01 0 -0.06" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>
          <geom type="mesh" mesh="jaw_soft" pos="-0.0046 0 -0.018" {Q} friction="0.01" rgba="0.85 0.76 0.87 1"/>
          <geom type="mesh" mesh="jaw" pos="-0.0045 0 -0.018" {Q} {VIS} rgba="1 0.4 0.12 1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
import os; os.makedirs("outputs", exist_ok=True); open("outputs/beak_stand.xml", "w").write(XML)

gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
                 rigid_options=gs.options.RigidOptions(friction_cone=gs.friction_cone.elliptic, impratio=10.0), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
stand = scene.add_entity(gs.morphs.MJCF(file="outputs/beak_stand.xml"))
# object centered between the pads (pad mid-plane x ~ -0.0083 when closed; mouth spans z -0.076..-0.044 below head)
obj = scene.add_entity(
    gs.morphs.Box(size=(OBJ, OBJ, OBJ), pos=(-0.0092 - OBJ / 2 - 0.0005, 0.0, OBJ / 2)),
    material=gs.materials.Rigid(friction=1.0, rho=11500.0),  # 12mm cube -> ~20 g
)
scene.build(n_envs=N)
mu = torch.tensor([m for m, _ in GRID], dtype=gs.tc_float, device=gs.device)
obj.set_friction_ratio(mu.unsqueeze(1), links_idx_local=[0])

LIFT, JAW = 0, 1  # dof order in the stand entity
stand.set_dofs_kp(np.array([2000.0, 200.0]))
stand.set_dofs_kv(np.array([200.0, 20.0]))
mass = float(obj.get_mass()) if hasattr(obj, "get_mass") else OBJ**3 * 11500


def run(lift_target, jaw_cmd, steps):
    stand.control_dofs_position(np.full((N, 1), lift_target), [LIFT])
    jaw_cmd()
    for _ in range(steps):
        scene.step()


def jaw_open():
    stand.control_dofs_position(np.full((N, 1), OBJ + 0.006), [JAW])


forces = np.array([[-f] for _, f in GRID])


def jaw_close():
    stand.control_dofs_force(forces, [JAW])


t0 = time.time()
run(0.0, jaw_open, 100)  # open above
# lower so the mouth (z -0.076..-0.044 under head) brackets the cube: head at z ~ 0.20 + d
run(-0.122, jaw_open, 150)  # head origin ~ 0.062 m above floor
run(-0.122, jaw_close, 100)  # bite
for z in np.linspace(-0.122, 0.0, 40):
    run(float(z), jaw_close, 5)
run(0.0, jaw_close, 100)  # hold 1 s at top
wall = time.time() - t0

z = obj.get_pos()[:, 2].cpu().numpy()
print(f"object mass ~{mass*1000:.1f} g  mg={mass*9.81:.3f} N  envs={N} wall={wall:.1f}s")
print("mu\\F  " + "  ".join(f"{f:>6.2f}N" for f in FORCES))
for i, m in enumerate(MUS):
    row = z[i * len(FORCES):(i + 1) * len(FORCES)]
    print(f"{m:<5} " + "  ".join(("  held " if v > 0.05 else "  drop ") for v in row) + "   z=" + " ".join(f"{v:.3f}" for v in row))
print("theory 2*mu*F >= mg:")
for m in MUS:
    print(f"{m:<5} " + "  ".join("  held " if 2 * m * f >= mass * 9.81 else "  drop " for f in FORCES))
