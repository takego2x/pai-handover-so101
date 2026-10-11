"""V16 用：くちばしの試験台で、①押し付け力 ②持ち上げの加速度 ③物体とくちばしのずれ を測る。
使い方: python verify/beak_probe.py <assets> <pyramidal|elliptic>
μ=0.5 に固定し、閉じる力 0.25 / 0.5 / 1.0 N の 3 通り（0.25N は理論では保持：2μF=0.25N > mg=0.195N）。
"""
import os
import sys

import numpy as np
import torch
import genesis as gs

ASSETS, CONE = os.path.abspath(sys.argv[1]), sys.argv[2]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
GRID = [(0.5, 0.25), (0.5, 0.5), (0.5, 1.0)]
N, OBJ = len(GRID), 0.012

# 試験台の MJCF は本線（sim/beak_mu_sweep.py）と同じものを使う
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim", "beak_mu_sweep.py")).read()
xml_block = src[src.index('XML = f"""'): src.index('"""\nimport os; os.makedirs')] + '"""'
Q, VIS = 'quat="0.5 0.5 0.5 0.5"', 'contype="0" conaffinity="0"'
exec(xml_block)  # noqa: S102  XML 変数を作る（ASSETS / Q / VIS を使う）
os.makedirs("outputs/verify", exist_ok=True)
open("outputs/verify/beak_probe.xml", "w").write(XML)  # noqa: F821

gs.init(backend=gs.cpu, logging_level="error")
cone = gs.friction_cone.elliptic if CONE == "elliptic" else gs.friction_cone.pyramidal
ropt = gs.options.RigidOptions(friction_cone=cone, impratio=10.0) if CONE == "elliptic" else gs.options.RigidOptions()
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10), rigid_options=ropt, show_viewer=False)
scene.add_entity(gs.morphs.Plane())
stand = scene.add_entity(gs.morphs.MJCF(file="outputs/verify/beak_probe.xml"))
obj = scene.add_entity(gs.morphs.Box(size=(OBJ,) * 3, pos=(-0.0092 - OBJ / 2 - 0.0005, 0.0, OBJ / 2)),
                       material=gs.materials.Rigid(friction=1.0, rho=11500.0))
scene.build(n_envs=N)
obj.set_friction_ratio(torch.tensor([m for m, _ in GRID], dtype=gs.tc_float, device=gs.device).unsqueeze(1),
                       links_idx_local=[0])
LIFT, JAW = 0, 1
stand.set_dofs_kp(np.array([2000.0, 200.0]))
stand.set_dofs_kv(np.array([200.0, 20.0]))
mass = OBJ ** 3 * 11500
forces = np.array([[-f] for _, f in GRID])


def step(lift, jaw_open, n, rec=None):
    stand.control_dofs_position(np.full((N, 1), lift), [LIFT])
    if jaw_open:
        stand.control_dofs_position(np.full((N, 1), OBJ + 0.006), [JAW])
    else:
        stand.control_dofs_force(forces, [JAW])
    for _ in range(n):
        scene.step()
        if rec is not None:
            rec.append((stand.get_dofs_position([LIFT]).cpu().numpy()[:, 0].copy(), obj.get_pos()[:, 2].cpu().numpy().copy()))


step(0.0, True, 100)
step(-0.122, True, 150)
step(-0.122, False, 100)

# ① 噛んだ直後の、接触点ごとの力（物体が受ける力）
c = obj.get_contacts(with_entity=stand)
obj_links = {int(l.idx) for l in obj.links}
print(f"== {CONE}")
for e, (mu, F) in enumerate(GRID):
    valid = c["valid_mask"][e].cpu().numpy().astype(bool)
    la = c["link_a"][e].cpu().numpy()[valid]
    fa, fb = c["force_a"][e].cpu().numpy()[valid], c["force_b"][e].cpu().numpy()[valid]
    pos = c["position"][e].cpu().numpy()[valid]
    f = np.array([fa[i] if la[i] in obj_links else fb[i] for i in range(len(la))])
    side = np.sign(pos[:, 0] - obj.get_pos()[e, 0].item())
    sq = [np.abs(f[side == s, 0]).sum() for s in (-1, 1)]
    print(f"[1 force] mu={mu} F={F}N  contacts={len(la)}  squeeze per side={sq[0]:.3f} / {sq[1]:.3f} N  "
          f"vertical/horizontal={np.abs(f[:, 2]).sum() / max(1e-9, np.abs(f[:, 0]).sum()):.2f}  "
          f"contact heights={sorted(set(np.round(pos[:, 2] * 1000, 1)))} mm")

# ② ③ 持ち上げ中の加速度と、物体とくちばしのずれ
rec = []
for z in np.linspace(-0.122, 0.0, 40):
    step(float(z), False, 5, rec)
lift = np.array([r[0] for r in rec])
rel = np.array([r[1] - r[0] for r in rec])
rel -= rel[0]
acc = np.diff(lift, 2, axis=0) / 0.01 ** 2
for e, (mu, F) in enumerate(GRID):
    a_lim = 2 * mu * F / mass - 9.81
    r = rel[:, e] * 1000
    print(f"[2 accel] mu={mu} F={F}N  limit={a_lim:5.2f}  measured max={acc[:, e].max():.2f} m/s2"
          f"   [3 slip] 0.5s {r[49]:+7.2f}  1.0s {r[99]:+7.2f}  2.0s {r[-1]:+7.2f} mm")
