"""試し：真上の固定カメラの深度から、箱の位置と向きを出し、シミュレータの値と比べる（答え合わせにだけ真の値を使う）。"""
import math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import so101 as s, softbox as sb, vision as vi  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_softboxes import surface  # noqa
ROOT = Path(__file__).resolve().parents[2]
sc = s.make_scene(); sc.add_entity(s.gs.morphs.Plane())
ents = [s.add_so101(sc, (0, 0), 0, friction=0.1), s.add_so101(sc, (0.44, 0), 180, friction=0.1)]
sc.add_entity(s.gs.morphs.Box(size=(0.05, 0.05, 0.02), pos=(0.22, 0, 0.01), fixed=True), surface=s.gs.surfaces.Default(color=(0.6, 0.6, 0.65)))
rng = np.random.default_rng(3); specs = sb.random_boxes(6, seed=22); boxes = []
for i, sp in enumerate(specs):
    x, y, yaw = rng.uniform(0.1, 0.35), rng.uniform(-0.3, -0.1) if i % 2 else rng.uniform(0.1, 0.3), rng.uniform(0, 90)
    f = sb.xml(ROOT / "outputs/handover/softbox" / f"vis_{i}.xml", sp["k"], sp["mass"], sp["mu"], sp["look"])
    boxes.append(sc.add_entity(s.gs.morphs.MJCF(file=f, pos=(x, y, 0), euler=(0, 0, yaw)), surface=surface(s.gs, sp["look"])))
on_stand = sc.add_entity(s.gs.morphs.MJCF(file=sb.xml(ROOT / "outputs/handover/softbox/vis_stand.xml", 5000, 0.1, 0.8, "cardboard"),
                                          pos=(0.22, 0.004, 0.02), euler=(0, 0, 17)), surface=surface(s.gs, "cardboard"))
boxes.append(on_stand)
cam = vi.add_top_camera(sc, center=(0.22, 0.0), height=1.2, fov=40, res=(1280, 1280))
sc.build(n_envs=1)
for b in boxes: sb.hold_spring(b)
for e in ents: e.set_dofs_position(np.tile(s.FOLD, (1, 1))); e.control_dofs_position(np.tile(s.FOLD, (1, 1)))
for _ in range(50): sc.step()
cands = vi.look(cam)
print("候補", len(cands))
for b in boxes:
    p = b.get_pos().cpu().numpy()[0]; q = b.get_quat().cpu().numpy()[0]
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    c = vi.nearest(cands, p[:2])
    if c is None:
        print(f"  箱 {np.round(p, 3)} → 見つからない"); continue
    dyaw = (math.degrees(c[3] - yaw) + 45) % 90 - 45
    print(f"  箱 位置の差 {math.hypot(c[0]-p[0], c[1]-p[1])*1000:4.1f} mm  向きの差 {dyaw:+5.1f}°  上の面 {c[2]*100:.1f} cm（本当 {(p[2]+0.015)*100:.1f}）  面積 {c[4]:.1f} cm²")
import imageio; rgb = np.asarray(cam.render(rgb=True)[0]); imageio.imwrite(ROOT / "outputs/handover/vision_top.png", rgb)
