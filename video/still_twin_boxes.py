"""見た目が同じで中身が違う 2 箱（seed 22 の段ボール box2・box3）を、寄りの静止画にする（「現実の場面」のページの右の絵）。

  docker compose run --rm sim python video/still_twin_boxes.py
  → outputs/video2/twin_boxes.png
"""
import sys
from pathlib import Path

import imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "handover"))
sys.path.insert(0, str(ROOT / "handover" / "spikes"))
import so101 as s  # noqa: E402
import softbox as sb  # noqa: E402
from vlm_softboxes import surface  # noqa: E402

specs = sb.random_boxes(6, seed=22)
twins = [sp for sp in specs if sp["look"] == "cardboard"]
assert len(twins) == 2, twins
sc = s.make_scene()
sc.add_entity(s.gs.morphs.Plane(), surface=s.gs.surfaces.Default(color=(0.16, 0.18, 0.21)))
for x, sp in zip((-0.025, 0.025), twins):
    f = sb.xml(ROOT / "outputs" / "handover" / "softbox" / f"twin_{sp['name']}.xml", sp["k"], sp["mass"], sp["mu"], sp["look"])
    sc.add_entity(s.gs.morphs.MJCF(file=f, pos=(x, 0, 0), euler=(0, 0, 0)), surface=surface(s.gs, sp["look"]))
cam = sc.add_camera(res=(960, 540), pos=(0.0, -0.13, 0.085), lookat=(0.0, 0.0, 0.012), fov=40, GUI=False)
sc.build(n_envs=1)
for _ in range(20):
    sc.step()
rgb = np.asarray(cam.render(rgb=True)[0])
out = ROOT / "outputs" / "video2" / "twin_boxes.png"
imageio.imwrite(out, rgb)
print(out, [(sp["name"], sp["mass"], sp["strength"]) for sp in twins])
