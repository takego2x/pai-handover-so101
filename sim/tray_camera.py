"""S7a: 材質の違う部品をトレイに並べ、上から撮った画像と正解（材質・画素位置）を書き出す。
使い方: python sim/tray_camera.py outputs/tray/tray_0.png [seed]
"""
import json
import sys

import numpy as np
import genesis as gs

out = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = np.random.default_rng(seed)

# (label, surface, shape, size) — appearance follows common industrial practice, not a guarantee
PARTS = [
    ("POM", gs.surfaces.Plastic(color=(0.95, 0.95, 0.92), roughness=0.6), "box", (0.04, 0.03, 0.02)),
    ("ABS", gs.surfaces.Plastic(color=(0.08, 0.08, 0.08), roughness=0.4), "box", (0.05, 0.03, 0.02)),
    ("aluminium", gs.surfaces.Metal(color=(0.80, 0.81, 0.83), roughness=0.3, metallic=1.0), "cylinder", (0.015, 0.04)),
    ("steel", gs.surfaces.Metal(color=(0.42, 0.44, 0.47), roughness=0.2, metallic=1.0), "cylinder", (0.012, 0.05)),
    ("NBR", gs.surfaces.Rough(color=(0.15, 0.12, 0.10)), "box", (0.03, 0.03, 0.015)),
]

gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(show_viewer=False, renderer=gs.renderers.Rasterizer())
scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough(color=(0.35, 0.37, 0.40)))
scene.add_entity(gs.morphs.Box(size=(0.36, 0.26, 0.01), pos=(0.0, 0.0, 0.005), fixed=True),
                 surface=gs.surfaces.Plastic(color=(0.15, 0.35, 0.65)))  # blue tray

slots = [(-0.11, -0.06), (0.0, -0.06), (0.11, -0.06), (-0.055, 0.06), (0.055, 0.06)]
order = rng.permutation(len(PARTS))
truth, ents = [], []
for slot, k in zip(slots, order):
    label, surf, shape, size = PARTS[k]
    x, y = slot[0] + rng.uniform(-0.01, 0.01), slot[1] + rng.uniform(-0.01, 0.01)
    if shape == "box":
        morph = gs.morphs.Box(size=size, pos=(x, y, 0.01 + size[2] / 2),
                              euler=(0, 0, float(rng.uniform(0, 90))))
    else:  # cylinder lying on its side
        r, h = size
        morph = gs.morphs.Cylinder(radius=r, height=h, pos=(x, y, 0.01 + r),
                                   euler=(90, 0, float(rng.uniform(0, 180))))
    ents.append(scene.add_entity(morph, surface=surf))
    truth.append({"label": label, "x": round(x, 3), "y": round(y, 3)})

cam = scene.add_camera(res=(960, 720), pos=(0.0, -0.05, 0.55), lookat=(0.0, 0.0, 0.0), fov=40, GUI=False)
scene.build()
scene.step()
rgb, _, seg, _ = cam.render(rgb=True, segmentation=True)

import imageio  # noqa: E402

import os  # noqa: E402

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
imageio.imwrite(out, rgb)
H, W = seg.shape[:2]
# 画像の番号 → 物体 の対応は Genesis の対応表で引く（番号は物体の idx と一致しない。背景が 0 に入るため）
idxc_of = {}
for idxc, key in scene.segmentation_idx_dict.items():
    ent_idx = key[0] if isinstance(key, tuple) else key
    idxc_of.setdefault(ent_idx, []).append(idxc)
for t, e in zip(truth, ents):  # セグメンテーションで部品ごとの画素の重心を求める（正解の画素位置）
    ys, xs = np.nonzero(np.isin(seg, idxc_of.get(e.idx, [])))
    if len(xs) == 0:
        raise SystemExit(f"part {t['label']} not visible in the image")
    t["px"], t["py"] = int(xs.mean()), int(ys.mean())
    t["point_1000"] = [round(ys.mean() / H * 1000), round(xs.mean() / W * 1000)]  # Gemini と同じ [y, x]、0〜1000
json.dump(truth, open(out.replace(".png", ".json"), "w"), indent=1)
print(out, [t["label"] for t in truth])
