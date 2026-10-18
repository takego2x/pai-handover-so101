"""試し：小箱 5 つ（段ボール 3 つ＝見た目は同じで中身が違う、発泡スチロール、金属）の画像を撮り、Gemini に重さを予想させる。

  docker compose run --rm sim python handover/spikes/vlm_boxes.py [--replay]
画像：outputs/handover/vlm/boxes.png、Gemini の答え：outputs/handover/vlm/boxes.gemini.txt（--replay ならそれを読み直すだけ）
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim"))
OUT = ROOT / "outputs" / "handover" / "vlm"
OUT.mkdir(parents=True, exist_ok=True)

# 正解（ロボットにも Gemini にも見せない）
BOXES = [  # (名前, 見た目, 重さ g, x [m])
    ("cardboard_a", "cardboard", 30, -0.12),
    ("foam", "foam", 10, -0.06),
    ("cardboard_b", "cardboard", 150, 0.0),
    ("metal", "metal", 400, 0.06),
    ("cardboard_c", "cardboard", 300, 0.12),
]


def cardboard_texture(size=256):
    """段ボールの絵：茶色の地、横に通るテープ、荷札。3 つとも同じ絵（見た目では中身の重さが分からない）。"""
    rng = np.random.default_rng(0)
    base = np.array([176, 132, 86], float)
    img = base + rng.normal(0, 6, (size, size, 3))       # 紙のむら
    img[:, :, :] += (np.sin(np.arange(size) / 3.0)[None, :, None] * 3)  # 段のすじ
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, size * 0.42, size, size * 0.58], fill=(205, 175, 120))   # テープ
    d.rectangle([size * 0.15, size * 0.65, size * 0.55, size * 0.9], fill=(240, 240, 235), outline=(90, 90, 90))  # 荷札
    for k in range(3):
        d.line([size * 0.19, size * (0.7 + 0.06 * k), size * 0.5, size * (0.7 + 0.06 * k)], fill=(120, 120, 120), width=2)
    return np.asarray(im)


def foam_texture(size=256):
    """発泡スチロールの絵：白地に、粒の丸いむら。"""
    rng = np.random.default_rng(1)
    im = Image.new("RGB", (size, size), (238, 238, 232))
    d = ImageDraw.Draw(im)
    for _ in range(900):
        x, y, r = rng.integers(0, size), rng.integers(0, size), rng.integers(3, 7)
        g = int(rng.integers(215, 250))
        d.ellipse([x - r, y - r, x + r, y + r], outline=(g - 15, g - 15, g - 20), fill=(g, g, g - 4))
    return np.asarray(im)


def metal_texture(size=256):
    """金属の絵：灰色の地に、横のへアライン（研磨のすじ）と、角のねじ。"""
    rng = np.random.default_rng(2)
    row = 150 + rng.normal(0, 10, size)
    img = np.repeat(row[:, None], size, axis=1)[:, :, None].repeat(3, axis=2) + np.array([0, 3, 8])
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for cx, cy in [(24, 24), (232, 24), (24, 232), (232, 232)]:
        d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(95, 98, 102), outline=(60, 60, 60))
        d.line([cx - 6, cy, cx + 6, cy], fill=(50, 50, 50), width=2)
    return np.asarray(im)


def cube_obj(path, side=0.03):
    """各面に絵を貼る座標（UV）を付けた立方体の OBJ。Genesis の基本形 Box には UV がなく、絵（テクスチャ）が描かれない。"""
    h = side / 2
    faces = [((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, -1, 0)), ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
             ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1))]
    v, vt, f = [], [], []
    for n, u, w in faces:
        n, u, w = map(np.array, (n, u, w))
        base = len(v)
        for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            v.append(h * n + h * a * u + h * b * w)
            vt.append(((a + 1) / 2, (b + 1) / 2))
        f.append((base + 1, base + 2, base + 3))
        f.append((base + 1, base + 3, base + 4))
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v] + [f"vt {a} {b}" for a, b in vt]
    lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in f]
    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


def dented_obj(path, side=0.03, depth=0.009, n=8):
    """潰れた箱の OBJ：指で挟まれる 2 つの面（箱の x の ±面）が内側にへこみ、上の面も少し沈む。面ごとに UV を付ける。"""
    h = side / 2
    faces = [((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, -1, 0)), ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
             ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1))]
    v, vt, f = [], [], []
    for nrm, u, w in faces:
        nrm, u, w = map(np.array, (nrm, u, w))
        base = len(v)
        for j in range(n + 1):
            for i in range(n + 1):
                a, b = -1 + 2 * i / n, -1 + 2 * j / n
                bump = (1 - a * a) * (1 - b * b)                    # 面の真ん中ほど深くへこむ
                d = depth if abs(nrm[0]) == 1 else (0.35 * depth if nrm[2] == 1 else 0.0)
                v.append(h * nrm + h * a * u + h * b * w - nrm * d * bump)
                vt.append(((a + 1) / 2, (b + 1) / 2))
        for j in range(n):
            for i in range(n):
                q = base + j * (n + 1) + i + 1
                f.append((q, q + 1, q + n + 2))
                f.append((q, q + n + 2, q + n + 1))
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v] + [f"vt {a:.4f} {b:.4f}" for a, b in vt]
    lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in f]
    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


def crushed_texture(base_fn, size=256):
    """潰れた絵：元の絵を暗くし、しわ（折れ線）を描く。"""
    rng = np.random.default_rng(3)
    im = Image.fromarray((base_fn(size).astype(float) * 0.78).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(9):
        x0, y0 = rng.integers(0, size, 2)
        pts = [(int(x0), int(y0))]
        for _ in range(4):
            pts.append((int(pts[-1][0] + rng.integers(-60, 60)), int(pts[-1][1] + rng.integers(-60, 60))))
        d.line(pts, fill=(60, 40, 25), width=3)
    return np.asarray(im)


def render():
    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning")
    sc = gs.Scene(show_viewer=False)
    sc.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.55, 0.57, 0.6)))
    tex = gs.textures.ImageTexture(image_array=cardboard_texture())
    obj = cube_obj(OUT / "cube_uv.obj")
    for name, look, mass, x in BOXES:
        if look == "cardboard":
            surf = gs.surfaces.Default(diffuse_texture=tex, roughness=0.9)
        elif look == "foam":
            surf = gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_array=foam_texture()), roughness=1.0)
        else:
            surf = gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_array=metal_texture()),
                                       roughness=0.25, metallic=0.9)
        sc.add_entity(gs.morphs.Mesh(file=obj, pos=(x, 0.0, 0.015), fixed=True), surface=surf)
    cam = sc.add_camera(res=(1024, 576), pos=(0.0, -0.20, 0.13), lookat=(0.0, 0.0, 0.01), fov=50, GUI=False)
    sc.build()
    rgb = cam.render()[0]
    Image.fromarray(rgb).save(OUT / "boxes.png")
    return OUT / "boxes.png"


PROMPT = """This image shows small boxes (each about 3 cm) on a table, seen by a robot that must pick them up.
For EVERY box, return one JSON object:
- "point": [y, x] of the box center, normalized to 0-1000
- "looks_like": what the box looks like it is made of
- "weight_g": your best guess of its total weight in grams (contents included), as [low, high]
- "fragile": true if squeezing it hard could crush or damage it
- "confidence": "low", "medium" or "high" for the weight guess. If boxes look identical, say so and give a wide range.
Return only a JSON array, no prose."""


def ask(image):
    import vlm_material as vm
    vm.PROMPT = PROMPT
    return vm.ask_gemini(image)


if __name__ == "__main__":
    img = OUT / "boxes.png"
    if "--replay" not in sys.argv:
        img = render()
        text = ask(img)
        (OUT / "boxes.gemini.txt").write_text(text)
    text = (OUT / "boxes.gemini.txt").read_text()
    body = text.split("```json")[-1].split("```")[0] if "```" in text else text
    items = json.loads(body)
    W, H = Image.open(img).size
    print(f"{'#':>2s} {'answer':40s} {'nearest box (truth)'}")
    for it in sorted(items, key=lambda t: t["point"][1]):
        y, x = it["point"]
        # 画像の左右の位置から、どの箱かを当てる（箱は x で横一列）
        k = int(round((x / 1000 - 0.5) / 0.115 * 1.0 + 2))
        k = min(max(k, 0), len(BOXES) - 1)
        name, look, mass, _ = BOXES[k]
        print(f"  {it.get('looks_like', '')[:18]:18s} {str(it.get('weight_g')):12s} fragile={it.get('fragile')!s:5s} "
              f"conf={it.get('confidence')}   <- {name} ({look}, {mass} g)")
