"""カメラ（Gemini）の判定：ランダムな潰れる箱を一列に並べた写真を撮り、Gemini に見た目から重さ・壊れやすさ・自信を答えさせる。

  docker compose run --rm -e GEMINI_API_KEY sim python handover/spikes/vlm_softboxes.py --seed 2 --n 6 [--ask]
写真：handover/data/softboxes_seed{seed}_n{n}.png、答え：同じ名前の .gemini.txt（--ask のときだけ呼ぶ。答えは保存して使い回す）
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "sim"))
import softbox as sb  # noqa: E402
from vlm_boxes import cardboard_texture, foam_texture, metal_texture  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "handover" / "data"

PROMPT = """This image shows small boxes (each about 3 cm) on a table, seen by a robot that must pick them up gently.
For EVERY box, return one JSON object:
- "point": [y, x] of the box center, normalized to 0-1000
- "looks_like": what the box looks like it is made of
- "weight_g": your best guess of its total weight in grams (contents included), as [low, high]
- "fragile": true if squeezing it hard could crush or damage it
- "confidence": "low", "medium" or "high" for the weight guess. If boxes look identical, say so and give a wide range.
Return only a JSON array, no prose."""


def surface(gs, look):
    """見た目の絵（作品の vlm_boxes と同じ質感）。潰れる箱（MJCF）にも貼る"""
    if look == "cardboard":
        return gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_array=cardboard_texture()), roughness=0.9)
    if look == "foam":
        return gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_array=foam_texture()), roughness=1.0)
    return gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_array=metal_texture()), roughness=0.25, metallic=0.9)


def render(specs, stem):
    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning")
    sc = gs.Scene(show_viewer=False)
    sc.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.55, 0.57, 0.6)))
    n = len(specs)
    for i, sp in enumerate(specs):
        x = (i - (n - 1) / 2) * 0.06
        f = sb.xml(ROOT / "outputs" / "handover" / "softbox" / f"photo_{stem}_{sp['name']}.xml", sp["k"], sp["mass"], sp["mu"], sp["look"])
        sc.add_entity(gs.morphs.MJCF(file=f, pos=(x, 0.0, 0.0)), surface=surface(gs, sp["look"]))
    w = 0.06 * n
    cam = sc.add_camera(res=(1024, 576), pos=(0.0, -0.10 - 0.55 * w, 0.08 + 0.30 * w), lookat=(0.0, 0.0, 0.01), fov=45, GUI=False)
    sc.build()
    rgb = cam.render()[0]
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(OUT / f"{stem}.png")
    return OUT / f"{stem}.png"


if __name__ == "__main__":   # 引数は、直接実行したときだけ読む（ほかの道具が surface を読み込むときに走らないように）
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--ask", action="store_true")
    args = ap.parse_args()
    stem = f"softboxes_seed{args.seed}_n{args.n}"
    if not args.ask and (OUT / f"{stem}.gemini.txt").exists():
        stem += "_preview"   # Gemini に見せた写真（答えと対の写真）を上書きしない
    specs = sb.random_boxes(args.n, seed=args.seed)
    img = render(specs, stem)
    print("写真", img)
    for sp in specs:
        print("  ", sp)
    if args.ask:
        import vlm_material as vm
        vm.PROMPT = PROMPT
        text = vm.ask_gemini(img)
        (OUT / f"{stem}.gemini.txt").write_text(text)
        print(text)
