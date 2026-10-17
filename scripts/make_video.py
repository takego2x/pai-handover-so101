"""絵コンテ（JSON）から提出用の動画を 1 本つなぐ。字幕・タイトルは画像で作って重ねる（この ffmpeg には文字描画の機能がないため）。

使い方: python scripts/make_video.py video/storyboard.json outputs/final.mp4

絵コンテの例:
[
  {"card": "μ は幅でしか流通していない", "sub": "PAI 最終課題", "sec": 4},
  {"clip": "outputs/clips/POM-A.mp4", "caption": "戦略 A：代表値 μ=0.61 で速く運ぶ → POM を落とす", "start": 0, "sec": 6},
  {"image": "outputs/eval/tradeoff.png", "caption": "成功率とサイクルタイム", "sec": 5}
]
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
FONT_CANDIDATES = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def font(size):
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    raise SystemExit("日本語フォントが見つかりません（FONT_CANDIDATES を確認）")


def ff(*args):
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, text=True)
    if r.returncode != 0:  # 戻り値を捨てない
        raise SystemExit(f"ffmpeg failed: {' '.join(args)}\n{r.stderr}")


def wrap(d, text, f, width):
    """画面の幅に収まるように折り返す（日本語は空白がないので、1 文字ずつ測る）。"""
    lines, cur = [], ""
    for ch in text:
        if d.textlength(cur + ch, font=f) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + [cur] if cur else lines


def draw_centered(d, text, y, size, fill):
    """中央に書く。幅を超えたら折り返し、書いた行の数を返す。"""
    f = font(size)
    lines = wrap(d, text, f, W - 100)
    for k, ln in enumerate(lines):
        w = d.textlength(ln, font=f)
        d.text(((W - w) / 2, y + k * (size + 12)), ln, font=f, fill=fill)
    return len(lines)


def card_png(path, title, sub=""):
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    draw_centered(d, title, H / 2 - 50, 52, "black")
    if sub:
        draw_centered(d, sub, H / 2 + 30, 30, (90, 90, 90))
    im.save(path)


def caption_png(path, text, size=30):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    n = len(wrap(d, text, font(size), W - 100))
    top = H - 30 - n * (size + 12)
    d.rectangle([0, top - 16, W, H], fill=(0, 0, 0, 170))
    draw_centered(d, text, top, size, "white")
    im.save(path)


def segment(seg, i, tmp):
    out = tmp / f"seg{i:02d}.mp4"
    enc = ["-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an"]
    scale = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:white"
    if "card" in seg:
        png = tmp / f"card{i:02d}.png"
        card_png(png, seg["card"], seg.get("sub", ""))
        ff("-loop", "1", "-t", str(seg["sec"]), "-i", str(png), *enc, str(out))
        return out
    sp = float(seg.get("speed", 1))  # 倍速（2 なら 2 倍速。字幕に「2 倍速」と書くこと：提出の決まり）
    src = ["-loop", "1", "-t", str(seg["sec"]), "-i", seg["image"]] if "image" in seg else \
        ["-ss", str(seg.get("start", 0)), "-t", str(seg["sec"] * sp), "-i", seg["clip"]]
    if sp != 1:
        scale = f"setpts=PTS/{sp},{scale}"
    if seg.get("caption"):
        cap = tmp / f"cap{i:02d}.png"
        caption_png(cap, seg["caption"])
        ff(*src, "-i", str(cap), "-filter_complex", f"[0:v]{scale},setsar=1[v];[v][1:v]overlay=0:0", *enc, str(out))
    else:
        ff(*src, "-vf", scale, *enc, str(out))
    return out


def main():
    board, out = json.load(open(sys.argv[1])), Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        segs = [segment(s, i, tmp) for i, s in enumerate(board)]
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in segs))
        ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out))
    # 出口で実測する：長さと解像度を測って、提出条件（1〜3 分）と照らす
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height",
                        "-of", "json", str(out)], capture_output=True, text=True)
    info = json.loads(r.stdout)
    dur = float(info["format"]["duration"])
    print(f"{out}  {dur:.1f}s  {info['streams'][0]['width']}x{info['streams'][0]['height']}"
          f"  {'OK (1-3 min)' if 60 <= dur <= 180 else 'NOT within 1-3 min'}")


if __name__ == "__main__":
    main()
