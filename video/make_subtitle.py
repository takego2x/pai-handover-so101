"""表紙（支給の映像）の直後に入れる、サブタイトルの画面（1280x720、3 秒の映像）を作る。

  uv run --no-project --with pillow==11.3.0 python video/make_subtitle.py
  → outputs/video2/subtitle.png と outputs/video2/subtitle.mp4（絵コンテでは clip として 2 番目に置く）
書体はスライドと同じ Meiryo（太字）。白地に黒、グレーの細い線だけ（スライドと同じく有彩色なし）。
"""
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "video2"
FONT_DIR = Path.home() / "Library" / "Application Support" / "LibreOffice" / "4" / "user" / "fonts"
SUB = "重さも強度もわからない箱を運搬するロボット作業"
SMALL = "Physical AI 応用編1　最終課題"
SEC = 3

W, H = 1280, 720
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
bold = ImageFont.truetype(str(FONT_DIR / "meiryob.ttc"), 46)
reg = ImageFont.truetype(str(FONT_DIR / "meiryo.ttc"), 24)
w = d.textlength(SMALL, font=reg)
d.text(((W - w) / 2, 262), SMALL, font=reg, fill=(0x59, 0x59, 0x59))
w = d.textlength(SUB, font=bold)
d.text(((W - w) / 2, 316), SUB, font=bold, fill=(0x1A, 0x1A, 0x1A))
d.line([(W / 2 - w / 2, 404), (W / 2 + w / 2, 404)], fill=(0x1A, 0x1A, 0x1A), width=2)
png, mp4 = OUT / "subtitle.png", OUT / "subtitle.mp4"
im.save(png)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", str(SEC), "-i", str(png), "-r", "25",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)], check=True)
print(png, mp4)
