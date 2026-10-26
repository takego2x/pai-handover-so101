"""提出動画の図（1280x720）：システム構成、一連の動作の流れ図（失敗の箇所を強調／成功の工夫を強調）。

使い方: uv run --no-project --python 3.12 --with pillow==11.3.0 python video/make_figs.py
  → outputs/video/fig_system.png, fig_flow_fail.png, fig_flow_ok.png
白地に黒の線。色は強調だけ（失敗＝赤、工夫＝青）。
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
OUT = Path(__file__).resolve().parents[1] / "outputs" / "video"
FONTS = ["/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"]
BLACK, GRAY, RED, BLUE = (0, 0, 0), (110, 110, 110), (200, 30, 30), (20, 80, 200)


def font(size, bold=True):
    return ImageFont.truetype(FONTS[0] if bold else FONTS[1], size)


def text_c(d, xy, s, size, fill=BLACK, bold=True):
    """xy を中心に、複数行（\\n 区切り）を書く。"""
    f = font(size, bold)
    lines = s.split("\n")
    lh = size + 8
    y0 = xy[1] - lh * len(lines) / 2 + 4
    for i, ln in enumerate(lines):
        w = d.textlength(ln, font=f)
        d.text((xy[0] - w / 2, y0 + i * lh), ln, font=f, fill=fill)


def box(d, x0, y0, x1, y1, s, size=24, color=BLACK, width=3, fill=None):
    d.rectangle([x0, y0, x1, y1], outline=color, width=width, fill=fill)
    text_c(d, ((x0 + x1) / 2, (y0 + y1) / 2), s, size, color if color != BLACK else BLACK)


def arrow(d, p, q, color=BLACK, width=3, label="", lsize=18, loff=(0, -16)):
    d.line([p, q], fill=color, width=width)
    import math
    a = math.atan2(q[1] - p[1], q[0] - p[0])
    L = 14
    d.polygon([q, (q[0] - L * math.cos(a - 0.4), q[1] - L * math.sin(a - 0.4)),
               (q[0] - L * math.cos(a + 0.4), q[1] - L * math.sin(a + 0.4))], fill=color)
    if label:
        text_c(d, ((p[0] + q[0]) / 2 + loff[0], (p[1] + q[1]) / 2 + loff[1]), label, lsize, color, bold=False)


def title(d, s, sub=""):
    text_c(d, (W / 2, 50), s, 38)
    if sub:
        text_c(d, (W / 2, 95), sub, 22, GRAY, bold=False)


def system():
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    title(d, "システム構成", "シミュレータ Genesis の中に、SO-101 を 2 台。AI（VLM）は見た目から予想し、ロボットは持って確かめる")
    # 上段：カメラ → Gemini
    box(d, 80, 170, 320, 260, "カメラの写真\n（箱 5 つ）", 24)
    box(d, 470, 160, 810, 270, "Gemini Robotics-ER\n（VLM）", 26)
    arrow(d, (320, 215), (470, 215))
    box(d, 940, 160, 1200, 270, "答え（JSON）\n材質・重さ\n壊れやすさ・自信", 20)
    arrow(d, (810, 215), (940, 215))
    # 下段：その 1 → 台座 → その 2 → 大箱
    box(d, 80, 400, 380, 560, "SO-101 その 1\n\n試し掴み → 力加減\n肩の力 → 重さ", 23)
    box(d, 520, 440, 760, 520, "台座", 26)
    box(d, 900, 400, 1200, 560, "SO-101 その 2\n\n情報で持つ\n重さで運ぶ速さ", 23)
    box(d, 940, 620, 1160, 690, "大箱", 24)
    arrow(d, (1070, 270), (230, 400), GRAY, label="探し方（弱い順／強めから）", lsize=20, loff=(-140, -8))
    arrow(d, (380, 480), (520, 480), label="箱を置く", lsize=18)
    arrow(d, (760, 480), (900, 480), label="箱を持つ", lsize=18)
    arrow(d, (380, 590), (900, 590), BLUE, width=4, label="情報：力加減・重さ", lsize=22, loff=(0, 22))
    d.line([(230, 560), (230, 590), (380, 590)], fill=BLUE, width=4)
    d.line([(900, 590), (1050, 590), (1050, 560)], fill=BLUE, width=4)
    arrow(d, (1050, 560), (1050, 620))
    text_c(d, (460, 690), "潰れの判定：2 本の指で挟む力の 0.05 秒平均 ＞ 箱の強さ（段ボール 15 N、発泡 6 N）", 18, GRAY, bold=False)
    im.save(OUT / "fig_system.png")


STEPS = [  # (中身, 失敗の印, 工夫の印)
    ("0 見る\nGemini が予想", "失敗 5\n「石」を信じる", "自信 high の\n答えだけ信じる"),
    ("1 試し掴み\n力加減を探す", "失敗 1\nいきなり最大の力", "弱い順に試し\n挟む力を見張る"),
    ("1 重さを測る\n肩の力の差", "", "同じ姿勢で\n空の手を基準に"),
    ("2 台座に置く\n（その 1）", "", ""),
    ("3 台座から持つ\n（その 2）", "失敗 2・3\n手先が横／弱く握る", "腕の向きで真下\n力加減 × 2\n（潰れの 7 割まで）"),
    ("4 大箱へ運ぶ\n（その 2）", "失敗 4\n余裕なしで速く", "重さで運ぶ時間\n1.5 秒 ×\n（1 ＋ 重さ ÷ 150 g）"),
]


def flow(kind):
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    if kind == "fail":
        title(d, "一連の動作：失敗 5 つの箇所", "これから、わざと失敗させた 5 つを見る")
    else:
        title(d, "一連の動作：成功のための工夫", "これから、5 つの箱を最初から最後まで通しで見る")
    n = len(STEPS)
    x0, gap, bw = 40, 22, (W - 80 - 22 * (n - 1)) / n
    y0, y1 = 300, 420
    color = RED if kind == "fail" else BLUE
    for i, (s, fail, ok) in enumerate(STEPS):
        xa = x0 + i * (bw + gap)
        mark = fail if kind == "fail" else ok
        box(d, xa, y0, xa + bw, y1, s, 21, color if mark else BLACK, 5 if mark else 3)
        if i < n - 1:
            arrow(d, (xa + bw, (y0 + y1) / 2), (xa + bw + gap, (y0 + y1) / 2))
        if mark:
            d.line([(xa + bw / 2, y1), (xa + bw / 2, y1 + 40)], fill=color, width=3)
            d.rectangle([xa - 6, y1 + 40, xa + bw + 6, y1 + 150], fill=(255, 238, 238) if kind == "fail" else (232, 240, 255))
            text_c(d, (xa + bw / 2, y1 + 95), mark, 18, color)
    # 上：箱の流れ
    text_c(d, (W / 2, 200), "箱 5 つ：段ボール 30 g・発泡スチロール 10 g・段ボール 150 g・金属 400 g・段ボール 300 g（段ボールは見た目が同じ）", 20, GRAY, bold=False)
    text_c(d, (W / 2, 245), "見た目で予想 → 持って確かめる → 情報を渡す → 情報で持って運ぶ", 24)
    im.save(OUT / f"fig_flow_{kind}.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    system()
    flow("fail")
    flow("ok")
    print("wrote", OUT / "fig_system.png", OUT / "fig_flow_fail.png", OUT / "fig_flow_ok.png")
