"""提出動画（潰れる箱の版）の図：コンセプト一覧、カメラ（Gemini）の判定、コンセプト図。1280x720、白地に黒の線、強調だけ色。

使い方: uv run --no-project --python 3.12 --with pillow==11.3.0 python video/make_figs2.py
  → outputs/video2/fig_concepts.png, fig_camera.png, fig_summary.png
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "video2"
FONTS = ["/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"]
BLACK, GRAY, BLUE, RED = (0, 0, 0), (110, 110, 110), (20, 80, 200), (200, 30, 30)


def font(size, bold=True):
    return ImageFont.truetype(FONTS[0] if bold else FONTS[1], size)


def text_c(d, xy, s, size, fill=BLACK, bold=True):
    f = font(size, bold)
    lines = s.split("\n")
    lh = size + 8
    y0 = xy[1] - lh * len(lines) / 2 + 4
    for i, ln in enumerate(lines):
        d.text((xy[0] - d.textlength(ln, font=f) / 2, y0 + i * lh), ln, font=f, fill=fill)


def text_l(d, xy, s, size, fill=BLACK, bold=False):
    d.text(xy, s, font=font(size, bold), fill=fill)


def box(d, x0, y0, x1, y1, s, size=22, color=BLACK, width=3, fill=None):
    d.rectangle([x0, y0, x1, y1], outline=color, width=width, fill=fill)
    text_c(d, ((x0 + x1) / 2, (y0 + y1) / 2), s, size, color)


def arrow(d, p, q, color=BLACK, width=3, label="", lsize=18, loff=(0, -16)):
    import math
    d.line([p, q], fill=color, width=width)
    a = math.atan2(q[1] - p[1], q[0] - p[0])
    L = 14
    d.polygon([q, (q[0] - L * math.cos(a - 0.4), q[1] - L * math.sin(a - 0.4)), (q[0] - L * math.cos(a + 0.4), q[1] - L * math.sin(a + 0.4))], fill=color)
    if label:
        text_c(d, ((p[0] + q[0]) / 2 + loff[0], (p[1] + q[1]) / 2 + loff[1]), label, lsize, color, bold=False)


def concepts():
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    text_c(d, (W / 2, 60), "コンセプト", 40)
    rows = [
        ("箱は、見た目だけでは分からない", "強度（潰れやすさ）・重さ・摩擦（滑りやすさ）が、箱ごとに違う"),
        ("1 台目は、触って確かめる", "ゆるい力から少しずつ試し、持ち上がる最小の力・重さ・強度を測る"),
        ("2 台目は、情報で素早く運ぶ", "受け取った情報で、試さずに 1 回で持ち、重さに合わせた速さで運ぶ"),
        ("カメラ（Gemini）で、試す回数を減らす", "見た目から壊れにくさを予想し、試し方を変える（強度は必ず触って測る）"),
    ]
    y = 150
    for i, (head, body) in enumerate(rows):
        d.rectangle([100, y, 1180, y + 110], outline=BLACK, width=2)
        text_l(d, (130, y + 18), f"{i + 1}", 44, BLUE, bold=True)
        text_l(d, (200, y + 18), head, 30, BLACK, bold=True)
        text_l(d, (200, y + 64), body, 22, GRAY)
        y += 130
    im.save(OUT / "fig_concepts.png")


def camera(seed=22, n=6):
    """左：Gemini に見せた写真。右：Gemini の答えと、本当の値"""
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    text_c(d, (W / 2, 45), "カメラ（Gemini）の判定：見た目から「壊れにくいか」を予想する", 30)
    photo = Image.open(ROOT / "handover" / "data" / f"softboxes_seed{seed}_n{n}.png").convert("RGB")
    photo = photo.resize((560, int(560 * photo.height / photo.width)))
    im.paste(photo, (40, 110))
    d.rectangle([40, 110, 40 + photo.width, 110 + photo.height], outline=GRAY, width=1)
    txt = (ROOT / "handover" / "data" / f"softboxes_seed{seed}_n{n}.gemini.txt").read_text()
    ans = sorted(json.loads(txt.split("```json")[-1].split("```")[0] if "```" in txt else txt), key=lambda a: a["point"][1])
    import sys
    sys.path.insert(0, str(ROOT / "handover"))
    import softbox as sb
    specs = sb.random_boxes(n, seed=seed)
    x0, y = 630, 120
    text_l(d, (x0, y), "左から   Gemini の答え                         本当", 20, GRAY, bold=True)
    y += 40
    for i, (a, sp) in enumerate(zip(ans, specs)):
        sturdy = not a.get("fragile", True)
        col = BLUE if sturdy else BLACK
        text_l(d, (x0, y), f"{i + 1}", 22, BLACK, bold=True)
        text_l(d, (x0 + 50, y), f"{a['looks_like']}・{'壊れにくい' if sturdy else '壊れやすい'}・{a['weight_g'][0]}〜{a['weight_g'][1]} g", 20, col, bold=sturdy)
        text_l(d, (x0 + 420, y), f"{int(sp['mass'] * 1000)} g・強度 {sp['strength']:.0f} N", 20, GRAY)
        y += 42
    text_l(d, (60, 470), "・材質は 6 つとも当たる。重さは当たらない（写真に中身は写らない）", 22)
    text_l(d, (60, 510), "・「壊れにくい」と答えた箱は、大きく刻んで試す（τ 0.25 から 0.05 ずつ）", 22, BLUE)
    text_l(d, (60, 550), "・それ以外は、小さく刻む（τ 0.2 から 0.025 ずつ）", 22)
    text_l(d, (60, 590), "・見た目どおりでない箱（1：金属に見えて強度 16 N）も、強度は触って測るので潰さない", 22)
    im.save(OUT / "fig_camera.png")


def summary(times=None):
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    text_c(d, (W / 2, 50), "まとめ：見て予想し、触って確かめ、情報で素早く運ぶ", 32)
    box(d, 40, 150, 260, 260, "カメラ\nGemini", 26)
    box(d, 330, 130, 640, 280, "1 台目（触る）\n\nゆるい力から少しずつ\n力加減・重さ・強度を測る", 21)
    box(d, 710, 150, 860, 260, "台座", 26)
    box(d, 930, 130, 1240, 280, "2 台目（運ぶ）\n\n情報で 1 回で持ち\n重さに合わせて運ぶ", 21)
    arrow(d, (260, 205), (330, 205), label="試し方", lsize=18)
    arrow(d, (640, 205), (710, 205), label="箱", lsize=18)
    arrow(d, (860, 205), (930, 205))
    d.line([(485, 280), (485, 330), (1085, 330), (1085, 280)], fill=BLUE, width=4)
    text_c(d, (785, 352), "情報：力加減・重さ・強度", 22, BLUE)
    arrow(d, (485, 280), (485, 420), RED, label="", lsize=18)
    box(d, 330, 420, 640, 500, "重さに対して弱すぎる箱\n→ 運ばずに人の確認へ", 20, RED)
    box(d, 930, 380, 1240, 460, "大箱", 24)
    arrow(d, (1085, 330), (1085, 380))
    if times:
        rows = [("1 台目 3 台・カメラなし", times[0]), ("1 台目 2 台・カメラなし", times[1]), ("1 台目 2 台・カメラあり", times[2])]
        y = 560
        text_l(d, (60, y - 40), "同じ 6 箱（強度・重さ・摩擦はランダム）を、最後まで終えるまでの時間", 22, GRAY, bold=True)
        for label, (sec, tries) in rows:
            text_l(d, (80, y), f"{label}", 24, BLUE if "あり" in label else BLACK, bold="あり" in label)
            text_l(d, (480, y), f"{sec:.1f} 秒　試した回数 {tries} 回", 24, BLUE if "あり" in label else BLACK, bold="あり" in label)
            y += 40
    im.save(OUT / "fig_summary.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    concepts()
    camera()
    summary(times=[(86.3, 26), (92.2, 21), (83.5, 17)])
    print("wrote", OUT)
