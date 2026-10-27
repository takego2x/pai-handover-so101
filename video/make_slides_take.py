"""提出動画の図の画面を、モノクロのスライド様式の PowerPoint で作る（表紙以外）。

デザインは 作者の別の講義資料 の寸法をそのまま流用：
  見出し 20pt 太字＋1.5pt の線／リード帯 F2F2F2・■ 2 行・14pt 太字／図表タイトル 13pt 太字＋1.0pt 下線／
  枠 0.75pt 1A1A1A（本文 12.5pt）／番号 0.66 cm 角の黒地白抜き／結論の箱 F2F2F2・0.75pt・13.5pt 太字／
  下の線 E8E8E8 0.5pt とページ番号の黒いチップ。書体 Meiryo UI、グレーだけ（有彩色なし）。

  <python-pptx の入った python> video/make_slides_take.py
  → outputs/video2/slides_take.pptx（画像への書き出しは PowerPoint で行う）
"""
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "video2"
sys.path.insert(0, str(ROOT / "handover"))
import softbox as sb  # noqa: E402


def run_of(name):
    """録画の記録（outputs/video2/<name>.json）から、時間 [s]・試した回数の合計・箱ごとの値。数字を手で写さない"""
    d = json.loads((OUT / f"{name}.json").read_text())
    return d["steps"] / 100, sum(b["tries"] or 0 for b in d["boxes"]), d


def r0(x):
    return int(x + 0.5)


SEED = 278                   # p6・p7 と比べる構成の箱の組（運べない箱が出ない組。p3・p5 は別の組）
# 1 台目 2 台・カメラなしは録画していない（outputs/debug/f2_s278_pick3.log、2026-10-27）
F2_NOCAM = (107.0, 11)
# (1 台目の台数, カメラ, (全部の箱の判断が終わるまでの時間 [s], 試した回数), ロボットの台数＝1 台目＋2 台目)
RESULTS = [("3 台", "なし", run_of("p6")[:2], 4), ("2 台", "なし", F2_NOCAM, 3), ("2 台", "あり", run_of("p7")[:2], 3)]


def robot_sec(t, n):
    """ロボットの総稼働時間 [台・秒]＝台数 × 全部の箱の判断が終わるまでの時間"""
    return t * n

FONT = "Meiryo UI"
INK, DARK, MID, LIGHT, BAND, RULE = (RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0x40, 0x40, 0x40), RGBColor(0xBF, 0xBF, 0xBF),
                                     RGBColor(0xF2, 0xF2, 0xF2), RGBColor(0xF2, 0xF2, 0xF2), RGBColor(0xE8, 0xE8, 0xE8))
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
X0, W = 1.27, 31.32                      # 左端と全幅 [cm]


def _font(run, size, bold=False, color=INK):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:latin"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", FONT)


def text(s, x, y, w, h, lines, size=12.5, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Cm(0.25)
    for i, ln in enumerate([lines] if isinstance(lines, str) else lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = ln
        _font(r, size, bold, color)
    return tb


def no_shadow(sp):
    """影なし。python-pptx の図形には既定のスタイル（効果の参照）が付き、LibreOffice や PowerPoint で影が出るので、スタイルごと外す"""
    st = sp._element.find(qn("p:style"))
    if st is not None:
        sp._element.remove(st)


def rect(s, x, y, w, h, fill=WHITE, line=INK, lw=0.75, dash=None, lines=None, size=12.5, bold=False, color=INK,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, shape=MSO_SHAPE.RECTANGLE):
    sp = s.shapes.add_shape(shape, Cm(x), Cm(y), Cm(w), Cm(h))
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(lw)
        if dash:
            ln = sp.line._get_or_add_ln()
            ln.append(ln.makeelement(qn("a:prstDash"), {"val": dash}))
    no_shadow(sp)
    if lines is not None:
        tf = sp.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Cm(0.4)
        for i, ln in enumerate([lines] if isinstance(lines, str) else lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            r = p.add_run()
            r.text = ln
            _font(r, size, bold, color)
    return sp


def hline(s, x, y, w, color=INK, lw=1.0, dash=None):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x), Cm(y), Cm(x + w), Cm(y))
    no_shadow(ln)
    ln.line.color.rgb = color
    ln.line.width = Pt(lw)
    if dash:
        e = ln.line._get_or_add_ln()
        e.append(e.makeelement(qn("a:prstDash"), {"val": dash}))
    return ln


def arrow(s, x1, y1, x2, y2, lw=1.75, color=INK):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    no_shadow(ln)
    ln.line.color.rgb = color
    ln.line.width = Pt(lw)
    e = ln.line._get_or_add_ln()
    e.append(e.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return ln


def chrome(s, heading, lead, page):
    """見出し・線・リード帯・下の線・ページ番号（見本 p3 と同じ寸法）"""
    text(s, X0, 0.81, W, 1.78, heading, size=20, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    hline(s, X0, 2.79, W, INK, 1.5)
    rect(s, X0, 3.2, W, 2.08, fill=BAND, line=None, lines=[f"■ {ln}" for ln in lead], size=14, bold=True)
    hline(s, X0, 17.91, W, RULE, 0.5)
    rect(s, 30.48, 18.03, 2.11, 0.76, fill=INK, line=None, lines=str(page), size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


def fig_title(s, x, y, w, t):
    text(s, x, y, w, 0.76, t, size=13, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.BOTTOM)
    hline(s, x, y + 0.76, w, INK, 1.0)


def conclusion(s, t, y=15.8):
    rect(s, X0, y, W, 1.32, fill=LIGHT, line=INK, lw=0.75, lines=t, size=13.5, bold=True, align=PP_ALIGN.CENTER)


def num(s, x, y, n, size=0.66):
    rect(s, x, y, size, size, fill=INK, line=None, lines=str(n), size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


def labeled_frame(s, x, y, w, h, label, lines, kind="fact", size=12.5):
    """枠＋上辺に重ねた白地のラベル（見本 p7 の「例え／原因／対策」）。kind：fact 細実線、issue 長破線、new 太枠"""
    lw, dash = {"fact": (0.75, None), "issue": (0.75, "lgDash"), "new": (2.75, None)}[kind]
    rect(s, x, y, w, h, fill=WHITE, line=INK, lw=lw, dash=dash)
    lab_w = 1.2 + 0.52 * len(label)          # 12.5pt の全角 1 字 ≒ 0.45 cm。左右の余白 0.4 cm ずつを足す
    lab = rect(s, x + 0.6, y - 0.38, lab_w, 0.76, fill=WHITE, line=None, lines=label, size=12.5, bold=True, align=PP_ALIGN.CENTER)
    lab.text_frame.word_wrap = False
    text(s, x + 0.45, y + 0.5, w - 0.9, h - 0.6, lines, size=size)


def tag(s, x, y, t):
    w = 0.9 + 0.5 * len(t)
    rect(s, x, y, w, 0.76, fill=INK, line=None, lines=t, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE)


# ---------------------------------------------------------------- ページ
def page_real(prs):
    """現実の場面（物流の返品処理）と、この作品での模擬の対応。右の絵は video/still_twin_boxes.py で描いた seed 22 の段ボール 2 つ"""
    from PIL import Image
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(s, "2 現実の場面｜物流の処理",
           ["物流の処理では、あらかじめ重さや滑りやすさ、壊れやすさを把握して処理しないといけない",
            "ロボットではこの情報が不足している"], 2)
    # 左：現実（生成 AI で作ったイメージ。作者提供、2026-10-27）。切らずに大きく置き、3 つの物に「？」を重ねる
    # 写真だけを大きく（模擬の絵・対応表・結論の箱は置かない。2026-10-27 作者）
    photo = Image.open(OUT / "real_returns.png")
    ph_, py_ = 10.2, 6.6                                     # 写真の高さと上端（上に図表タイトル、下に注記）
    pw_ = ph_ * photo.width / photo.height
    px_ = X0 + (W - pw_) / 2
    fig_title(s, px_, 5.6, pw_, "重い？滑る？潰れる？")
    s.shapes.add_picture(str(OUT / "real_returns.png"), Cm(px_), Cm(py_), height=Cm(ph_))
    k_ = pw_ / photo.width                                    # 画素 → cm
    for u, v in [(470, 300), (915, 530), (1380, 600)]:        # 手に持った箱・金属のケース・ラップを巻いた箱
        rect(s, px_ + u * k_ - 1.4, py_ + v * k_ - 0.65, 2.8, 1.3, fill=INK, line=WHITE, lw=1.5, lines="？kg", size=22,
             bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    text(s, px_, 16.8, pw_, 0.6, "*写真は生成 AI で作成したイメージ", size=9, color=DARK)


def page_concepts(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(s, "3 コンセプト",
           ["箱の強度・重さ・摩擦は、見た目だけでは分からない（箱ごとにランダム）",
            "①触って確かめる 1 台目、②情報で運ぶ 2 台目、③カメラで試す回数を減らせるのでは？"], 3)
    rows = [
        ("1 台目は触って確かめる", "ゆるい力から少しずつ試す／持ち上がる最小の力・重さ・強度を測る", "①"),
        ("2 台目は情報で素早く運ぶ", "受け取った情報で 1 回で持つ／重さに合わせた速さで運ぶ", "②"),
        ("カメラで試す回数を減らす", "Gemini が見た目から壊れにくさを予想／強度は必ず触って測る", "③"),
        ("潰さず、落とさない", "握る力は推定強度の 6 割まで／持ち上がった力の 3 倍の余裕で握って運ぶ", "④"),
    ]
    y = 6.0
    fig_title(s, X0, y - 0.1, W, "作品の考え方")
    y += 1.2
    for i, (head, body, _) in enumerate(rows):
        num(s, X0 + 0.2, y + 0.42, i + 1, size=0.9)
        text(s, X0 + 1.5, y, 10.5, 1.75, head, size=14, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, X0 + 12.2, y, W - 12.4, 1.75, body, size=12.5, anchor=MSO_ANCHOR.MIDDLE)
        hline(s, X0, y + 1.85, W, MID, 0.75)
        y += 1.95
    conclusion(s, "見て予想し、触って確かめ、情報で素早く運ぶ", y=15.8)


def top_numbered(seed=SEED, first=2, n=6):
    """固定カメラ（真上）の画像を、箱のある所だけ切り抜き、箱に番号（1〜6）を振る。
    箱の位置は soft_line.py の置き方と同じ式で出し、カメラの式（vision._points の逆）で画素にする"""
    import math
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(ROOT / "handover" / "data" / f"top_first{first}_n{n}_seed{seed}.png").convert("RGB")
    D, R, BOX_AZ, stations = 0.44, 0.22, [40, 64, 88], {1: [0], 2: [0, -100], 3: [0, 100, -100]}[first]
    side = {0: 1, 100: -1, -100: 1}
    cx, cy, H, fov, res = 0.05, 0.0, 1.8, 40, 1280           # soft_line.py の add_top_camera と同じ
    f = (res / 2) / math.tan(math.radians(fov) / 2)
    z = H - 0.03                                              # 箱の上の面までの距離
    pts = []
    for j in range(n):
        a = stations[j % len(stations)]
        yaw = math.radians(a + 180)
        w = yaw + side[a] * math.radians(BOX_AZ[j // len(stations)])
        bx, by = D * math.cos(math.radians(a)) + R * math.cos(w), D * math.sin(math.radians(a)) + R * math.sin(w)
        pts.append(((bx - cx) * f / z + (res - 1) / 2, -(by - cy) * f / z + (res - 1) / 2))
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    pad = 90
    box = (int(min(us)) - pad, int(min(vs)) - pad, int(max(us)) + pad, int(max(vs)) + pad)
    crop = img.crop(box)
    dr = ImageDraw.Draw(crop)
    font = ImageFont.truetype(str(Path.home() / "Library" / "Application Support" / "LibreOffice" / "4" / "user" / "fonts" / "meiryob.ttc"), 24)
    for i, (u, v) in enumerate(pts):
        x, y = u - box[0] + 26, v - box[1] - 26              # 箱の右上に、黒地に白の番号
        dr.rectangle([x - 15, y - 15, x + 15, y + 15], fill=(26, 26, 26))
        dr.text((x, y), str(i + 1), fill=(255, 255, 255), font=font, anchor="mm")
    out = OUT / "top_numbered.png"
    crop.save(out)
    return out, crop.size


def page_camera(prs, seed=SEED, n=6):
    import re
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(s, "11 カメラ判定で早くする｜Gemini の見た目の予想",
           ["固定カメラ（真上）の画像を Gemini に見せ、箱ごとに材質・壊れやすさ・重さを答えさせる",
            "「壊れにくい」と見た箱だけ大きく刻んで試せば、試す回数を減らせるのでは？"], 11)
    fig_title(s, X0, 5.7, 13.2, "Gemini に見せた画像（固定カメラ、一部を拡大）")
    pic, (pw, ph) = top_numbered(seed, 2, n)
    wcm = min(11.0, 8.4 * pw / ph)                              # 高さ 8.4 cm まで
    s.shapes.add_picture(str(pic), Cm(X0 + (13.2 - wcm) / 2), Cm(6.75), width=Cm(wcm))
    # 箱ごとの Gemini の答え：p7（1 台目 2 台・カメラあり）の記録の「試し始め」に書いてある
    marks = run_of("p7")[2]["marks"]
    said = {m["box"]: re.search(r"カメラ：(.+?)・(壊れ\S+?)・\[(\d+), (\d+)\] g", m["what"]) for m in marks if "試し始め" in m["what"]}
    ans = [{"looks_like": said[f"box{i}"].group(1), "fragile": said[f"box{i}"].group(2) != "壊れにくい",
            "weight_g": [said[f"box{i}"].group(3), said[f"box{i}"].group(4)]} for i in range(n)]
    specs = sb.random_boxes(n, seed=seed)
    x1, w1 = 15.6, 17.0
    fig_title(s, x1, 5.7, w1, "Gemini の答えと、本当の値")
    cols = [("", 0.9), ("Gemini の答え", 9.0), ("本当", 7.1)]
    y = 6.75
    xx = x1
    for name, cw in cols:
        rect(s, xx, y, cw, 0.8, fill=INK, line=None, lines=name, size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        xx += cw
    y += 0.8
    for i, (a, sp) in enumerate(zip(ans, specs)):
        sturdy = not a.get("fragile", True)
        xx = x1
        num(s, xx + 0.12, y + 0.12, i + 1)
        xx += cols[0][1]
        text(s, xx, y, cols[1][1], 0.9, f"{a['looks_like']}／{'壊れにくい' if sturdy else '壊れやすい'}／{a['weight_g'][0]}〜{a['weight_g'][1]} g",
             size=11.5, bold=sturdy, anchor=MSO_ANCHOR.MIDDLE)
        xx += cols[1][1]
        text(s, xx, y, cols[2][1], 0.9, f"{int(sp['mass'] * 1000)} g／強度 {sp['strength']:.0f} N", size=11.5, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
        hline(s, x1, y + 0.9, w1, MID, 0.5)
        y += 0.9
    # 見た目どおりでない箱：「壊れにくい」と答えたのに強度が弱め（25 N 未満）、または「壊れやすい」と答えたのに強い（40 N 以上）
    odd = [(i + 1, a, sp) for i, (a, sp) in enumerate(zip(ans, specs))
           if (not a.get("fragile", True) and sp["strength"] < 25) or (a.get("fragile", True) and sp["strength"] >= 40)]
    note = ("┗ 見た目どおりでない箱（" + "、".join(f"{i}：{'壊れにくく' if not a.get('fragile', True) else '壊れやすく'}見えて強度 {sp['strength']:.0f} N"
                                               for i, a, sp in odd) + "）も、強度は触って測るので潰さない") if odd else \
           "┗ 見た目の予想は試し始めの力にだけ使い、強度は必ず触って測る"
    labeled_frame(s, X0, 13.45, W, 1.95, "試し方",
                  ["①「壊れにくい」と答えた箱：τ 0.25 から 0.05 ずつ／それ以外とカメラなし：τ 0.2 から 0.025 ずつ", note], kind="fact", size=12)
    (_, _, (t3, k3), _), (_, _, (t2, k2), _), (_, _, (tc, kc), _) = RESULTS
    conclusion(s, f"1 台目 2 台で、試す回数 {k2} → {kc} 回、時間 {t2:.1f} → {tc:.1f} 秒（金属に見える箱を 1 回で決める）", y=15.8)


def page_summary(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(s, "12 まとめ｜コンセプト図",
           ["見た目で予想 ⇒ 触って確かめる ⇒ 情報を渡す ⇒ 素早く運ぶ",
            "2 台目は試さずに 1 回で持ち、止まらずに運ぶ"], 12)
    fig_title(s, X0, 5.7, W, "見て予想し、触って確かめ、情報で素早く運ぶ")
    y, h = 7.2, 3.2
    steps = [("カメラ", ["箱の位置を測る", "Gemini が予想"], "fact", 4.6),
             ("1 台目", ["触って確かめる", "力加減・重さ・強度"], "new", 7.4),
             ("台座", ["箱と情報を", "渡す"], "fact", 4.6),
             ("2 台目", ["情報で 1 回で持つ", "重さに合わせて運ぶ"], "new", 7.4),
             ("大箱", ["入れる"], "fact", 3.6)]
    gap = (W - sum(st[3] for st in steps)) / (len(steps) - 1)
    x = X0
    centers = []
    for i, (label, body, kind, w) in enumerate(steps):
        lw = 2.75 if kind == "new" else 0.75
        rect(s, x, y, w, h, fill=WHITE, line=INK, lw=lw)
        rect(s, x, y, w, 0.85, fill=INK, line=None, lines=label, size=12.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        text(s, x, y + 1.0, w, h - 1.1, body, size=12.5, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        centers.append((x, w))
        if i < len(steps) - 1:
            arrow(s, x + w + 0.1, y + h / 2, x + w + gap - 0.1, y + h / 2)
        x += w + gap
    # 情報の流れ（1 台目 → 2 台目）
    xa = centers[1][0] + centers[1][1] / 2
    xb = centers[3][0] + centers[3][1] / 2
    yi = y + h + 0.9
    hline(s, xa, yi, xb - xa, INK, 2.25)
    v = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(xa), Cm(y + h), Cm(xa), Cm(yi))
    no_shadow(v)
    v.line.width = Pt(2.25)
    v.line.color.rgb = INK
    arrow(s, xb, yi, xb, y + h + 0.05, lw=2.25)
    tag(s, (xa + xb) / 2 - 3.6, yi - 0.38, "情報：力加減・重さ・強度")
    # 結果（右下）
    rows = [(f"1 台目 {n}・カメラ{c}", f"{t:.1f} 秒 × {m} 台 ＝ {r0(robot_sec(t, m))} 台・秒") for n, c, (t, k), m in RESULTS]
    xr, wr, yr = 15.6, 17.0, 12.0
    text(s, xr, yr - 0.1, wr, 0.6, "同じ 6 箱を終えるまでの、ロボットの総稼働時間", size=10.5, bold=True, color=DARK)
    yr += 0.6
    for i, (k, v) in enumerate(rows):
        last = i == len(rows) - 1
        text(s, xr, yr, 6.8, 0.75, k, size=11.5, bold=last, anchor=MSO_ANCHOR.MIDDLE)
        text(s, xr + 6.8, yr, wr - 6.8, 0.75, v, size=11.5, bold=last, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.RIGHT)
        hline(s, xr, yr + 0.75, wr, MID, 0.5)
        yr += 0.78
    (t3, _), m3 = RESULTS[0][2], RESULTS[0][3]
    (tc, _), mc = RESULTS[2][2], RESULTS[2][3]
    cut = r0((1 - robot_sec(tc, mc) / robot_sec(t3, m3)) * 100)
    conclusion(s, f"触って確かめるから潰さない。ロボットを 4 台から 3 台に減らし、総稼働時間を {cut}% 少なく", y=15.8)


# ---------------------------------------------------------------- 映像を入れるページ
PX_PER_CM = 1280 / 33.867
SLOTS = []      # (スライドの番号 0 始まり, [(名前, x, y, w, h [px]), ...])


def video_slot(s, name, x, y, w):
    """映像を置く場所（16:9）。枠だけ描き、中は空ける。後で映像をはめ込む"""
    h = w * 9 / 16
    rect(s, x, y, w, h, fill=RGBColor(0x26, 0x2B, 0x33), line=INK, lw=0.75)
    return (name, round(x * PX_PER_CM), round(y * PX_PER_CM), round(w * PX_PER_CM), round(h * PX_PER_CM))


def add_video_slide(prs, slots):
    SLOTS.append((len(prs.slides._sldIdLst) - 1, slots))


def table(s, x, y, widths, header, rows, hl=None, row_h=0.95, size=11.5):
    """黒地に白抜きの見出し行＋細い罫の行。hl の行は番号を黒地にし、文字を太字にする（いま映っている箱）"""
    xx = x
    for name, cw in zip(header, widths):
        rect(s, xx, y, cw, 0.8, fill=INK, line=None, lines=name, size=10.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        xx += cw
    y += 0.8
    for i, row in enumerate(rows):
        on = hl is not None and i == hl
        xx = x
        for j, (cell, cw) in enumerate(zip(row, widths)):
            if j == 0:
                rect(s, xx + (cw - 0.7) / 2, y + (row_h - 0.7) / 2, 0.7, 0.7, fill=INK if on or hl is None else MID, line=None,
                     lines=str(cell), size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
            else:
                text(s, xx, y, cw, row_h, cell, size=size, bold=on, color=INK if (on or hl is None) else DARK,
                     anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
            xx += cw
        hline(s, x, y + row_h, sum(widths), MID, 0.5)
        y += row_h
    return y


VX, VW = X0, 15.5                      # 映像の位置（左）と幅 [cm]。高さ 8.7 cm で、結論の箱（15.8 cm）の上に収まる
RX, RW = 18.4, 32.59 - 18.4            # 右の列


def page_success(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(s, "7 実施したい動作｜受け渡しの成功例",
           ["1 台目が箱を取って台座へ、2 台目が台座から取って大箱へ",
            "箱 1 つ（段ボール 62 g）の成功例"], 7)
    fig_title(s, VX, 5.7, VW, "映像：成功例（約 2.1 倍速）")
    slot = video_slot(s, "p3", VX, 6.75, VW)
    labeled_frame(s, RX, 7.1, RW, 7.9, "動作の順", [""], kind="fact")
    steps = ["1 台目：ゆるい力で試し掴み", "1 台目：重さ・強度を測る", "1 台目：台座に置き、情報を渡す",
             "2 台目：情報で 1 回で持つ", "2 台目：大箱へ入れる"]
    y = 8.0
    for i, st in enumerate(steps):
        num(s, RX + 0.6, y + 0.12, i + 1)
        text(s, RX + 1.5, y, RW - 2.0, 0.9, st, size=12.5, anchor=MSO_ANCHOR.MIDDLE)
        y += 1.3
    conclusion(s, "箱ごとに強度・重さ・摩擦が違うと、同じ動作では失敗する（次頁）", y=15.8)
    add_video_slide(prs, [slot])


def page_issues(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(s, "8 実施に向けた課題｜強度・重さ・摩擦",
           ["①強度：潰してしまう／②重さ：ゆるく掴むと落とす／③摩擦：同じ力でも滑る",
            "どれも、見た目だけでは分からない"], 8)
    cw, gap = 10.0, (W - 3 * 10.0) / 2
    cols = [("① 強度：潰す", "p4a", ["力いっぱい握る", "⇒ 挟む力 100 N、強度 7.5 N の箱が潰れる"]),
            ("② 重さ：落とす", "p4b", ["ぎりぎりの力で 250 g を持つ", "⇒ 速く運ぶと落ちる"]),
            ("③ 摩擦：滑る", "p4c", ["同じ力・同じ重さ（250 g）", "⇒ 摩擦 0.4（右）は滑る、1.0（左）は持てる"])]
    slots = []
    for i, (title, name, body) in enumerate(cols):
        x = X0 + i * (cw + gap)
        fig_title(s, x, 5.7, cw, title)
        slots.append(video_slot(s, name, x, 6.75, cw))
        rect(s, x, 12.6, cw, 2.6, fill=WHITE, line=INK, lw=0.75, dash="lgDash", lines=body, size=11.5)
    conclusion(s, "強めても弱めても失敗する ⇒ 箱ごとに、ちょうどよい力を知る必要がある", y=15.8)
    add_video_slide(prs, slots)


P5_ROWS = [(str(i + 1), f"{b['tries']} 回", f"{r0(b['m_hat'] * 1000)} g（{r0(b['mass'] * 1000)}）", f"{r0(b['strength_est'])} N（{r0(b['strength'])}）")
           for i, b in enumerate(run_of("p5")[2]["boxes"])]


def page_solution(prs, hl):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(s, "9 解決に向けた対応｜1 台目が少しずつ試して把握する",
           ["1 台目がゆるい力から少しずつ試し、持ち上がる力・重さ・強度を測る",
            "2 台目は受け取った情報で、試さずに 1 回で持つ（重さ・強度・摩擦はランダム）"], 9)
    fig_title(s, VX, 5.7, VW, "映像：ランダムな箱 3 つ（約 2.9 倍速）")
    slot = video_slot(s, "p5", VX, 6.75, VW)
    fig_title(s, RX, 5.7, RW, "1 台目が測った値（カッコ内は本当の値）")
    y = table(s, RX, 6.75, [1.6, 2.8, 5.0, 4.79], ["箱", "試した回数", "重さ", "強度"], P5_ROWS, hl=hl)
    labeled_frame(s, RX, y + 0.9, RW, 15.4 - (y + 0.9), "強度の測り方",
                  ["箱を少し持ち上げ、握った指の角度 ⇒ 硬さ ⇒ 強度", "┗ 握る力は推定強度の 6 割まで（推定の誤差に備える）"], kind="fact", size=11.5)
    conclusion(s, "触って確かめるので潰さない。重さと強度を情報として 2 台目へ渡す", y=15.8)
    add_video_slide(prs, [slot])


def page_apply(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(s, "10 解決後の動作｜1 台目 3 台で情報を集める",
           ["1 台目 3 台が触って情報を集め、台座へ置く",
            "2 台目は情報で次々と大箱へ運ぶ（6 箱、重さ・強度・摩擦はランダム）"], 10)
    fig_title(s, VX, 5.7, VW, "映像：1 台目 3 台＋2 台目（約 4 倍速）")
    slot = video_slot(s, "p6", VX, 6.75, VW)
    sec, tries, d = run_of("p6")
    fig_title(s, RX, 5.7, RW, f"6 箱を {sec:.1f} 秒で大箱へ（試した回数の合計 {tries} 回）")
    rows = [(str(i + 1), f"{b['tries']} 回", f"{r0(b['m_hat'] * 1000)} g（{r0(b['mass'] * 1000)}）", f"{r0(b['strength_est'])} N（{r0(b['strength'])}）")
            for i, b in enumerate(d["boxes"])]
    y = table(s, RX, 6.75, [1.6, 2.8, 5.0, 4.79], ["箱", "試した回数", "重さ", "強度"], rows, row_h=0.9)
    text(s, RX, y + 0.2, RW, 0.6, "*カッコ内は本当の値", size=9.5, color=DARK)
    conclusion(s, "情報を集める台数を増やせば、2 台目は止まらずに運べる", y=15.8)
    add_video_slide(prs, [slot])


def page_camera_run(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(s, "11 カメラ判定で早くする｜1 台目を 2 台に",
           ["Gemini が「壊れにくい」と見た箱だけ、大きく刻んで試す",
            "ロボットの総稼働時間（台数 × 時間）で比べる：1 台目を 1 台減らしても、" + (
                f"3 台より {r0((1 - RESULTS[2][2][0] / RESULTS[0][2][0]) * 100)}% 早く終わる" if RESULTS[2][2][0] < RESULTS[0][2][0]
                else f"時間は {r0((RESULTS[2][2][0] / RESULTS[0][2][0] - 1) * 100)}% しか延びない")], 11)
    fig_title(s, VX, 5.7, VW, "映像：1 台目 2 台＋カメラ（約 4 倍速）")
    slot = video_slot(s, "p7", VX, 6.75, VW)
    fig_title(s, RX, 5.7, RW, "同じ 6 箱の判断を終えるまで")
    rows = [(str(i + 1), n, c, f"{t:.1f} 秒", f"{k} 回", f"{r0(robot_sec(t, m))} 台・秒") for i, (n, c, (t, k), m) in enumerate(RESULTS)]
    y = table(s, RX, 6.75, [1.0, 2.0, 1.9, 2.9, 2.5, 3.89], ["", "1 台目", "カメラ", "時間", "試した回数", "総稼働時間"], rows, hl=2)
    text(s, RX, y + 0.2, RW, 2.0, ["*試した回数＝1 台目が 6 箱を掴んで試した回数の合計",
                                   "*総稼働時間＝ロボットの台数（1 台目＋2 台目）× 時間",
                                   "*1 組の箱（6 箱）を 1 回ずつ流した結果。6 箱とも大箱へ",
                                   "*同じ箱でも、掴む台や位置で持ち上がる力は変わる"], size=9.5, color=DARK)
    (t3, _), m3 = RESULTS[0][2], RESULTS[0][3]
    (tc, _), mc = RESULTS[2][2], RESULTS[2][3]
    conclusion(s, f"ロボット 4 台 → 3 台で、総稼働時間 {r0(robot_sec(t3, m3))} → {r0(robot_sec(tc, mc))} 台・秒"
                  f"（{r0((1 - robot_sec(tc, mc) / robot_sec(t3, m3)) * 100)}% 少ない）", y=15.8)
    add_video_slide(prs, [slot])


def page_env(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(s, "4 開発環境｜活用した OSS・オープンなロボット・AI",
           ["シミュレータ・ロボットの形・計算の部品は OSS、目の AI は Gemini（API）",
            "GPU も実機も使わず、Mac の CPU（Docker）で作品の全部を動かす"], 4)
    fig_title(s, X0, 5.7, W, "使ったもの（版・ライセンスは手元の実行環境で確認）")
    rows = [("1", "物理シミュレータ", "Genesis 1.4.3", "剛体物理（接触・摩擦・関節）、IK、カメラの描画（真上：RGB・深度、手首：RGB）", "Apache-2.0"),
            ("2", "ロボットモデル", "SO-101（SO-ARM100）", "MJCF。オープンハードウェアの 6 軸アーム＋グリッパ", "Apache-2.0"),
            ("3", "凸分解", "CoACD 1.0.14", "衝突形状の生成（グリッパの隙間を保持）", "MIT"),
            ("4", "数値計算・画像処理", "PyTorch 2.14／NumPy 2.5／SciPy 1.18", "推定・校正の計算、深度画像から箱の位置・向き（ラベリング）", "BSD-3"),
            ("5", "VLM", "Gemini Robotics-ER 2", "画像 → 箱の位置（点）・材質・重量・脆さ（JSON）、google-genai 2.29", "API（OSS 外）"),
            ("6", "動画処理・資料生成", "FFmpeg／python-pptx", "動画の切り出し・合成、スライド生成（LibreOffice でラスタ化）", "LGPL／MIT"),
            ]
    table(s, X0, 6.75, [1.4, 4.6, 7.4, 13.2, 4.72], ["", "区分", "名前（版）", "使い方", "ライセンス"], rows, row_h=1.0, size=11.5)
    text(s, X0, 13.95, W, 1.2, ["*実行環境：Apple シリコンの Mac、Docker（Python 3.12、CPU のみ）。教材は Google Colab 無料版（CPU）",
                                 "*SO-101 の形は TheRobotStudio/SO-ARM100（a758567）を使用"], size=9.5, color=DARK)
    conclusion(s, "OSS とオープンなロボットの形で、手元の CPU だけで再現できる", y=15.8)


def page_system(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(s, "5 システム構成｜見る・触る・渡す・運ぶ",
           ["読むのは、カメラ 2 つ（固定・手首）と、自分の関節の角度とトルクだけ（力覚センサーなし）",
            "①見て予想、②触って推定、③台座で渡す、④情報で運ぶ、の 4 つに分ける"], 5)
    fig_title(s, X0, 5.7, W, "部品とデータの流れ")
    # 上段：4 つの役割
    y, h = 7.1, 4.2
    cols = [("① 見る", ["固定カメラ（真上）", "深度 ⇒ 箱の位置・向き", "画像 ⇒ Gemini ⇒ 壊れにくいか"], "fact", 7.4),
            ("② 触る：1 台目", ["試し掴み（ゆるい力から）", "指の角度・肩の力、手首カメラ", "⇒ 力加減・重さ・強度"], "new", 8.0),
            ("③ 渡す：台座", ["箱と情報を置く", "使用中の札で", "2 台の腕を譲り合う"], "fact", 5.6),
            ("④ 運ぶ：2 台目", ["固定カメラで箱を見て持つ", "握る力：推定強度の 6 割まで", "重さに合わせた速さ"], "new", 8.4)]
    gap = (W - sum(c[3] for c in cols)) / (len(cols) - 1)
    x = X0
    for i, (label, body, kind, w) in enumerate(cols):
        rect(s, x, y, w, h, fill=WHITE, line=INK, lw=2.75 if kind == "new" else 0.75)
        rect(s, x, y, w, 0.85, fill=INK, line=None, lines=label, size=12.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        text(s, x, y + 1.05, w, h - 1.2, body, size=12, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        if i < len(cols) - 1:
            arrow(s, x + w + 0.08, y + h / 2, x + w + gap - 0.08, y + h / 2)
        x += w + gap
    # 下段：土台（シミュレータと実行環境）
    yb = 12.0
    rect(s, X0, yb, 5.0, 2.4, fill=INK, line=None, lines=["実行基盤"], size=12.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    rect(s, X0 + 5.0, yb, W - 5.0, 2.4, fill=WHITE, line=INK, lw=0.75,
         lines=["Genesis（剛体物理・IK・レンダリング）／SO-101 × 2〜4 台／潰れる箱（本体＋ばね関節の板）",
                "Docker・Python 3.12・CPU のみ／各アームの手順を 1 step ずつ進め、4 台を同時に制御"], size=12)
    text(s, X0, 14.55, W, 1.1, ["*固定カメラ：高さ 1.8 m の支柱から真下向き（RGB・深度）。手首カメラ：SO-101 公式の取り付け位置の 32 mm 角 UVC（RGB のみ）",
                                 "*「持てた」の判断：持ち上げる前後で手首カメラの指のあいだの見た目が変わらず、肩の力が箱の重さの分だけ増えたとき"], size=9.5, color=DARK)
    conclusion(s, "見て予想し、触って確かめ、情報で素早く運ぶ（力覚センサーを使わず、関節の値で推定）", y=15.8)


def frame_label(s, x, y, t, size=11):
    """枠の左上に重ねる、黒地に白抜きの名札"""
    w = 0.7 + sum(0.21 if ord(c) < 128 else 0.4 for c in t) * size / 11
    lab = rect(s, x + 0.3, y - 0.36, w, 0.72, fill=INK, line=None, lines=t, size=size, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    lab.text_frame.word_wrap = False


def page_arch(prs):
    """ソフトウェアの配置（どこで動き、何が流れるか）。自作のコードは太い破線、借りた部品は細い実線"""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(s, "6 システム構成｜ソフトウェアの配置",
           ["Mac の Docker コンテナの中で、Genesis（OSS）の上に自作のコードを載せる。GPU は使わない",
            "外に出るのは Gemini への画像 1 枚だけ（答えは保存し、2 回目からは読むだけ）"], 6)
    # Mac ⊃ Docker コンテナ
    rect(s, X0, 6.0, 24.2, 9.4, fill=WHITE, line=INK, lw=1.0)
    frame_label(s, X0, 6.0, "Mac（Apple シリコン）")
    rect(s, X0 + 0.45, 7.05, 23.3, 8.0, fill=LIGHT, line=INK, lw=0.75)
    frame_label(s, X0 + 0.45, 7.05, "Docker コンテナ（Python 3.12・CPU のみ・OSMesa で描画）")
    # 左：Genesis（OSS）
    gx, gy, gw, gh = X0 + 0.95, 8.15, 7.2, 6.5
    rect(s, gx, gy, gw, gh, fill=WHITE, line=INK, lw=0.75)
    frame_label(s, gx, gy, "Genesis 1.4.3（OSS）")
    for i, t in enumerate(["剛体物理（接触・摩擦・関節）", "逆運動学（IK）", "カメラ 2 つ（真上・手首）", "SO-101 × 2〜4 台（MJCF）"]):
        rect(s, gx + 0.4, gy + 0.65 + i * 1.42, gw - 0.8, 1.2, fill=WHITE, line=MID, lw=0.75, lines=t, size=11.5, align=PP_ALIGN.CENTER)
    # 右：自作のコード
    cx, cy, cw, ch = X0 + 12.2, 8.15, 11.05, 6.5
    rect(s, cx, cy, cw, ch, fill=WHITE, line=INK, lw=2.25, dash="dash")
    frame_label(s, cx, cy, "自作のコード")
    mine = [("soft_line.py", "手順：1 台目が測り、2 台目が運ぶ"), ("so101.py", "手先の動き（S 字）・握る力の上限"),
            ("vision.py", "カメラ 2 つ ⇒ 箱の位置・向き"), ("softbox.py", "潰れる箱（ばね付きの板）")]
    for i, (name, what) in enumerate(mine):
        y = cy + 0.65 + i * 1.42
        rect(s, cx + 0.4, y, cw - 0.8, 1.2, fill=WHITE, line=MID, lw=0.75)
        text(s, cx + 0.5, y, 3.3, 1.2, name, size=11.5, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, cx + 3.7, y, cw - 4.2, 1.2, what, size=11, anchor=MSO_ANCHOR.MIDDLE)
    # Genesis ⇔ 自作のコード
    ax0, ax1 = gx + gw + 0.1, cx - 0.1
    arrow(s, ax1, 10.0, ax0, 10.0)
    text(s, ax0, 8.7, ax1 - ax0, 1.2, ["関節の目標", "握る力の上限"], size=10, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.BOTTOM)
    arrow(s, ax0, 12.4, ax1, 12.4)
    text(s, ax0, 12.5, ax1 - ax0, 1.8, ["関節の角度・トルク", "画像・深度"], size=10, align=PP_ALIGN.CENTER)
    # 外：Gemini
    mx, mw = X0 + 25.5, W - 25.5
    rect(s, mx, 8.15, mw, 2.4, fill=WHITE, line=INK, lw=0.75, lines=["Gemini", "Robotics-ER 2", "（Google の API）"], size=11.5,
         bold=True, align=PP_ALIGN.CENTER)
    arrow(s, cx + cw + 0.05, 8.9, mx - 0.05, 8.9)
    arrow(s, mx - 0.05, 10.0, cx + cw + 0.05, 10.0)
    text(s, mx, 10.75, mw, 3.6, ["送る：画像（PNG）", "返る：JSON", "（箱の点・材質・壊れやすさ）"], size=10, color=DARK)
    conclusion(s, "物理と描画は OSS（Genesis）、手順と推定は自作。判断に使うのはカメラと関節の値だけ", y=15.8)


if __name__ == "__main__":
    import json as _json
    prs = Presentation()
    prs.slide_width, prs.slide_height = Cm(33.867), Cm(19.05)
    order = []
    page_real(prs); order.append("real")
    page_concepts(prs); order.append("concepts")
    page_env(prs); order.append("env")
    page_system(prs); order.append("system")
    page_arch(prs); order.append("arch")
    page_success(prs); order.append("p3")
    page_issues(prs); order.append("p4")
    for hl in range(3):
        page_solution(prs, hl); order.append(f"p5_{hl}")
    page_apply(prs); order.append("p6")
    page_camera(prs); order.append("camera")
    page_camera_run(prs); order.append("p7")
    page_summary(prs); order.append("summary")
    OUT.mkdir(parents=True, exist_ok=True)
    prs.save(OUT / "slides_take.pptx")
    slots = {order[i]: [dict(zip(("name", "x", "y", "w", "h"), sl)) for sl in sls] for i, sls in SLOTS}
    (OUT / "slides_take_layout.json").write_text(_json.dumps({"order": order, "slots": slots}, ensure_ascii=False, indent=1))
    print("wrote", OUT / "slides_take.pptx", len(order), "slides")
