"""動画の素材（図・グラフ・合成した映像）を outputs/video/ に作る。Mac で動かす（日本語のフォントと ffmpeg を使う）。

  uv run --with pillow==11.3.0 --with matplotlib==3.10.7 --with numpy==2.3.4 python video/make_assets.py

先に撮っておくもの（Docker の中で。video/README の手順）:
  outputs/video/clip_trials.mp4 と parts3.json（試し掴み）、carry_0/1/2.mp4（運ぶ 3 通り）、clip_line.mp4（ライン）、
  L1_grid.mp4（24 マス）、outputs/lesson/policy_W_hist.json（学習の記録。無ければ学習曲線は作らない）
"""
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V = ROOT / "outputs" / "video"
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_B = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
W, H = 1280, 720


def f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def ff(*args):
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg failed: {' '.join(args)}\n{r.stderr}")


def dur(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def box(d, xy, text, fill, size=26, color="black", bold=False):
    d.rounded_rectangle(xy, radius=14, fill=fill, outline=(60, 60, 60), width=2)
    x0, y0, x1, y1 = xy
    lines = text.split("\n")
    fh = size + 8
    y = (y0 + y1) / 2 - fh * len(lines) / 2
    for ln in lines:
        w = d.textlength(ln, font=f(size, bold))
        d.text(((x0 + x1 - w) / 2, y), ln, font=f(size, bold), fill=color)
        y += fh


def arrow(d, a, b, label=""):
    d.line([a, b], fill=(40, 40, 40), width=5)
    ang = np.arctan2(b[1] - a[1], b[0] - a[0])
    p1 = (b[0] - 18 * np.cos(ang - 0.4), b[1] - 18 * np.sin(ang - 0.4))
    p2 = (b[0] - 18 * np.cos(ang + 0.4), b[1] - 18 * np.sin(ang + 0.4))
    d.polygon([b, p1, p2], fill=(40, 40, 40))
    if label:
        d.text(((a[0] + b[0]) / 2 - d.textlength(label, font=f(22)) / 2, (a[1] + b[1]) / 2 - 38), label, font=f(22),
               fill=(40, 40, 40))


def diagram():
    """場面 3：全体の構成。"""
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((60, 40), "全体の構成", font=f(40, True), fill="black")
    box(d, (60, 150, 420, 560), "第一層（3 台）\n\n① 見る\nVLM（Gemini）が\n材質の候補を挙げる\n\n② 触る\n力を変えて試し掴み\n（落としてよい）",
        (235, 242, 250), 26)
    box(d, (510, 280, 770, 430), "データ\n重さ・μ", (255, 246, 220), 30, bold=True)
    box(d, (860, 150, 1220, 560), "第二層（1 台）\n\n③ 運ぶ\n学習した方策が\n握る力・加速度・\n指の向きを決め、\n弧を描いて運ぶ",
        (232, 245, 233), 26)
    arrow(d, (420, 355), (510, 355))
    arrow(d, (770, 355), (860, 355))
    d.text((60, 590), "AI 1：VLM（見る）　AI 2：学習した方策（運ぶ）", font=f(28), fill=(30, 30, 30))
    d.text((60, 640), "ノードのつなぎ方：dora-rs と ROS 2 の両方で動かした（第一層 ×3、第二層、記録）", font=f(24),
           fill=(80, 80, 80))
    im.save(V / "diagram.png")


def tray():
    """場面 4：トレイの画像に、Gemini が挙げた材質の候補を書き込む。"""
    src = ROOT / "outputs" / "tray" / "tray_0.png"
    txt = (ROOT / "data" / "vlm_responses_gemini" / "tray_0.gemini.txt").read_text()
    items = json.loads(txt.split("```json")[1].split("```")[0])
    im = Image.open(src).convert("RGB")
    w0, h0 = im.size
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    for k, it in enumerate(items):
        y, x = it["point"]
        px, py = x / 1000 * w0, y / 1000 * h0
        d.ellipse([px - 9, py - 9, px + 9, py + 9], outline=(230, 50, 50), width=4)
        d.text((px + 12, py - 14), str(k + 1), font=f(26, True), fill=(230, 50, 50))
    d.text((w0 + 16, 30), "Gemini の答え", font=f(28, True), fill="black")
    d.text((w0 + 16, 68), "（部品ごとの材質の候補）", font=f(18), fill=(80, 80, 80))
    y = 110
    for k, it in enumerate(items):
        d.text((w0 + 16, y), f"{k + 1}.", font=f(22, True), fill=(230, 50, 50))
        d.text((w0 + 50, y), "\n".join(it["candidates"]), font=f(20), fill="black")
        y += 30 * len(it["candidates"]) + 10
    d.text((w0 + 16, y + 10), "材質は候補まで。\nμ は見た目では\n分からない\n→ 試す範囲を\n狭めるのに使う", font=f(20),
           fill=(80, 80, 80))
    canvas.save(V / "tray_gemini.png")


def trials():
    """場面 5：試し掴みの映像（左）と、力の幅が狭まるグラフ（右）を並べた動画。"""
    d = json.load(open(V / "parts3.json"))
    k = 2  # 撮った環境（鋼）
    hist, truth, data = d["history"][k], d["truth"][k], d["data"][k]
    clip = V / "clip_trials.mp4"
    T = dur(clip)
    fps = 25
    t_first, t_each = 4.5, 1.6  # 1 回目の試行の始まり（重さを測った後）と、1 回の長さ [秒]
    frames = V / "_trials_frames"
    frames.mkdir(exist_ok=True)
    for p in frames.glob("*.png"):
        p.unlink()
    n = int((T + 3.0) * fps)  # 最後の結果（7 回目と μ）を 3 秒見せる
    for i in range(n):
        t = i / fps
        done = int(np.clip((t - t_first) // t_each, 0, len(hist))) if t >= t_first + t_each else 0
        fig, ax = plt.subplots(figsize=(6.4, 7.2), dpi=100)
        ax.set_xlim(0.5, len(hist) + 0.5)
        ax.set_ylim(0.3, 70)
        ax.set_yscale("log")
        ax.set_xlabel("試し掴みの回数")
        ax.set_ylabel("握る力 [N]（対数）")
        ax.set_title(f"鋼 {truth['mass_kg'] * 1000:.0f} g：落ちる境目を探す", fontsize=15)
        ax.axhline(60, color="gray", ls=":", lw=1)
        ax.text(0.6, 63, "重さを測る 60 N", color="gray", fontsize=10)
        for j in range(done):
            F, held, lo, hi = hist[j]
            ax.fill_between([j + 0.6, j + 1.4], lo, hi, color=(0.85, 0.9, 1.0))
            ax.plot(j + 1, F, "o" if held else "x", ms=12, mew=3, color="tab:green" if held else "tab:red")
        if done:
            lo, hi = hist[done - 1][2], hist[done - 1][3]
            ax.text(0.6, 22, f"持てた最小 {hi:.2f} N ／ 落ちた最大 {lo:.2f} N", fontsize=12)
        if done == len(hist):
            ax.text(0.6, 9.5, f"→ μ = mg / (2F) = {data['mu']:.3f}\n（正解 {truth['mu']:.3f}、"
                               f"{(data['mu'] / truth['mu'] - 1) * 100:+.0f}%：安全側）", fontsize=13, color="black",
                    bbox=dict(facecolor="white", edgecolor="tab:blue"))
        ax.plot([], [], "o", color="tab:green", label="持てた")
        ax.plot([], [], "x", color="tab:red", mew=3, label="落ちた")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(frames / f"{i:04d}.png")
        plt.close(fig)
    graph = V / "_trials_graph.mp4"
    ff("-framerate", str(fps), "-i", str(frames / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(graph))
    ff("-i", str(clip), "-i", str(graph), "-filter_complex",
       "[0:v]crop=820:720:230:0,scale=640:-2,pad=640:720:0:(oh-ih)/2:white,tpad=stop_mode=clone:stop_duration=3[a];"
       "[1:v]scale=640:720[b];[a][b]hstack",
       "-r", "25", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(V / "scene_trials.mp4"))


def carry():
    """場面 7：同じ部品を 3 通りで運ぶ映像を横に並べ、上に名前と結果を書く。"""
    labels = [("データなし", "60 N で強く握る → 運べた", (60, 60, 60)),
              ("式の方策・指は半径の向き", "19.8 N → 回って落とした", (200, 40, 40)),
              ("学習した方策（AI 2）", "13.0 N・指は 85° → 運べた", (30, 130, 50))]
    head = Image.new("RGB", (W, 150), "white")
    d = ImageDraw.Draw(head)
    d.text((20, 10), "同じ部品（POM 90 g、μ 0.059 の滑りやすい部品）を 3 通りで運ぶ", font=f(28, True), fill="black")
    for k, (a, b, col) in enumerate(labels):
        x = k * 427 + 10
        d.text((x, 62), a, font=f(24, True), fill=col)
        d.text((x, 100), b, font=f(22), fill=col)
    head.save(V / "_carry_head.png")
    ins = sum([["-i", str(V / f"carry_{k}.mp4")] for k in range(3)], [])
    ff(*ins, "-loop", "1", "-i", str(V / "_carry_head.png"), "-filter_complex",
       "[0:v]crop=560:570:360:40,scale=426:-2[a];[1:v]crop=560:570:360:40,scale=426:-2[b];[2:v]crop=560:570:360:40,scale=426:-2[c];"
       "[a][b][c]hstack=3,pad=1280:720:1:150:white[v];[v][3:v]overlay=0:0:shortest=1",
       "-r", "25", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(V / "scene_carry.mp4"))


def results():
    """場面 10：握る力の比べ。"""
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    names = ["データなし\n（全部品 60 N）", "データあり・式の方策\n（中央値）", "データあり・学習した方策\n（中央値、90 部品）"]
    vals = [60, 3.64, 2.68]
    cols = ["gray", "tab:blue", "tab:green"]
    b = ax.bar(names, vals, color=cols)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:g} N", ha="center", fontsize=22)
    ax.set_ylabel("握る力 [N]", fontsize=16)
    ax.set_title("見て・触って測れば、落とさず、小さい力で運べる（どれも成功 90/90）", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_ylim(0, 70)
    fig.tight_layout()
    fig.savefig(V / "results.png")
    plt.close(fig)


def learning():
    """場面 6：学習の記録（世代ごとの点数と、指の向き）。"""
    p = ROOT / "outputs" / "lesson" / "policy_W_hist.json"
    if not p.exists():
        print("skip learning curve (no history yet)")
        return
    h = json.load(open(p))["history"]
    g = np.array([x["gen"] for x in h])
    best, mean = np.array([x["best"] for x in h]), np.array([x["mean"] for x in h])
    # 指は左右対称なので、−78° と +102° は同じ持ち方。0°（半径の向き）〜 90°（接線の向き）に直して描く
    yaw = 90 - np.abs(90 - np.abs(np.array([x["yaw_deg"] for x in h])) % 180)
    kF, a = np.array([x["kF"] for x in h]), np.array([x["a"] for x in h])
    frames = V / "_learn_frames"
    frames.mkdir(exist_ok=True)
    for q in frames.glob("*.png"):
        q.unlink()
    fps, sec = 25, 10
    for i in range(fps * sec):
        n = max(1, int(len(g) * (i + 1) / (fps * sec * 0.8)))
        n = min(n, len(g))
        fig, axs = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100)
        axs[0].plot(g[:n], best[:n], "-o", color="tab:green", label="いちばん良い方策")
        axs[0].plot(g[:n], mean[:n], "-", color="tab:gray", label="16 通りの平均")
        axs[0].set_xlim(-1, len(g))
        axs[0].set_ylim(min(mean.min(), -1.5) - 0.2, 1.0)
        axs[0].set_xlabel("世代")
        axs[0].set_ylabel("点数（運べたら 1 − 0.3·力の倍率 − 時間、落としたら −3）")
        axs[0].legend(loc="lower right")
        axs[0].set_title("シミュレータの中で学ぶ（CEM、1 世代 = 16 方策 × 16 部品）", fontsize=14)
        axs[1].plot(g[:n], yaw[:n], "-o", color="tab:purple")
        axs[1].axhline(90, color="gray", ls=":")
        axs[1].text(0, 92, "接線の向き（90°）", color="gray")
        axs[1].set_ylim(-10, 110)
        axs[1].set_xlim(-1, len(g))
        axs[1].set_xlabel("世代")
        axs[1].set_ylabel("指の向き [度]（0 = 半径の向き、90 = 接線の向き）")
        axs[1].set_title(f"指の向き → {yaw[n - 1]:.0f}°　力の倍率 {kF[n - 1]:.2f}　加速度 {a[n - 1]:.1f}", fontsize=14)
        fig.tight_layout()
        fig.savefig(frames / f"{i:04d}.png")
        plt.close(fig)
    ff("-framerate", str(fps), "-i", str(frames / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
       str(V / "scene_learning.mp4"))


def failures():
    """場面 9：失敗と考察（3 つ）。右に、傾いてぶら下がった部品の窓の画面。"""
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((50, 36), "失敗と考察", font=f(40, True), fill="black")
    items = [("① 学習の「発見」は見かけだった",
              "指の向きを ±90° で切っていたため、でたらめな候補でも\n端（＝接線）に張り付いて正解に見えた。切らずに学び直し、85° に"),
             ("② 15 個送って 14 個が黙って消えた（dora）",
              "記録は「1/1 成功」。送った数と比べていなかった。\n届くはずの数と比べ、足りなければ MISSING と出すように"),
             ("③ ぶら下がった部品を「持てた」と数えた",
              "高さだけで判定していた。ビューアで見て気づき、\n傾きも判定に加えた（一気に持ち上げた場合 20/24 → 16/24）")]
    y = 120
    for t, body in items:
        d.text((50, y), t, font=f(28, True), fill=(180, 40, 40))
        d.text((70, y + 44), body, font=f(22), fill=(40, 40, 40))
        y += 180
    shot = Image.open(V / "tilt_viewer.png").convert("RGB").crop((200, 20, 1100, 820)).resize((380, 338))
    im.paste(shot, (870, 330))
    d.text((870, 675), "③ の場面（Genesis ビューア）", font=f(18), fill=(80, 80, 80))
    im.save(V / "failures.png")


if __name__ == "__main__":
    V.mkdir(parents=True, exist_ok=True)
    for fn in (diagram, tray, results, failures, carry, trials, learning):
        fn()
        print("done:", fn.__name__)
