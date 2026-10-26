"""スライドの中に映像をはめ込んで、提出動画を作る。

  スライドの土台の画像（1280x720、映像の場所は枠だけ）に、決めた位置・大きさで映像を重ねる。
  映像が短いページは、最後のコマで止めておく。字幕は入れない（説明はスライドの中の帯・枠・表で示す）。

  python video/compose_slides.py video/storyboard_slides.json outputs/video2/PAI_submit_v3.mp4
絵コンテ：[{"page": "p5_0", "sec": 12, "videos": {"p5": {"clip": "...", "start": 0, "speed": 2.9}}}, {"clip": "表紙.mp4", "sec": 2.96}, ...]
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1280, 720, 30


def ff(*args):
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, text=True)
    if r.returncode != 0:      # 戻り値を捨てない
        raise SystemExit(f"ffmpeg failed: {' '.join(args)}\n{r.stderr}")


def main():
    board, out = json.load(open(sys.argv[1])), Path(sys.argv[2])
    layout = json.load(open(ROOT / "outputs" / "video2" / "slides_take_layout.json"))
    pages = {name: ROOT / "outputs" / "video2" / "pages" / f"{name}.png" for name in layout["order"]}
    enc = ["-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an"]
    with tempfile.TemporaryDirectory() as td:
        tmp, segs = Path(td), []
        for i, seg in enumerate(board):
            o = tmp / f"seg{i:02d}.mp4"
            if "clip" in seg:      # 表紙など、全画面の映像
                ff("-ss", str(seg.get("start", 0)), "-t", str(seg["sec"]), "-i", str(ROOT / seg["clip"]),
                   "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:white,setsar=1",
                   "-t", str(seg["sec"]), *enc, str(o))
            else:
                ins = ["-loop", "1", "-t", str(seg["sec"]), "-i", str(pages[seg["page"]])]
                chain, last = "[0:v]scale=1280:720,setsar=1[b0]", "b0"
                for k, sl in enumerate(layout["slots"].get(seg["page"], []), start=1):
                    v = seg["videos"][sl["name"]]
                    sp = float(v.get("speed", 1))
                    ins += ["-ss", str(v.get("start", 0)), "-t", str(seg["sec"] * sp), "-i", str(ROOT / v["clip"])]
                    # 倍速 → 枠の大きさへ → 足りない分は最後のコマで止める
                    chain += (f";[{k}:v]setpts=PTS/{sp},scale={sl['w']}:{sl['h']},setsar=1,"
                              f"tpad=stop_mode=clone:stop_duration={seg['sec']}[v{k}]"
                              f";[{last}][v{k}]overlay={sl['x']}:{sl['y']}:shortest=0[b{k}]")
                    last = f"b{k}"
                ff(*ins, "-filter_complex", chain, "-map", f"[{last}]", "-t", str(seg["sec"]), *enc, str(o))
            segs.append(o)
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in segs))
        out.parent.mkdir(parents=True, exist_ok=True)
        ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out))
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                               capture_output=True, text=True).stdout)
    print(f"{out}  {dur:.1f}s  {'OK (1-3 min)' if 60 <= dur <= 180 else 'NOT within 1-3 min'}")


if __name__ == "__main__":
    main()
