"""line.py の録画から、場面ごとの 12 秒の切り出しを作る（コマ番号は録画と同じ名前の .json の marks から）。

  python handover/cut.py <録画の名前.json> <画角> <箱> <始まりの工程> <終わりの工程> <出力.mp4> [--sec 12]
  工程: 1-search 1-weigh 2-place 3-B pick 4-B carry end（終わりは last で録画の最後、「2:1-search」で別の箱の工程）
  区間が 12 秒より長いと、12 秒に収まるよう倍速にする（倍率は標準出力に出す。字幕で明記する）
"""
import argparse
import json
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("meta")
ap.add_argument("cam")
ap.add_argument("box", type=int)
ap.add_argument("start")
ap.add_argument("stop")
ap.add_argument("out")
ap.add_argument("--sec", type=float, default=12.0)
args = ap.parse_args()

meta_path = Path(args.meta)
meta = json.loads(meta_path.read_text())
video = meta_path.with_name(f"{meta_path.stem}_{args.cam}.mp4")
frames = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                             "stream=nb_read_frames", "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout)
spf = meta["steps"] / frames   # 1 コマあたりの step 数（録画は 25 fps、シミュレーションは 100 step/秒 → 4）


def at(phase):
    box, phase = (int(phase.split(":")[0]), phase.split(":")[1]) if ":" in phase else (args.box, phase)   # 「2:1-search」で別の箱の工程
    hits = [m["step"] for m in meta["marks"] if m["box"] == box and m["phase"] == phase]
    if not hits:
        raise SystemExit(f"箱 {args.box} に工程 {phase} の印がない（失敗で途中で終わった箱は end まで行かない）。"
                         f"印: {[(m['box'], m['phase']) for m in meta['marks']]}")
    return hits[0]


t0 = at(args.start) / spf / meta["fps"]
t1 = (at(args.stop) / spf if args.stop != "last" else frames) / meta["fps"]
speed = max(1.0, (t1 - t0) / args.sec)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}", "-i", str(video),
                "-vf", f"setpts=PTS/{speed:.4f}", "-r", "25", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out], check=True)
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", args.out],
                           capture_output=True, text=True, check=True).stdout)
print(f"{args.out}: {t0:.1f}-{t1:.1f} s of {video.name} -> {dur:.1f} s (x{speed:.1f})")
