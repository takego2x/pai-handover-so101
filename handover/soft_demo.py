"""課題を見せる 3 場面（潰れる箱で）：
  crush：柔らかい箱（強度 7.5 N）を、最大の力でいきなり握る → 板が沈みきる（潰れた）→ そのまま持ち上げて運ぶ
  drop ：重い箱（250 g）を、ぎりぎり持ち上がる力（τ 0.3）で掴み、速く運ぶ → 途中で落とす
  slip ：重さが同じ（250 g）で摩擦だけ違う箱（1.0 と 0.4）を、2 台が同じ力（τ 0.3）で同時に持ち上げる → 摩擦の小さいほうだけ滑り落ちる

  docker compose run --rm sim python handover/soft_demo.py --mode slip [--record outputs/handover/demo_slip.mp4]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101 as s  # noqa: E402
import softbox as sb  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent / "spikes"))
from vlm_softboxes import surface  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["crush", "drop", "slip"], required=True)
ap.add_argument("--record")
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
R, ROBOT_MU = 0.22, 0.1
Y2 = 0.22                                     # slip の 2 台目の腕の位置（y）。2 つの箱が並んで見える近さ

CASES = {  # 腕ごとの箱 (k, 重さ kg, 摩擦, 見た目)
    "crush": [(2500.0, 0.15, 0.8, "cardboard")],
    "drop": [(20000.0, 0.25, 0.8, "cardboard")],
    "slip": [(20000.0, 0.25, 1.0, "cardboard"), (20000.0, 0.25, 0.4, "cardboard")],
}[args.mode]
sc = s.make_scene()
sc.add_entity(s.gs.morphs.Plane())
ents, boxes = [], []
for i, (k, m, mu, look) in enumerate(CASES):
    y = i * Y2
    ents.append(s.add_so101(sc, (0, y), 0, friction=ROBOT_MU))
    f = sb.xml(ROOT / "outputs" / "handover" / "softbox" / f"demo_{args.mode}_{i}.xml", k, m, mu, look,
               show_plate=(args.mode == "crush"))   # 潰す場面だけ、板が沈むのを見せる
    boxes.append(sc.add_entity(s.gs.morphs.MJCF(file=f, pos=(R, y, 0)), surface=surface(s.gs, look)))
yc = (len(CASES) - 1) * Y2 / 2
cam = None
if args.record:
    # 箱の向こう側（腕の正面）から、箱に寄って撮る。slip は 2 つの箱の真ん中を見る
    cam = (sc.add_camera(res=(1280, 720), pos=(0.42, -0.16, 0.13), lookat=(0.22, 0.0, 0.035), fov=40, GUI=False) if len(CASES) == 1 else
           sc.add_camera(res=(1280, 720), pos=(0.52, yc - 0.20, 0.16), lookat=(0.22, yc, 0.04), fov=50, GUI=False))
sc.build(n_envs=1)
for b in boxes:
    sb.hold_spring(b)
arms = [s.Arm(sc, e, (0, i * Y2), 0, 1) for i, e in enumerate(ents)]
for _ in range(30):
    sc.step()
if cam:
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    cam.start_recording(save_to_filename=args.record, fps=25)
r = R - s.JAW
peak = [0.0] * len(arms)
marks = []
t = [0]


def every(_k):
    t[0] += 1
    for i, (a, b) in enumerate(zip(arms, boxes)):
        peak[i] = max(peak[i], float(s.squeeze(a, b)[0]))


def all_move(z, g, n):
    s.move(sc, [(a, 0.0, r, z, g) for a in arms], n, every)


def mark(what):
    marks.append({"step": t[0], "what": what})
    print(f"[{t[0] / 100:5.1f} s] {what}", flush=True)


for a in arms:
    a.set_grip_tau(s.TAU_MAX)
all_move(0.08, s.OPEN, 120)
all_move(0.02, s.OPEN, 100)
mark("下ろした")
if args.mode == "crush":
    for a in arms:
        a.set_grip_tau(s.TAU_MAX)            # いきなり最大の力で
    all_move(0.02, s.CLOSE, 100)
    mark(f"最大の力で握った：沈み {sb.squash(boxes[0]) * 1000:.1f} mm（強度を超えると 8 mm 沈みきる）、挟む力の最大 {peak[0]:.1f} N")
    all_move(0.06, s.CLOSE, 100)
    s.move(sc, [(arms[0], 0.6, 0.2, 0.06, s.CLOSE)], 150, every)
    mark(f"潰したまま運んだ：潰れた＝{sb.crushed(boxes[0])}")
else:
    tau = 0.3
    for a in arms:
        a.set_grip_tau(s.TAU_MAX)
    all_move(0.02, s.PRE, 40)
    for a in arms:
        a.set_grip_tau(tau)
    all_move(0.02, s.CLOSE, 200)
    mark(f"τ {tau} で握った：挟む力 {[round(float(s.squeeze(a, b)[0]), 1) for a, b in zip(arms, boxes)]} N")
    all_move(0.06, s.CLOSE, 120)
    mark(f"持ち上げた：箱の高さ {[round(float(b.get_pos().cpu().numpy()[0, 2]) * 100, 1) for b in boxes]} cm")
    if args.mode == "drop":
        s.move(sc, [(arms[0], 0.9, 0.2, 0.06, s.CLOSE)], 40, every)      # 速く運ぶ（0.4 秒で 50°）
        s.move(sc, [(arms[0], 0.9, 0.2, 0.06, s.CLOSE)], 60, every)
        mark(f"速く運んだ：箱の高さ {float(boxes[0].get_pos().cpu().numpy()[0, 2]) * 100:.1f} cm")
    else:
        all_move(0.06, s.CLOSE, 80)
        mark(f"待った：箱の高さ {[round(float(b.get_pos().cpu().numpy()[0, 2]) * 100, 1) for b in boxes]} cm（摩擦 1.0 と 0.4）")
if cam:
    cam.stop_recording()
    Path(args.record).with_suffix(".json").write_text(json.dumps({"steps": t[0], "marks": marks}, ensure_ascii=False, indent=1))
