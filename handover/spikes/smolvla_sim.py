"""試し：SmolVLA（lerobot/smolvla_base、微調整なし）に、シミュレータの SO-101 を動かさせる。

Mac 本体で handover/spikes/smolvla_server.py を起動しておき、Docker の中でこれを流す:
  docker compose run --rm sim python handover/spikes/smolvla_sim.py --record outputs/smolvla/try.mp4
カメラ 3 つ（真上・手首・正面）の画像と関節の角度を推論役へ送り、返ってきた 50 コマ分の動きを、30 コマ／秒として順に指示する。
動きの当てはめ：学習データの平均からのずれ × ばらつき［度］を、シミュレータの構えの姿勢からのずれ［ラジアン］として足す。
"""
import argparse
import base64
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "handover"))
sys.path.insert(0, str(ROOT / "handover" / "spikes"))
import so101 as s  # noqa: E402
import softbox as sb  # noqa: E402
import vision as vi  # noqa: E402
from vlm_softboxes import surface  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--record", default="outputs/smolvla/try.mp4")
ap.add_argument("--chunks", type=int, default=6)
ap.add_argument("--task", default="Pick up the cardboard box and put it in the brown tray.")
ap.add_argument("--server", default="http://host.docker.internal:8765")
args = ap.parse_args()

STD = np.array([26.4, 52.4, 49.9, 37.0, 59.4, 19.0])     # 学習データの関節のばらつき［度］（smolvla_server.py と同じ）
STEPS_PER_ACTION = 3                                      # 学習データは 30 コマ／秒、シミュレータは 1 step 0.01 秒

sc = s.make_scene()
sc.add_entity(s.gs.morphs.Plane())
arm_e = s.add_so101(sc, (0, 0), 0, friction=0.1, model=vi.camera_model(s.MODEL))
spec = {"k": 7000, "mass": 0.08, "mu": 0.8, "look": "cardboard"}
f = sb.xml(ROOT / "outputs" / "smolvla" / "box.xml", spec["k"], spec["mass"], spec["mu"], spec["look"])
box = sc.add_entity(s.gs.morphs.MJCF(file=f, pos=(0.20, 0.0, 0)), surface=surface(s.gs, "cardboard"))
bx, by = 0.0, 0.22                                        # 大箱（根元から見て +90°）
for dx, dy, sx, sy in [(0.07, 0, 0.004, 0.144), (-0.07, 0, 0.004, 0.144), (0, 0.07, 0.144, 0.004), (0, -0.07, 0.144, 0.004)]:
    sc.add_entity(s.gs.morphs.Box(size=(sx, sy, 0.04), pos=(bx + dx, by + dy, 0.02), fixed=True),
                  surface=s.gs.surfaces.Default(color=(0.75, 0.6, 0.4)))
top = vi.add_top_camera(sc, center=(0.1, 0.1), height=0.9, fov=40, res=(256, 256))
front = sc.add_camera(res=(256, 256), pos=(0.55, -0.05, 0.30), lookat=(0.12, 0.05, 0.03), fov=50, GUI=False)
hand = vi.add_hand_camera(sc, res=(256, 256))
view = sc.add_camera(res=(1280, 720), pos=(0.45, -0.45, 0.40), lookat=(0.10, 0.08, 0.03), fov=50, GUI=False)
sc.build(n_envs=1)
sb.hold_spring(box)
arm = s.Arm(sc, arm_e, (0, 0), 0, 1)
vi.attach_hand(hand, arm_e)
for _ in range(30):
    sc.step()
# 箱の手前の上で構える（ここまでは決まった手順。ここから先を SmolVLA に任せる）
s.move(sc, [(arm, 0.0, 0.15, 0.10, s.OPEN)], 150)
home = arm_e.get_dofs_position().cpu().numpy()[0].copy()


def png(a):
    b = io.BytesIO()
    Image.fromarray(np.asarray(a)[..., :3].astype(np.uint8)).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


rec = ROOT / args.record
rec.parent.mkdir(parents=True, exist_ok=True)
view.start_recording(save_to_filename=str(rec), fps=25)
log, z0, zmax, t = [], float(box.get_pos().cpu().numpy()[0, 2]), 0.0, 0
for c in range(args.chunks):
    imgs = [np.asarray(top.render(rgb=True)[0]), vi.hand_rgb(hand), np.asarray(front.render(rgb=True)[0])]
    if c == 0:
        Image.fromarray(np.concatenate([i[..., :3] for i in imgs], axis=1).astype(np.uint8)).save(rec.with_name(rec.stem + "_inputs.png"))
    q = arm_e.get_dofs_position().cpu().numpy()[0]
    body = json.dumps({"state": q.tolist(), "images": [png(i) for i in imgs], "task": args.task}).encode()
    resp = json.loads(urllib.request.urlopen(urllib.request.Request(args.server, body, {"Content-Type": "application/json"}), timeout=120).read())
    acts = np.array(resp["action_norm"])
    for a in acts:
        tgt = home + a * STD * math.pi / 180
        tgt[5] = float(np.clip(tgt[5], s.CLOSE, s.OPEN))
        arm_e.control_dofs_position(tgt[None])
        for _ in range(STEPS_PER_ACTION):
            sc.step()
            t += 1
            zmax = max(zmax, float(box.get_pos().cpu().numpy()[0, 2]))
    tip = arm.tip()[0]
    p = box.get_pos().cpu().numpy()[0]
    log.append({"chunk": c, "tip": np.round(tip, 3).tolist(), "box": np.round(p, 3).tolist(),
                "joints_deg": np.round(np.degrees(arm_e.get_dofs_position().cpu().numpy()[0]), 1).tolist()})
    print(f"chunk {c}: 手先 {log[-1]['tip']}  箱 {log[-1]['box']}  関節 {log[-1]['joints_deg']}", flush=True)
view.stop_recording()
p = box.get_pos().cpu().numpy()[0]
res = {"task": args.task, "steps": t, "box_lifted_mm": round((zmax - z0) * 1000, 1),
       "box_moved_mm": round(float(np.linalg.norm(p[:2] - np.array([0.20, 0.0]))) * 1000, 1),
       "in_tray": bool(abs(p[0] - bx) < 0.068 and abs(p[1] - by) < 0.068), "chunks": log}
rec.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(f"箱が上がった最大 {res['box_lifted_mm']} mm、箱が動いた {res['box_moved_mm']} mm、大箱に入った {res['in_tray']} → {rec}")
