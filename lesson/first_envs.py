"""L0b-3: 同じ場面を 4 つの環境に複製し、環境ごとに部品の μ だけを変える（並列の環境）。

傾き 20° の坂に 4 cm 角の箱を置く。理論では、μ < tan 20° = 0.364 なら滑り、それ以上なら止まる。
使い方: python lesson/first_envs.py [--mistake slope_mu] [--viewer [--env 0]]
  --mistake slope_mu : 坂の μ を 0.01 にしない（1.0 のまま）
  --viewer           : Genesis ビューアで 1 つの環境（--env の番号。0 = μ 0.2）を見て、4 つの箱が坂を下った距離を rerun へ送る
動画: outputs/first_envs.mp4（4 つの環境を横に並べ、左上に μ。最後の 1 秒は、滑った = 赤、止まった = 緑）
"""
import argparse
import math

import common as c  # 画面で見る部品と、録画をまとめる部品（stop_recording）だけ借りる
import numpy as np
import torch
import genesis as gs

ap = argparse.ArgumentParser()
ap.add_argument("--mistake", choices=["slope_mu"])
ap.add_argument("--viewer", action="store_true", help="Genesis ビューアと rerun で見る")
ap.add_argument("--env", type=int, default=0, help="ビューアで見る環境の番号（0〜3）")
args = ap.parse_args()
view = c.viewer_view(env=args.env, pos=(0.0, -1.0, 0.5), lookat=(0.0, 0.0, 0.22)) if args.viewer else None
rr = c.rerun_start("L0b_first_envs") if args.viewer else None

MUS = [0.2, 0.3, 0.45, 0.6]  # 環境 0〜3 の、箱の μ
N = len(MUS)
TILT = 20.0                   # 坂の傾き [度]
th = math.radians(TILT)

gs.init(backend=gs.cpu, logging_level="warning")                                          # ① 起動
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),                  # ② 場面
                 rigid_options=gs.options.RigidOptions(box_box_detection=True,             #   箱どうしの接触を正確に
                                                       friction_cone=gs.friction_cone.elliptic, impratio=10.0),
                 **(c.viewer_kwargs(view) if view                                          #   窓は --viewer のとき
                    else {"vis_options": gs.options.VisOptions(split_envs=True)}))        #   録画は環境ごとに別の動画
scene.add_entity(gs.morphs.Plane())                                                       # ③ 物を置く
# 坂：長さ 60 cm の板を y 軸のまわりに 20° 傾け、動かないように固定する。坂の μ は 0.01（接触の μ は大きいほうが効く）
slope = scene.add_entity(gs.morphs.Box(size=(0.6, 0.3, 0.02), pos=(0.0, 0.0, 0.25), euler=(0, TILT, 0), fixed=True),
                         material=gs.materials.Rigid(friction=1.0 if args.mistake == "slope_mu" else 0.01))
# 箱：坂の上側（-x の側）の面の上に、坂と同じ向きで置く。面の上向き（法線）= (sin θ, 0, cos θ)
n = np.array([math.sin(th), 0.0, math.cos(th)])
along = np.array([math.cos(th), 0.0, -math.sin(th)])  # 坂を下る向き
start = np.array([0.0, 0.0, 0.25]) - 0.15 * along + n * (0.01 + 0.02 + 0.001)
box = scene.add_entity(gs.morphs.Box(size=(0.04,) * 3, pos=tuple(start), euler=(0, TILT, 0)),
                       material=gs.materials.Rigid(friction=1.0),
                       surface=gs.surfaces.Default(color=(0.9, 0.35, 0.15)))
cam = None if view else scene.add_camera(res=(320, 240), pos=(0.0, -0.9, 0.4), lookat=(0.0, 0.0, 0.22), fov=50, GUI=False)
scene.build(n_envs=N)                                                                     # ④ 4 つに複製して組み立てる

# 環境ごとに、箱の摩擦に倍率を掛ける。箱の基準の摩擦は 1.0 なので、倍率がそのまま箱の μ になる
box.set_friction_ratio(torch.tensor(MUS, dtype=gs.tc_float, device=gs.device).unsqueeze(1), links_idx_local=[0])

if cam:
    c.start_recording(cam, "outputs/first_envs.mp4")
for k in range(200):                                                                      # ⑤ 2 秒進める
    scene.step()  # 1 回の step で、4 つの環境が同時に 0.01 秒進む
    if rr:  # 4 つの環境の箱が、坂に沿って下った距離 [cm]。1 本の線が 1 つの環境
        rr.set_time("sim", duration=(k + 1) * 0.01)
        d = (box.get_pos().cpu().numpy() - start) @ along * 100
        for i in range(N):
            rr.log(f"moved_cm/mu={MUS[i]:.2f}", rr.Scalars(d[i]))
pos = box.get_pos().cpu().numpy()  # 形は (4, 3)：環境の数 × (x, y, z)
moved = (pos - start) @ along * 100  # 坂に沿って下った距離 [cm]
slid = moved > 1.0
theory = np.array(MUS) < math.tan(th)
if cam:
    c.stop_recording(cam, "outputs/first_envs.mp4", cols=2, labels=[f"mu={m:.2f}" for m in MUS], ok=~slid)
print(f"tilt {TILT:.0f} deg: theory slides if mu < tan {TILT:.0f} = {math.tan(th):.3f}   mistake={args.mistake}")
print("  mu   moved [cm]  sim      theory")
for i in range(N):
    print(f"{MUS[i]:5.2f}  {moved[i]:9.1f}  {'slides' if slid[i] else 'stays':7s}  {'slides' if theory[i] else 'stays'}")
print(f"match {int((slid == theory).sum())}/{N}")
if view:
    c.hold_viewer(scene)
