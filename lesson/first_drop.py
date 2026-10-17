"""L0b-1: Genesis の 5 段の形を体験する。箱を 1 個、高さ 1 m から落とし、理論（自由落下）と比べる。

使い方: python lesson/first_drop.py [--height 1.0] [--dt 0.01]
動画: outputs/first_drop.mp4
"""
import argparse
import math

import genesis as gs

ap = argparse.ArgumentParser()
ap.add_argument("--height", type=float, default=1.0, help="箱の中心の、最初の高さ [m]")
ap.add_argument("--dt", type=float, default=0.01, help="1 ステップで進める時間 [s]")
args = ap.parse_args()

# ① 起動：計算は CPU で（Docker の中では Mac の GPU は使えない）
gs.init(backend=gs.cpu, logging_level="warning")

# ② 場面を作る：時間の刻み dt（既定 0.01 秒）。画面のウィンドウは出さない（Docker には画面がない）
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=args.dt), show_viewer=False)

# ③ 物を置く：床と、10 cm 角の箱。カメラも「物」の 1 つとして置く
scene.add_entity(gs.morphs.Plane())
box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, args.height)),
                       surface=gs.surfaces.Default(color=(0.9, 0.35, 0.15)))
cam = scene.add_camera(res=(640, 480), pos=(0.0, -2.6, 0.55), lookat=(0.0, 0.0, 0.5), fov=45, GUI=False)  # 真横から

# ④ 組み立てる：ここで初めて計算の準備（コンパイル）が走る。初回は数十秒かかる
scene.build()

# ⑤ 時計を進める：1 回の step で dt 秒。合計 1.5 秒
cam.start_recording(save_to_filename="outputs/first_drop.mp4", fps=25)
print("  t [s]   z sim [m]   z theory [m]")
n_steps, every = round(1.5 / args.dt), round(0.1 / args.dt)
for i in range(n_steps):
    scene.step()
    t = (i + 1) * args.dt
    if (i + 1) % every == 0:
        z = float(box.get_pos()[2])  # 箱の中心の高さ。床に着くと 0.05（箱の半分）で止まる
        z_theory = max(args.height - 0.5 * 9.81 * t**2, 0.05)  # 自由落下 z = h - g t^2 / 2
        print(f"{t:6.2f}   {z:9.3f}   {z_theory:12.3f}")
cam.stop_recording()
t_land = math.sqrt(2 * (args.height - 0.05) / 9.81)
print(f"theory: lands at t = {t_land:.2f} s   video: outputs/first_drop.mp4")
