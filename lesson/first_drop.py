"""L0b-1: Genesis の 5 段の形を体験する。箱を 1 個、高さ 1 m から落とし、理論（自由落下）と比べる。

使い方: python lesson/first_drop.py [--height 1.0] [--dt 0.01] [--viewer]
  --viewer : Genesis ビューア（desktop の箱）で動きを見て、高さを rerun（rerun の箱）でグラフにする。
             先に docker compose up -d desktop rerun。終わったら窓は開いたまま（Ctrl+C で終わる）
動画: outputs/first_drop.mp4（--viewer を付けないとき）
"""
import argparse
import math

import common as c  # 画面で見る部品（--viewer）だけ借りる。Genesis の使い方は、下にそのまま書いている
import genesis as gs

ap = argparse.ArgumentParser()
ap.add_argument("--height", type=float, default=1.0, help="箱の中心の、最初の高さ [m]")
ap.add_argument("--dt", type=float, default=0.01, help="1 ステップで進める時間 [s]")
ap.add_argument("--viewer", action="store_true", help="Genesis ビューアと rerun で見る")
args = ap.parse_args()
view = c.viewer_view(pos=(0.0, -2.6, 0.8), lookat=(0.0, 0.0, 0.4)) if args.viewer else None
rr = c.rerun_start("L0b_first_drop") if args.viewer else None

# ① 起動：計算は CPU で（Docker の中では Mac の GPU は使えない）
gs.init(backend=gs.cpu, logging_level="warning")

# ② 場面を作る：時間の刻み dt（既定 0.01 秒）。--viewer のときだけ窓（ビューア）を出す
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=args.dt),
                 show_viewer=bool(view),
                 viewer_options=gs.options.ViewerOptions(res=(1280, 800), camera_pos=view["pos"], camera_lookat=view["lookat"],
                                                         refresh_rate=round(1 / args.dt)) if view else None)

# ③ 物を置く：床と、10 cm 角の箱。録画のときはカメラも「物」の 1 つとして置く
scene.add_entity(gs.morphs.Plane())
box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, args.height)),
                       surface=gs.surfaces.Default(color=(0.9, 0.35, 0.15)))
cam = None if view else scene.add_camera(res=(640, 480), pos=(0.0, -2.6, 0.55), lookat=(0.0, 0.0, 0.5), fov=45, GUI=False)

# ④ 組み立てる：ここで初めて計算の準備（コンパイル）が走る。初回は数十秒かかる
scene.build()

# ⑤ 時計を進める：1 回の step で dt 秒。合計 1.5 秒
if cam:
    cam.start_recording(save_to_filename="outputs/first_drop.mp4", fps=25)
print("  t [s]   z sim [m]   z theory [m]")
n_steps, every = round(1.5 / args.dt), round(0.1 / args.dt)
for i in range(n_steps):
    scene.step()
    t = (i + 1) * args.dt
    z = float(box.get_pos()[2])  # 箱の中心の高さ。床に着くと 0.05（箱の半分）で止まる
    z_theory = max(args.height - 0.5 * 9.81 * t**2, 0.05)  # 自由落下 z = h - g t^2 / 2
    if rr:  # rerun へ：時刻 t に、2 本の線（シミュレータと理論）と、3D の箱
        rr.set_time("sim", duration=t)
        rr.log("height/sim", rr.Scalars(z))
        rr.log("height/theory", rr.Scalars(z_theory))
        rr.log("world/box", rr.Boxes3D(centers=[[0.0, 0.0, z]], half_sizes=[[0.05, 0.05, 0.05]], colors=[[230, 90, 40]]))
    if (i + 1) % every == 0:
        print(f"{t:6.2f}   {z:9.3f}   {z_theory:12.3f}")
if cam:
    cam.stop_recording()
t_land = math.sqrt(2 * (args.height - 0.05) / 9.81)
print(f"theory: lands at t = {t_land:.2f} s" + ("" if view else "   video: outputs/first_drop.mp4"))
if view:
    c.hold_viewer(scene)
