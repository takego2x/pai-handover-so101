"""L0b-2: ロボットを置き、手先を指定した位置へ動かす（IK = 逆運動学）。

手先の目標（緑の玉）を 3 か所に順に置き、IK で関節の角度を計算して、関節をその角度へ向かわせる。
使い方: python lesson/first_arm.py [--mistake no_wait] [--viewer]
  --mistake no_wait : 関節に指令を出したあと、時計をほとんど進めずに（5 step）次の目標へ行く
  --viewer          : Genesis ビューアで見て、手先の軌道・目標までの距離・関節の角度を rerun へ送る
動画: outputs/first_arm.mp4
"""
import argparse

import common as c  # 画面で見る部品（--viewer）だけ借りる
import numpy as np
import genesis as gs

ap = argparse.ArgumentParser()
ap.add_argument("--mistake", choices=["no_wait"])
ap.add_argument("--viewer", action="store_true", help="Genesis ビューアと rerun で見る")
args = ap.parse_args()
view = c.viewer_view(pos=(0.6, -1.6, 0.9), lookat=(0.3, 0.0, 0.3)) if args.viewer else None
rr = c.rerun_start("L0b_first_arm") if args.viewer else None

TARGETS = [(0.5, 0.0, 0.4), (0.4, 0.3, 0.2), (0.3, -0.3, 0.5)]  # 手先の目標 [m]（ロボットの根元が原点）
DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # 手先を真下に向ける向き（クォータニオン。教材 §2.3）

gs.init(backend=gs.cpu, logging_level="warning")                                       # ① 起動
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), **c.viewer_kwargs(view))  # ② 場面（窓は --viewer のとき）
scene.add_entity(gs.morphs.Plane())                                                    # ③ 物を置く
arm = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))        #    ロボット（Genesis に同梱）
for p in TARGETS:  # 目標の印。fixed = 動かない、collision = False = ぶつからない（見るためだけ）
    scene.add_entity(gs.morphs.Sphere(radius=0.02, pos=p, fixed=True, collision=False),
                     surface=gs.surfaces.Default(color=(0.2, 0.8, 0.3)))
cam = None if view else scene.add_camera(res=(640, 480), pos=(0.5, -1.9, 0.9), lookat=(0.3, 0.0, 0.3), fov=45, GUI=False)
scene.build()                                                                          # ④ 組み立てる

hand = arm.get_link("hand")  # 手先（指の付け根）。IK はこのリンクの位置と向きを合わせる
print(f"joints: {arm.n_dofs}  (腕 7 + 指 2)")
# 関節の「ばね」の強さ（kp）と減衰（kv）。関節ごとに、目標の角度へ引き戻す力の強さを決める（L1 以降も同じ値）
arm.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
arm.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
# 構えの姿勢（HOME）から始める。全部 0 の姿勢から IK を解くと、思わぬ解に行くことがある（教材 L3-a）
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
arm.set_dofs_position(HOME)
arm.control_dofs_position(HOME)
if cam:
    cam.start_recording(save_to_filename="outputs/first_arm.mp4", fps=25)
step_no = 0


def step(n, target):
    """時計を n step 進める。--viewer のときは、step ごとに rerun へ送る。"""
    global step_no
    for _ in range(n):
        scene.step()
        step_no += 1
        if rr:
            pos = hand.get_pos().cpu().numpy()
            rr.set_time("sim", duration=step_no * 0.01)
            rr.log("world/hand", rr.Points3D([pos], radii=0.015, colors=[[60, 120, 230]]))
            rr.log("world/targets", rr.Points3D(TARGETS, radii=0.02, colors=[[50, 200, 80]]))
            rr.log("distance_to_target_mm", rr.Scalars(np.linalg.norm(pos - np.array(target)) * 1000))
            q = arm.get_dofs_position().cpu().numpy()
            for j in range(7):  # 腕の関節 7 個の角度 [rad]
                rr.log(f"joints/q{j + 1}", rr.Scalars(q[j]))


step(50, TARGETS[0])
wait = 5 if args.mistake == "no_wait" else 150
print("target [m]              reached [m]             error [mm]")
for p in TARGETS:
    q = arm.inverse_kinematics(link=hand, pos=np.array(p), quat=DOWN,   # 手先をここに → 関節 9 個の角度
                               init_qpos=arm.get_dofs_position())         # いまの姿勢から探し始める
    arm.control_dofs_position(q)  # 関節を q へ向かわせる指令。まだ動かない
    step(wait, p)                 # ⑤ 時計を進めると、関節が q へ向かって動く
    got = hand.get_pos().cpu().numpy()
    err = np.linalg.norm(got - np.array(p)) * 1000
    print(f"({p[0]:.2f}, {p[1]:+.2f}, {p[2]:.2f})    ({got[0]:.3f}, {got[1]:+.3f}, {got[2]:.3f})    {err:8.1f}")
if cam:
    cam.stop_recording()
    print("video: outputs/first_arm.mp4")
if view:
    c.hold_viewer(scene)
