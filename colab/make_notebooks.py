"""Colab のノートを作る（ノートの JSON を手で直さず、このファイルを直して作り直す）。

  python colab/make_notebooks.py   → colab/H0_check.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def write(name, cells):
    nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    (HERE / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print("wrote", HERE / name)


H0 = [
    md("""
# H0 確認：Colab（無料版・CPU）で Genesis と SO-101 が動くか

このノートは教材の前の **確認だけ** です。上から順に「▶」を押し、最後のセルの表をそのまま Claude に貼ってください。

- ランタイムは **CPU のまま** でよい（「ランタイム」→「ランタイムのタイプを変更」→ CPU）。無料版の GPU は割り当てられないことがあるので、教材は CPU で作る
- 全部で 10 分ほど（入れるのに数分、形を細かく分ける計算に 1 分ほど）
"""),
    md("## 1. 入れる"),
    code("""
import time, os, sys, subprocess
T = {}
t0 = time.time()
!apt-get -qq install -y libosmesa6 > /dev/null
!pip -q install genesis-world==1.4.3 rerun-sdk[notebook]==0.38.1
T["install_s"] = round(time.time() - t0)
print("install", T["install_s"], "s")
"""),
    md("## 2. SO-101 の形を取ってくる（TheRobotStudio/SO-ARM100、手元と同じ版 a758567、Apache-2.0）"),
    code("""
if not os.path.exists("SO-ARM100"):
    !git clone -q https://github.com/TheRobotStudio/SO-ARM100.git
!cd SO-ARM100 && git checkout -q a758567 && git log --oneline -1
MODEL = "/content/SO-ARM100/Simulation/SO101/so101_new_calib.xml"
print(os.path.exists(MODEL), MODEL)
"""),
    md("## 3. 2 台の SO-101 と箱を置いて組み立てる"),
    code("""
os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # 画面のない計算機で絵を描く（GPU なし）
import numpy as np
import genesis as gs
gs.init(backend=gs.cpu, logging_level="warning")
sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
              rigid_options=gs.options.RigidOptions(box_box_detection=True), show_viewer=False)
sc.add_entity(gs.morphs.Plane())
D = 0.44
armA = sc.add_entity(gs.morphs.MJCF(file=MODEL, pos=(0, 0, 0), decompose_robot_error_threshold=0.0), material=gs.materials.Rigid(friction=1.0))
armB = sc.add_entity(gs.morphs.MJCF(file=MODEL, pos=(D, 0, 0), euler=(0, 0, 180), decompose_robot_error_threshold=0.0), material=gs.materials.Rigid(friction=1.0))
box = sc.add_entity(gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.2, -0.1, 0.015)), surface=gs.surfaces.Default(color=(0.72, 0.55, 0.36)))
cam = sc.add_camera(res=(768, 432), pos=(0.22, -0.55, 0.30), lookat=(0.22, 0.0, 0.05), fov=55, GUI=False)
t0 = time.time()
sc.build()
T["build_s"] = round(time.time() - t0, 1)
# 関節が全部 0 の姿勢では、前腕を水平に伸ばす（手先は根元から 22 cm 以上）。44 cm 離して向かい合わせると、真ん中で
# 2 台の手先が重なって刺さる。組み立てた直後に、たたんだ姿勢にしてから動かす（作品の共通部品 so101.py の FOLD と同じ）
FOLD = np.array([0.0, -1.7, 1.6, 1.2, 0.0, 0.0])
for a in (armA, armB):
    a.set_dofs_position(FOLD)
    a.control_dofs_position(FOLD)
print("build", T["build_s"], "s   dofs per arm:", armA.n_dofs)
"""),
    md("## 4. 速さを測る（1 秒ぶん＝100 step）と、1 コマ描く"),
    code("""
t0 = time.time()
for _ in range(100):
    sc.step()
T["sec_per_sim_sec"] = round(time.time() - t0, 2)
t0 = time.time()
rgb = cam.render(rgb=True)[0]
T["render_s"] = round(time.time() - t0, 2)
T["render_mean"] = round(float(np.asarray(rgb).mean()), 1)   # 0 なら真っ黒（描けていない）
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4.5)); plt.imshow(rgb); plt.axis("off"); plt.show()
print(T)
"""),
    md("## 5. 腕を動かして 3 秒の動画にし、ノートの中で再生する"),
    code("""
from IPython.display import HTML
import base64, imageio
frames = []
target = np.array([0.0, -0.6, 0.6, 1.2, 0.0, 0.8])   # たたんだ姿勢から、肩と肘を起こし、指を開く
t0 = time.time()
for k in range(300):
    s = min(1.0, k / 150)
    armA.control_dofs_position(FOLD + (target - FOLD) * s)
    sc.step()
    if k % 4 == 0:
        frames.append(cam.render(rgb=True)[0])
T["video_s"] = round(time.time() - t0, 1)
imageio.mimsave("check.mp4", frames, fps=25)
HTML(f'<video controls width=768 src="data:video/mp4;base64,{base64.b64encode(open("check.mp4","rb").read()).decode()}">')
"""),
    md("## 6. rerun をノートの中に出す（講義の第 5 回の道具）"),
    code("""
import rerun as rr
rr.init("H0_check")
for k in range(100):
    rr.set_time("t", duration=k * 0.01)
    rr.log("shoulder_q", rr.Scalars(float(np.sin(k / 10))))
try:
    rr.notebook_show(width=640, height=300)
    T["rerun_notebook"] = "shown"
except Exception as e:
    T["rerun_notebook"] = f"error: {e}"[:80]
"""),
    md("## 7. 結果（この表をそのまま Claude に貼る）"),
    code("""
import platform, torch
T["python"] = platform.python_version()
T["torch"] = torch.__version__
T["cpus"] = os.cpu_count()
T["genesis"] = gs.__version__
T["video_ok"] = os.path.getsize("check.mp4") > 10000
for k, v in T.items():
    print(f"{k:16s} {v}")
"""),
]

SETUP = """
import os, time
if "google.colab" in str(get_ipython()):
    !apt-get -qq install -y libosmesa6 fonts-noto-cjk > /dev/null
    !pip -q install genesis-world==1.4.3
    if not os.path.exists("SO-ARM100"):
        !git clone -q https://github.com/TheRobotStudio/SO-ARM100.git && cd SO-ARM100 && git checkout -q a758567
    MODEL = "/content/SO-ARM100/Simulation/SO101/so101_new_calib.xml"
else:   # 手元の Docker（答え合わせ用）
    MODEL = "/work/third_party/SO-ARM100/Simulation/SO101/so101_new_calib.xml"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np, matplotlib.pyplot as plt, genesis as gs
# グラフの日本語：Colab は matplotlib を先に読み込んでいるので、入れたフォントを直接登録する（名前だけ指定しても見つからない）
from matplotlib import font_manager
for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
"""

H1A = [
    md("""
# H1a SO-101 の体と、指示の出し方

**ねらい**：SO-101 に「関節をこの角度へ」と指示すると、モーターの中で何が起きているかを知る。
**ばね（kp）・減衰（kv）・力の上限** を自分で書き換えて、目標と実際の角度のずれをグラフで確かめる。

この 3 つは、作品の中でずっと効いている。
- **力の上限**：指の力の上限が、そのまま「握る力の設定」（作品の 0.25 N·m など）
- **ばねと減衰**：一気に目標を置くと行き過ぎる（わざと失敗 `jump`）
- **自分の腕の重さ**：肩のモーターがどれだけ力を出しているかで、持った箱の重さを測る（H1c）

**進め方**：上から ▶。「✏️ 変えてみる」のセルは、**予想を 1 行書いてから** 値を変えて ▶ し直す。
"""),
    md("## 1. 準備（Colab では 3〜5 分。2 回目からは速い）"),
    code(SETUP),
    md("""
## 2. SO-101 を 1 台置く

`gs.morphs.MJCF`：ロボットの形と関節を書いたファイル（MJCF。MuJoCo の形式）を読む。
`decompose_robot_error_threshold=0.0`：ぶつかり判定のために、形を細かい凸の部品に分けて包む（0 がいちばん細かい。H2 で粗くして失敗させる）。
組み立て（`build`）は時間がかかるので **1 回だけ**。条件を変えるたびに `reset` で初めの状態に戻す。
"""),
    code("""
gs.init(backend=gs.cpu, logging_level="warning")
sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10), show_viewer=False)
sc.add_entity(gs.morphs.Plane())
arm = sc.add_entity(gs.morphs.MJCF(file=MODEL, decompose_robot_error_threshold=0.0))
cam = sc.add_camera(res=(768, 432), pos=(0.45, -0.45, 0.30), lookat=(0.05, 0.0, 0.12), fov=45, GUI=False)
t0 = time.time(); sc.build(); print(f"build {time.time() - t0:.0f} s")
"""),
    md("""
## 3. 関節を一覧にする

SO-101 の関節は 6 つ。5 つが腕、最後の 1 つが **指**（動く側の指を回す関節）。
どの関節にも、角度の範囲と、モーターが出せる **力の上限**（回す力＝トルク、単位 N·m）がある。
"""),
    code("""
names = [j.name for j in arm.joints if j.n_dofs == 1]
lo, hi = [x.cpu().numpy() for x in arm.get_dofs_limit()]
f_lo, f_hi = [x.cpu().numpy() for x in arm.get_dofs_force_range()]
kp, kv = arm.get_dofs_kp().cpu().numpy(), arm.get_dofs_kv().cpu().numpy()
print(f"{'#':2s} {'関節':14s} {'角度の範囲 [rad]':>18s} {'力の上限 [N·m]':>14s} {'kp':>8s} {'kv':>6s}")
for i, n in enumerate(names):
    print(f"{i:2d} {n:14s} {lo[i]:8.2f} 〜 {hi[i]:5.2f} {f_hi[i]:14.2f} {kp[i]:8.1f} {kv[i]:6.2f}")
"""),
    md("""
**期待される出力**（手元の実測）：6 行。`gripper`（指）は 5 番。力の上限は全部 3.35 N·m、kp は 998.22、kv は 2.73。
kp・kv・力の上限は、実機のモーター（Feetech STS3215）に合わせた値で、モデルのファイルに書いてある。
"""),
    md("""
## 4. 関節に角度を指示して、目標と実際を比べる

`control_dofs_position(目標の角度)`：**「この角度へ行け」という指示**。モーターは、目標とのずれに比例した力（ばね kp）と、速さに比例したブレーキ（減衰 kv）を出す。
出せる力は、力の上限で頭打ちになる。式で書くと：

  力 ＝ kp ×（目標 − 角度） − kv × 角速度　　（ただし ±力の上限 で頭打ち）

`step_response` は、肩（`shoulder_lift`、1 番）だけに 0 → −0.6 rad（後ろへ起こす向き）を指示して、2 秒間の角度と力を記録する関数。
前へ倒す向き（＋）にすると、手先が床に当たって止まり、kp を変えても結果が変わらない（教材を作るときに実際に踏んだ。グラフだけ見ると、うまくいった実験に見える）。
"""),
    code("""
J = 1   # 肩（shoulder_lift）

def step_response(target=-0.6, kp=998.22, kv=2.731, f_max=3.35, ramp=0, steps=200, mode="position"):
    \"\"\"肩だけを動かす。ramp > 0 なら、目標を ramp step かけて少しずつ置く（S 字）。記録した角度と力を返す。\"\"\"
    sc.reset()
    arm.set_dofs_kp(np.full(6, 998.22)); arm.set_dofs_kv(np.full(6, 2.731))
    arm.set_dofs_force_range(np.full(6, -3.35), np.full(6, 3.35))
    arm.set_dofs_kp([kp], [J]); arm.set_dofs_kv([kv], [J])
    arm.set_dofs_force_range([-f_max], [f_max], [J])
    q, f = [], []
    for k in range(steps):
        s = 1.0 if ramp == 0 else min(1.0, (k + 1) / ramp)
        s = s if ramp == 0 else 3 * s ** 2 - 2 * s ** 3
        goal = np.zeros(6); goal[J] = target * s
        if mode == "position":
            arm.control_dofs_position(goal)
        elif mode == "force":       # 力を直接指示する（ここでは 0 N·m＝力を抜く）
            arm.control_dofs_force(np.zeros(6))
        sc.step()
        q.append(float(arm.get_dofs_position()[J])); f.append(float(arm.get_dofs_control_force()[J]))
    return np.array(q), np.array(f)

def plot(runs, target=-0.6):
    t = np.arange(len(next(iter(runs.values()))[0])) * 0.01
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    for label, (q, f) in runs.items():
        ax[0].plot(t, q, label=label); ax[1].plot(t, f, label=label)
    ax[0].axhline(target, color="gray", ls="--", lw=0.8); ax[0].set(xlabel="s", ylabel="肩の角度 [rad]")
    ax[1].set(xlabel="s", ylabel="肩のモーターの力 [N·m]"); ax[0].legend(); plt.show()
    for label, (q, f) in runs.items():
        u = np.sign(target) * q                       # 目標の向きに測った角度（向きを見ないと、反対へ倒れたのを「届いた」と数える）
        reach = f"{np.argmax(u >= 0.9 * abs(target)) * 0.01:.2f} s" if (u >= 0.9 * abs(target)).any() else "届かない"
        print(f"{label:28s} 行き過ぎ {max(u.max() - abs(target), 0):.3f} rad   9 割に届く {reach:>8s}   2 秒後のずれ {target - q[-1]:+.4f} rad   最後の力 {f[-1]:+.2f} N·m")

plot({"いまの値（kp 998, kv 2.7）": step_response()})
"""),
    md("""
**期待される出力**（手元の実測）
```
いまの値（kp 998, kv 2.7）  行き過ぎ 0.078 rad   9 割に届く   0.18 s   2 秒後のずれ -0.0002 rad   最後の力 -0.21 N·m
```
- 0.18 秒で目標の 9 割に届き、0.078 rad（約 4.5°）行き過ぎてから戻る
- 右のグラフ：力は最初、力の上限（3.35 N·m）に張り付く。止まると −0.21 N·m だけ出し続ける。**これが腕の重さを支える力**（H1c で、箱を持ったときのこの力の増え方から重さを測る）
"""),
    md("""
## ✏️ 変えてみる 1：ばね kp
**予想**：kp を 1/10（100）にすると、行き過ぎと 2 秒後のずれはどうなる？ 10 倍（10000）では？
"""),
    code("""
plot({f"kp {kp}": step_response(kp=kp) for kp in [20, 100, 998.22, 10000]})
"""),
    md("""
## ✏️ 変えてみる 2：減衰 kv
**予想**：kv を 0 にすると？ 30 にすると？
"""),
    code("""
plot({f"kv {kv}": step_response(kv=kv) for kv in [0.0, 2.731, 30.0]})
"""),
    md("""
## ✏️ 変えてみる 3：力の上限
**予想**：肩の力の上限を 1.0 N·m、0.3 N·m にすると、腕は目標に届く？
（作品では、**指** の力の上限を 0.2〜0.8 N·m に絞って「握る力」にしている。肩で同じことをすると何が起きるか）
"""),
    code("""
plot({f"力の上限 {fm} N·m": step_response(f_max=fm) for fm in [3.35, 1.0, 0.3]})
"""),
    md("""
## ✏️ 変えてみる 4：目標の出し方（一気に／S 字で少しずつ）
**予想**：目標を一気に置く（いまの書き方）のと、1 秒かけて S 字で少しずつ置くのとで、行き過ぎはどう違う？
作品の共通部品 `move` は S 字で置いている。一気に置くのが、わざと失敗 `jump`。
"""),
    code("""
plot({"一気に": step_response(ramp=0), "S 字で 0.5 秒": step_response(ramp=50), "S 字で 1 秒": step_response(ramp=100)})
"""),
    md("""
## ✏️ 変えてみる 5：指示の種類
位置ではなく **力** を指示する（`control_dofs_force`）。0 N·m を指示すると、モーターは力を抜く。
**予想**：肩はどうなる？
"""),
    code("""
plot({"位置で指示": step_response(), "力 0 を指示": step_response(mode="force")})
"""),
    md("""
## 5. 映像で確かめる
どれか 1 つの条件で、腕の動きを録画して見る。`step_response` の中身と同じ指示を出しながら、4 step に 1 コマ描く。
**変えてみる**：`kp=` や `f_max=` を変えて ▶ し直す。
"""),
    code("""
from IPython.display import HTML
import base64, imageio

def film(target=-0.6, kp=998.22, kv=2.731, f_max=3.35, steps=150, name="h1a.mp4"):
    sc.reset()
    arm.set_dofs_kp([kp], [J]); arm.set_dofs_kv([kv], [J]); arm.set_dofs_force_range([-f_max], [f_max], [J])
    frames = []
    for k in range(steps):
        goal = np.zeros(6); goal[J] = target
        arm.control_dofs_position(goal); sc.step()
        if k % 4 == 0:
            frames.append(cam.render(rgb=True)[0])
    imageio.mimsave(name, frames, fps=25)
    return HTML(f'<video controls width=640 src="data:video/mp4;base64,{base64.b64encode(open(name, "rb").read()).decode()}">')

film(f_max=0.3)
"""),
    md("""
## 確かめる問い
1. 2 秒後に目標とのずれが残るのはなぜ？ kp を大きくすると小さくなるのはなぜ？
2. 作品で「握る力」を決めているのは、指のモーターの何？
3. 作品の共通部品 `move` が、目標を S 字で少しずつ置くのはなぜ？

<details><summary>答え</summary>

1. 腕の重さを支える力が要るから。ばねの力 kp ×（ずれ）が重さと釣り合う所で止まる。kp が大きいほど、同じ力を小さいずれで出せる
2. 力の上限（`set_dofs_force_range`）。指は「閉じきれ」と指示されていて、箱に当たって止まると、上限いっぱいの力で押す
3. 一気に置くと、ずれが大きく、力の上限いっぱいで加速して行き過ぎるから（わざと失敗 `jump` で箱の上に着地する）
</details>
"""),
]

H1B = [
    md("""
# H1b 手先で指示する（逆運動学と回転）

**ねらい**：H1a では関節に角度で指示した。作品では「手先をこの位置へ、真下に向けて」と指示している。
手先の位置と向きから関節の角度を逆算するのが **逆運動学（IK）**。ここでは、
- 手先の **位置** の指示：どの点を動かしているのか（目印 `gripperframe`）
- 手先の **向き** の指示：真下に向ける回転の作り方
- **届く範囲**：真下向きのまま、どこまで届くか
- **解き直す間隔**：動きながら IK を何 step ごとに解き直すか
- **2 台目の向き**：180° 回して置いた腕では、回転の作り方を変えないと手先が横を向く（わざと失敗 `world_tilt`）

を、自分で値を書き換えて確かめる。
"""),
    md("## 1. 準備"),
    code(SETUP),
    md("""
## 2. 道具：回転を扱う関数

- `rot(q)`：クォータニオン（向きを表す 4 つの数 w, x, y, z）→ 回転の行列（3×3）
- `quat_of(R)`：回転の行列 → クォータニオン（IK に向きを渡すときの形）
- `Ry(a)`・`Rz(a)`：y 軸・z 軸のまわりに a [rad] 回す行列。**行列を左から掛けると、世界の軸のまわりに回す**
- `OFF`：手首の部品（`gripper` リンク）の原点から、**目印**（固定側の指の内側の先端、`gripperframe`）までのずれ。IK は部品の原点を動かすので、目印を目標に置くには、このずれを引いて指示する
"""),
    code("""
import math
OFF = np.array([-0.0079, -0.000218, -0.0981])        # gripper の原点 → 目印（gripper の向きで見た長さ [m]）
FOLD = np.array([0.0, -1.7, 1.6, 1.2, 0.0, 0.0])     # たたんだ姿勢（H0 で刺さった話を参照）
OPEN, CLOSE = 1.2, -0.17                             # 指の角度（開く／閉じきる）

def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def quat_of(R):
    \"\"\"回転の行列 → クォータニオン。どの向きでも 0 で割らない形（作品の共通部品と同じ）。\"\"\"
    t = np.trace(R)
    if t > 0:
        S = math.sqrt(t + 1.0) * 2; q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.array(q); return q / np.linalg.norm(q)

def Ry(a): return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
def Rz(a): return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
"""),
    md("""
## 3. 2 台を置く

その 1 は原点に、その 2 は 44 cm 先に **180° 回して**（向かい合わせに）置く。赤い玉は「手先をここへ」の目標（ぶつからない飾り）。
"""),
    code("""
gs.init(backend=gs.cpu, logging_level="warning")
sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10), show_viewer=False)
sc.add_entity(gs.morphs.Plane())
D = 0.44
A = sc.add_entity(gs.morphs.MJCF(file=MODEL, pos=(0, 0, 0), decompose_robot_error_threshold=0.0))
B = sc.add_entity(gs.morphs.MJCF(file=MODEL, pos=(D, 0, 0), euler=(0, 0, 180), decompose_robot_error_threshold=0.0))
goal = sc.add_entity(gs.morphs.Sphere(radius=0.008, pos=(0.22, 0, 0.06), fixed=True, collision=False),
                     surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1)))
cam = sc.add_camera(res=(768, 432), pos=(0.22, -0.50, 0.28), lookat=(0.22, 0.0, 0.06), fov=50, GUI=False)
t0 = time.time(); sc.build(); print(f"build {time.time() - t0:.0f} s")
gA, gB = A.get_link("gripper"), B.get_link("gripper")

# 関節 0 の姿勢での手首の向き R0 を測る（真下向きの回転は、この向きを起点に作る）。測ったら、たたんだ姿勢へ
A.set_dofs_position(np.zeros(6)); B.set_dofs_position(np.zeros(6))
R0A, R0B = rot(gA.get_quat().cpu().numpy()), rot(gB.get_quat().cpu().numpy())
FINGER = R0A.T @ np.array([1.0, 0, 0])      # 手首の部品から見た「指の向き」（関節 0 の姿勢では、その 1 の指は世界の +x を向く）
for arm in (A, B):
    arm.set_dofs_position(FOLD); arm.control_dofs_position(FOLD)

def tip(arm, g):
    \"\"\"目印（固定側の指の内側の先端）の位置。\"\"\"
    return g.get_pos().cpu().numpy() + rot(g.get_quat().cpu().numpy()) @ OFF

def tilt_from_down(g):
    \"\"\"指の向きが、真下から何度ずれているか。\"\"\"
    d = rot(g.get_quat().cpu().numpy()) @ FINGER
    return math.degrees(math.acos(max(-1.0, min(1.0, -d[2]))))

def show():
    sc.visualizer.update(force=True)   # 瞬間移動（set_dofs_position）の後は、時間を進めないと絵が古いままなので、描く前に更新する
    plt.figure(figsize=(8, 4.5)); plt.imshow(cam.render(rgb=True)[0]); plt.axis("off"); plt.show()

print("R0（その 1）=\\n", R0A.round(3))
show()
"""),
    md("""
**期待される出力**：2 台ともたたんだ姿勢。R0 は、関節 0 の姿勢で手首の部品が世界に対してどう回っているか。
"""),
    md("""
## 4. 手先を、位置と向きで指示する

真下向きの回転 `Rd = Ry(90°) @ R0`：関節 0 の姿勢（指は水平に前を向く）から、**世界の y 軸のまわりに 90° 倒す** と、指が真下を向く。
`inverse_kinematics(link=手首, pos=目標 − Rd @ OFF, quat=向き)` は、そうなる関節の角度を返す。
ここでは、返った角度に **瞬間移動**（`set_dofs_position`）して、目印が目標に来たか・真下を向いたかだけを確かめる（重さで垂れる話は 6 で）。
"""),
    code("""
def reach(arm, g, R0, base, yaw, r, z, tilt=math.pi / 2, world_axis=False):
    \"\"\"arm の手先（目印）を、根元から見て前へ r [m]・高さ z [m] に、真下向き（tilt）で置く角度を IK で解き、そこへ瞬間移動する。
    world_axis=True：世界の y 軸のまわりに倒す（その 1 は同じ。180° 回したその 2 では違う）\"\"\"
    target = np.array([base[0] + r * math.cos(yaw), base[1] + r * math.sin(yaw), z])
    Rd = Ry(tilt) @ R0 if world_axis else Rz(yaw) @ Ry(tilt) @ Rz(-yaw) @ R0    # 腕自身の向きで見た横の軸のまわりに倒す
    arm.set_dofs_position(FOLD)
    q = arm.inverse_kinematics(link=g, pos=target - Rd @ OFF, quat=quat_of(Rd), init_qpos=arm.get_dofs_position(),
                               max_solver_iters=200).cpu().numpy()
    q[5] = OPEN
    arm.set_dofs_position(q)
    err = np.linalg.norm(tip(arm, g) - target) * 1000
    return q, err, tilt_from_down(g)

goal.set_pos(np.array([0.22, 0, 0.06]))
q, err, tl = reach(A, gA, R0A, (0, 0), 0.0, r=0.22, z=0.06)
print(f"関節の角度 {q.round(2)}   目印のずれ {err:.1f} mm   真下からのずれ {tl:.1f}°")
show()
"""),
    md("""
**期待される出力**：目印のずれは数 mm 以下、真下からのずれは 1° 未満。窓の絵で、その 1 の指先が赤い玉に下りている。
"""),
    md("""
## ✏️ 変えてみる 1：倒す角度
**予想**：`tilt` を 60°、120° にすると、指はどちらを向く？
"""),
    code("""
for deg in [60, 90, 120]:
    q, err, tl = reach(A, gA, R0A, (0, 0), 0.0, r=0.22, z=0.06, tilt=math.radians(deg))
    print(f"tilt {deg:3d}°   目印のずれ {err:5.1f} mm   真下からのずれ {tl:5.1f}°")
"""),
    md("""
## ✏️ 変えてみる 2：届く範囲
真下向きのまま、根元からの距離 r と高さ z を振る。**手首（`wrist_flex`、3 番）が可動範囲の端（1.66 rad）に張り付いた** ら、それ以上は真下を向けない。
**予想**：届くのは、根元から何 cm〜何 cm？ 高い所と低い所、どちらが届きやすい？
"""),
    code("""
RS = [0.12, 0.16, 0.20, 0.24, 0.28, 0.32]
ZS = [0.02, 0.06, 0.10, 0.15]
lim = 1.658
print("目印のずれ [mm] / 真下からのずれ [°]（* は手首が可動範囲の端）")
print("z \\\\ r  " + "".join(f"{r * 100:>12.0f}cm" for r in RS))
for z in ZS:
    row = []
    for r in RS:
        q, err, tl = reach(A, gA, R0A, (0, 0), 0.0, r, z)
        row.append(f"{err:5.0f}/{tl:3.0f}{'*' if abs(q[3]) > lim - 0.02 else ' '}")
    print(f"{z * 100:3.0f}cm   " + "".join(f"{c:>14s}" for c in row))
"""),
    md("""
**見るところ**：ずれが小さく（数 mm）真下（0°）のマスが「届く」。作品では **根元から 16〜28 cm・低い所** だけを使っている（台座 2 cm、運ぶ高さ 6〜7 cm）。
"""),
    md("""
## 5. 2 台目：180° 回して置いた腕
その 2 で、同じ「真下に倒す」を 2 通りの作り方で解く。
- **腕自身の向きで倒す**：`Rz(yaw) @ Ry(90°) @ Rz(−yaw) @ R0`（いったん腕の向きを 0 に戻して倒し、また腕の向きへ回す）
- **世界の軸で倒す**：`Ry(90°) @ R0`（その 1 と同じ式。わざと失敗 `world_tilt`）

**予想**：世界の軸で倒すと、その 2 の指はどちらを向く？
"""),
    code("""
goal.set_pos(np.array([D - 0.22, 0, 0.06]))
A.set_dofs_position(FOLD)   # その 1 はたたんでおく（前の表の最後の姿勢が残っていると見づらい）
for world_axis in [False, True]:
    q, err, tl = reach(B, gB, R0B, (D, 0), math.pi, r=0.22, z=0.06, world_axis=world_axis)
    print(f"{'世界の軸で倒す' if world_axis else '腕自身の向きで倒す'}   目印のずれ {err:5.1f} mm   真下からのずれ {tl:5.1f}°")
    show()
"""),
    md("""
**なぜ**：`Ry(90°)` を左から掛けると、**世界の** y 軸のまわりに回る。その 1（向き 0°）では、世界の y 軸が腕の横の軸と同じなので、指が前から下へ倒れる。
180° 回したその 2 では、R0B の指は世界の −x を向いている。世界の y 軸のまわりに +90° 回すと、−x を向いた指は **上** へ倒れる（または IK が届かない姿勢になり、横を向いて止まる）。
"""),
    md("""
## 6. 動かしながら解き直す（実際の動き）
4・5 は瞬間移動だった。実際は、目標を S 字で少しずつ動かしながら、**5 step ごとに IK を解き直して** 関節に指示する（作品の共通部品 `move` と同じ）。
ここでは、掴みに行くときと同じ動き：真下向きで箱の上（高さ 8 cm）に置いてから、**高さ 2 cm（3 cm の箱の真ん中）まで下ろす**。
動かす間、指は **閉じておく**（開いたまま動かすと、向かいのその 2 の指に当たる。✏️ 4 と 📕 2）。

**予想**：下ろす時間を 1 秒・0.3 秒にすると、また目標を一気に置く（わざと失敗 `jump`）と、手先は 2 cm より下へ行き過ぎる？
（作品では、行き過ぎると箱の上に着地したり、箱を押しのけたりした）
"""),
    code("""
def descend(arm, g, R0, r=0.22, z_from=0.08, z_to=0.02, n=100, every=5, grip=CLOSE, jump=False):
    \"\"\"真下向きで (r, z_from) に置いてから、n step かけて S 字で z_to まで下ろす。every step ごとに IK を解き直す。
    jump=True：目標を一度に置く（わざと失敗 jump）。手先の通り道を返す\"\"\"
    sc.reset()
    B.set_dofs_position(FOLD); B.control_dofs_position(FOLD)
    q, _, _ = reach(arm, g, R0, (0, 0), 0.0, r, z_from); q[5] = grip
    arm.set_dofs_position(q); arm.control_dofs_position(q)
    for _ in range(30):
        sc.step()
    start, target = tip(arm, g), np.array([r, 0.0, z_to])
    Rd = Ry(math.pi / 2) @ R0
    path = []
    for k in range(n + 50):
        s = min(1.0, (k + 1) / (0.7 * n)); s = 1.0 if jump else 3 * s ** 2 - 2 * s ** 3
        cur = start + (target - start) * s
        if k % every == 0:
            q = arm.inverse_kinematics(link=g, pos=cur - Rd @ OFF, quat=quat_of(Rd), init_qpos=arm.get_dofs_position(),
                                       max_solver_iters=50).cpu().numpy()
            q[5] = grip
            arm.control_dofs_position(q)
        sc.step()
        path.append(tip(arm, g))
    return np.array(path)

goal.set_pos(np.array([0.22, 0, 0.02]))
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
for label, kw in [("S 字で 1 秒", dict(n=100)), ("S 字で 0.3 秒", dict(n=30)), ("一気に（jump）", dict(n=100, jump=True))]:
    P = descend(A, gA, R0A, **kw)
    t_ = np.arange(len(P)) * 0.01
    ax[0].plot(t_, P[:, 2] * 100, label=label); ax[1].plot(t_, (P[:, 0] - 0.22) * 1000, label=label)
    print(f"{label:12s}  2 cm より下へ行き過ぎ {max(0.02 - P[:, 2].min(), 0) * 1000:4.1f} mm   横へのずれの最大 {np.abs(P[:, 0] - 0.22).max() * 1000:4.1f} mm   着いた所 {P[-1].round(3)}")
ax[0].axhline(2, color="gray", ls="--", lw=0.8); ax[0].set(xlabel="s", ylabel="目印の高さ [cm]"); ax[0].legend()
ax[1].set(xlabel="s", ylabel="横（前後）へのずれ [mm]"); plt.show()
show()
"""),
    md("""
**期待される出力**（手元の実測）
```
S 字で 1 秒      2 cm より下へ行き過ぎ  0.5 mm   横へのずれの最大  6.7 mm
S 字で 0.3 秒    2 cm より下へ行き過ぎ  0.9 mm   横へのずれの最大  5.7 mm
一気に（jump）   2 cm より下へ行き過ぎ  9.3 mm   横へのずれの最大 12.1 mm
```
一気に置くと、目印が床から 1 cm まで下がる。箱があれば、箱の上の面に当たって押しのける。
"""),
    md("""
## ✏️ 変えてみる 3：IK を解き直す間隔
**予想**：`every` を 1・25 にすると？（作品は 5。並列の環境が多いと IK の計算が重いので、間引いている）
"""),
    code("""
for every in [1, 5, 25]:
    t0 = time.time(); P = descend(A, gA, R0A, n=100, every=every)
    print(f"{every:2d} step ごと   2 cm より下へ行き過ぎ {max(0.02 - P[:, 2].min(), 0) * 1000:4.1f} mm   横へのずれの最大 {np.abs(P[:, 0] - 0.22).max() * 1000:4.1f} mm   計算 {time.time() - t0:4.1f} s")
"""),
    md("""
## ✏️ 変えてみる 4：指を開いたまま、たたんだ姿勢から動かす
作品の初めと同じ、たたんだ姿勢から箱の上へ動かす。**予想**：指を開いたまま（`OPEN`）だと、着いたときのずれは？ 腕のモーターの力は？
"""),
    code("""
def from_fold(grip, target=np.array([0.22, 0, 0.06]), n=150):
    \"\"\"たたんだ姿勢から、target へ真下向きで S 字に動かす。着いたときのずれ [mm] を返す\"\"\"
    sc.reset()
    for a in (A, B):
        a.set_dofs_position(FOLD); a.control_dofs_position(FOLD)
    for _ in range(30):
        sc.step()
    start, Rd = tip(A, gA), Ry(math.pi / 2) @ R0A
    for k in range(n + 50):
        s = min(1.0, (k + 1) / (0.7 * n)); s = 3 * s ** 2 - 2 * s ** 3
        if k % 5 == 0:
            q = A.inverse_kinematics(link=gA, pos=start + (target - start) * s - Rd @ OFF, quat=quat_of(Rd),
                                     init_qpos=A.get_dofs_position(), max_solver_iters=50).cpu().numpy()
            q[5] = grip
            A.control_dofs_position(q)
        sc.step()
    return np.linalg.norm(tip(A, gA) - target) * 1000

goal.set_pos(np.array([0.22, 0, 0.06]))
for grip, label in [(CLOSE, "指を閉じて"), (OPEN, "指を開いたまま")]:
    e = from_fold(grip)
    print(f"{label:8s}  着いたときのずれ {e:5.1f} mm   腕のモーターの力 {A.get_dofs_control_force().cpu().numpy()[:4].round(2)} N·m")
c = A.get_contacts(with_entity=B)
print("その 2 と触れている部品の番号:", sorted({int(x) for x in c["link_a"].cpu().numpy().ravel()} | {int(x) for x in c["link_b"].cpu().numpy().ravel()}))
print("（その 1 の部品の番号:", [(l.idx, l.name) for l in A.links][-2:], " その 2:", [(l.idx, l.name) for l in B.links][-2:], "）")
show()
"""),
    md("""
## 📕 作者の失敗の記録：2 台が刺さっていた（2026-10-20）
- **何が起きたか**：H0 の確認の絵で、2 台の手先が真ん中で重なっていた（作者が気づいた）
- **気づいた数字**：エラーは出ない。シミュレーション 1 秒ぶんの計算が 1.2 秒かかっていた
- **切り分け**：組み立てた直後の姿勢を見ると、関節が全部 0（前腕を水平に伸ばす）。手先は根元から 22 cm 以上。2 台を 44 cm 離して向かい合わせると、真ん中で重なる
- **原因**：初めの姿勢を決めていなかった
- **対策**：組み立てた直後に、角度とモーターの目標の両方をたたんだ姿勢 `FOLD` にする（角度だけだと、次の step から伸びに行く）
- **学び**：計算が 1.2 秒 → 0.15 秒になった。**ぶつかり続けている物は、エラーを出さずに計算だけを重くする**。絵で確かめる

## 📕 作者の失敗の記録 2：条件を変えても、結果が同じだった（2026-10-20）
- **何が起きたか**：6 を最初に書いたとき（たたんだ姿勢から動かし、IK を解き直す間隔を 1・5・25 step に振った）、どれでも「着いたときのずれ 50.2 mm」で同じだった
- **気づいた数字**：肩・肘・手首のモーターの力が、3 つとも力の上限（±3.35 N·m）に張り付いていた。何かを押している
- **切り分け**：触れている部品を `get_contacts` で調べると、その 1 の動く側の指（`moving_jaw`）と、その 2 の動く側の指。その 2 を横へ向けても、その 1 の指を閉じても、ずれは 0.5 mm
- **原因**：指を開いたまま動かし、開いた指が、向かいのその 2 のたたんだ指に当たっていた
- **対策**：動かす間は指を閉じる（作品でも、運ぶ途中の指は閉じるか半分だけ開く。その 2 は離れた場所で待つ）
- **学び**：H1a の床と同じ形。**結果が条件によらず同じなら、何かに当たっている**。モーターの力が上限に張り付いていないかを見る
- **もう 1 つ**：たたんだ姿勢から動かすと、手先は真下を向いておらず低い所（1.2 cm）から始まる。動き出しで手首が大きく振れ、「下への行き過ぎ」が初めの高さを拾っていた。演習は、真下向きで箱の上に置いてから下ろす形に変えた

## 確かめる問い
1. IK に `pos=目標 − Rd @ OFF` を渡すのはなぜ？
2. その 1 では `Ry(90°) @ R0` で真下を向くのに、その 2 では向かないのはなぜ？
3. 作品で、運ぶ高さを 8 cm 以上にしないのはなぜ？

<details><summary>答え</summary>

1. IK が動かすのは手首の部品の原点で、目標に置きたいのは指先の目印。部品の向き Rd で見たずれ OFF の分だけ、手前に指示する
2. 行列を左から掛けると世界の軸のまわりに回る。その 1 は腕の横の軸が世界の y 軸と同じだが、180° 回したその 2 は、指が世界の −x を向いているので、同じ回し方では反対（上）へ倒れる
3. 根元から 20 cm で 8 cm 以上に上げると、手首が可動範囲の端に張り付いて、真下を向けきれない（✏️ 2 の表）
</details>
"""),
]

H1C = [
    md("""
# H1c 力のモーメント：握る力と重さ

**ねらい**：作品の芯の 2 つの量を、自分で測る。
- **握る力**：指のモーターの力の上限 τ [N·m] を決めると、指が箱を挟む力 F [N] が決まる（力のモーメント F × r ＝ τ）
- **重さ**：箱を持って止めると、肩のモーターが出す力が増える。その増え方から重さが出る（Δτ ＝ m g L）

そして、**持てる条件 2μF ≥ mg**（2 本の指の摩擦で、箱の重さを支える）を SO-101 で確かめる。
理論どおりになる所と、ならない所（関節の摩擦、掴みに行く動き）を見分ける。
"""),
    md("## 1. 準備"),
    code(SETUP),
    md("""
## 2. 道具と場面
H1b の回転の関数に加えて、`jaws()`：指が箱を押す力を、固定側の指・動く側の指に分けて読む（`get_contacts`：触れている所の一覧と力）。
挟む力は **2 本の小さいほう**（片側の指が触れただけでは、箱は逃げるので挟んでいない）。

箱は 3 cm 角、床の上、根元から 22 cm。重さは `set_mass` で変える。
"""),
    code("""
import math
OFF = np.array([-0.0079, -0.000218, -0.0981]); FOLD = np.array([0.0, -1.7, 1.6, 1.2, 0.0, 0.0])
OPEN, CLOSE, PRE, JAW = 1.2, -0.17, 0.5, 0.019    # 指の角度（開く／閉じきる／速く閉じる 1 段目の止め所）、目印 → 箱の中心
def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
def quat_of(R):
    t = np.trace(R)
    if t > 0:
        S = math.sqrt(t + 1.0) * 2; q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.array(q); return q / np.linalg.norm(q)
def Ry(a): return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

gs.init(backend=gs.cpu, logging_level="warning")
sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
              rigid_options=gs.options.RigidOptions(box_box_detection=True, friction_cone=gs.friction_cone.elliptic, impratio=10.0),
              show_viewer=False)
sc.add_entity(gs.morphs.Plane())
A = sc.add_entity(gs.morphs.MJCF(file=MODEL, decompose_robot_error_threshold=0.0), material=gs.materials.Rigid(friction=1.0))
RB = 0.22
box = sc.add_entity(gs.morphs.Box(size=(0.03,) * 3, pos=(RB, 0, 0.015)), material=gs.materials.Rigid(friction=1.0, rho=1000.0),
                    surface=gs.surfaces.Default(color=(0.72, 0.55, 0.36)))
cam = sc.add_camera(res=(768, 432), pos=(0.22, -0.30, 0.15), lookat=(0.20, 0.0, 0.03), fov=45, GUI=False)
t0 = time.time(); sc.build(); print(f"build {time.time() - t0:.0f} s")
g, upper = A.get_link("gripper"), A.get_link("upper_arm")
FIXED, MOVING = g.idx, A.get_link("moving_jaw_so101_v1").idx
A.set_dofs_position(np.zeros(6)); Rd = Ry(math.pi / 2) @ rot(g.get_quat().cpu().numpy())   # 真下向き（H1b）
FL0 = A.get_dofs_frictionloss().cpu().numpy()

def ik(p):
    return A.inverse_kinematics(link=g, pos=p - Rd @ OFF, quat=quat_of(Rd), init_qpos=A.get_dofs_position(), max_solver_iters=100).cpu().numpy()
def tip():
    return g.get_pos().cpu().numpy() + rot(g.get_quat().cpu().numpy()) @ OFF
def jaws():
    \"\"\"(固定側の指が押す力, 動く側の指が押す力) [N]\"\"\"
    c = A.get_contacts(with_entity=box)
    la, lb = c["link_a"].cpu().numpy(), c["link_b"].cpu().numpy()
    f = np.linalg.norm(c["force_a"].cpu().numpy(), axis=-1)
    if "valid_mask" in c: f = f * c["valid_mask"].cpu().numpy()      # 並列の環境のときだけ付く（空きの欄の印）
    return float((f * ((la == FIXED) | (lb == FIXED))).sum()), float((f * ((la == MOVING) | (lb == MOVING))).sum())
def squeeze():
    return min(jaws())
def tilt():
    q = box.get_quat().cpu().numpy(); return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))
def set_grip(tau):
    \"\"\"指のモーターの力の上限 = 握る力の設定 [N·m]（ほかの関節は 3.35）\"\"\"
    lim = np.full(6, 3.35); lim[5] = tau; A.set_dofs_force_range(-lim, lim)

cur = [None]
def go(p, grip, n, every=None):
    \"\"\"目印を p へ、真下向きのまま n step かけて S 字で動かす（H1b の move と同じ）。指の角度は grip。\"\"\"
    start = cur[0]
    for k in range(n):
        s = min(1.0, (k + 1) / (0.7 * n)); s = 3 * s * s - 2 * s ** 3
        if k % 5 == 0 or k == n - 1:
            q = ik(start + (p - start) * s); q[5] = grip; A.control_dofs_position(q)
        sc.step()
        if every: every()
    cur[0] = np.array(p, float)
def start(mass, frictionloss=None):
    \"\"\"初めの状態に戻し、箱の重さを mass [kg] に。指を開いて、箱の上 8 cm に置く\"\"\"
    sc.reset(); box.set_mass(mass); set_grip(3.35)
    fl = FL0.copy()
    if frictionloss is not None: fl[5] = frictionloss
    A.set_dofs_frictionloss(fl)
    cur[0] = np.array([RB - JAW, 0, 0.08]); q = ik(cur[0]); q[5] = OPEN
    A.set_dofs_position(q); A.control_dofs_position(q)
    for _ in range(30): sc.step()
def grasp(tau, mass, frictionloss=None):
    \"\"\"下ろして、指を 2 段階で閉じる（速く 1 段目まで → 力の上限を tau に絞って閉じきる。作品の close_gently）\"\"\"
    start(mass, frictionloss)
    go([RB - JAW, 0, 0.02], OPEN, 100)
    go(cur[0], PRE, 40)
    set_grip(tau); go(cur[0], CLOSE, 200)
def show():
    sc.visualizer.update(force=True); plt.figure(figsize=(8, 4.5)); plt.imshow(cam.render(rgb=True)[0]); plt.axis("off"); plt.show()

grasp(0.6, 0.15); print("0.6 N·m で 150 g を挟んだ：固定側 %.2f N、動く側 %.2f N" % jaws()); show()
"""),
    md("""
## 3. 握る力の設定 τ → 挟む力 F

指は「閉じきれ」（角度 −0.17）と指示されている。箱に当たって止まると、モーターは **力の上限 τ いっぱい** で押し続ける。
指の関節の軸から、箱に触れる所までの長さを r とすると、**F × r ＝ τ**（力のモーメント。てこと同じ）。

**予想**：τ を 2 倍にすると F は？ F ÷ τ はいくつ？
"""),
    code("""
TAUS = [0.3, 0.4, 0.6, 0.8]
F = []
for tau in TAUS:
    grasp(tau, 0.15); ff, fm = jaws(); F.append(min(ff, fm))
    print(f"τ {tau:.2f} N·m   固定側 {ff:5.2f} N   動く側 {fm:5.2f} N   挟む力 {min(ff, fm):5.2f} N   F ÷ τ {min(ff, fm) / tau:5.1f} N/(N·m)")
k = float(np.dot(TAUS, F) / np.dot(TAUS, TAUS))     # 原点を通る直線 F ＝ k τ の傾き
print(f"傾き {k:.1f} N/(N·m)  →  r ＝ 1 ÷ 傾き ＝ {100 / k:.1f} cm（指の関節の軸から、箱に触れる所まで）")
plt.figure(figsize=(5, 3.5)); plt.plot(TAUS, F, "o-"); plt.xlabel("握る力の設定 τ [N·m]"); plt.ylabel("挟む力 F [N]"); plt.show()
"""),
    md("""
**期待される出力**（手元の実測）：F ÷ τ は 13〜15、傾き 13.6 N/(N·m)、r ＝ 7.4 cm。固定側と動く側の力はほぼ同じ（箱が止まっているので、左右の力が釣り合う）。

**作品では**：この傾き（作品では 14.5 N/(N·m) と測った）を使って、「箱の強さ（段ボール 15 N）の 7 割を超えない τ」を決めている。

## ✏️ 変えてみる 1：弱い τ
**予想**：τ を 0.1、0.2 にすると、挟む力は？
"""),
    code("""
for tau in [0.1, 0.2]:
    grasp(tau, 0.15); ff, fm = jaws()
    print(f"τ {tau:.2f} N·m   固定側 {ff:5.2f} N   動く側 {fm:5.2f} N   挟む力 {min(ff, fm):5.2f} N")
"""),
    md("""
**期待される出力**：τ 0.1 では両方 0 N、τ 0.2 では動く側だけ 1.7 N（固定側は 0）。挟む力は、どちらも 0。

**なぜ**：指を開いて下ろした所では、固定側の指と箱のあいだに数 mm のすき間がある。動く側の指は、箱を床の上で押して滑らせ、固定側の指に当てる必要がある。
弱い τ では、床の摩擦と、指の関節自身の摩擦・減衰に負けて、2 秒では挟みきれない（4 で、掴めるかどうかに効く）。
"""),
    md("""
## 4. 持てる条件 2μF ≥ mg を確かめる
2 本の指がそれぞれ F で挟むと、摩擦で支えられるのは最大 μF × 2。これが箱の重さ mg 以上なら落ちない。
F ＝ 傾き × τ なので、**落ちない最小の τ ＝ m g ÷（2 μ × 傾き）**。μ は 1.0（指も箱も 1.0）。

確かめ方：まず強い力（0.6 N·m）で確実に掴んで 6 cm 持ち上げる。**持ったまま τ を 0.005 ずつ下げていき**、箱が落ちた τ を記録する。
（掴みに行く動きを混ぜないため。混ぜると 5 のように別の理由で決まってしまう）
"""),
    code("""
def drop_tau(mass, frictionloss=None, tau0=0.3, rate=0.005):
    grasp(0.6, mass, frictionloss); go([RB - JAW, 0, 0.06], CLOSE, 100); go(cur[0], CLOSE, 50)
    z0, tau = float(box.get_pos().cpu().numpy()[2]), tau0
    while tau > 0:
        set_grip(tau)
        for _ in range(20): sc.step()
        if float(box.get_pos().cpu().numpy()[2]) < z0 - 0.005:
            return tau
        tau = round(tau - rate, 4)
    return 0.0

def table(frictionloss=None):
    for m in [0.03, 0.15, 0.30]:
        theory = m * 9.81 / (2 * 1.0 * k)
        print(f"{m * 1000:4.0f} g   落ちた τ {drop_tau(m, frictionloss):.3f} N·m   理論 {theory:.3f} N·m")

print(f"指の関節の摩擦 {FL0[5]:.3f} N·m（モデルの値）"); table()
"""),
    md("""
**期待される出力**（手元の実測）
```
  30 g   落ちた τ 0.000 N·m   理論 0.011 N·m
 150 g   落ちた τ 0.040 N·m   理論 0.054 N·m
 300 g   落ちた τ 0.110 N·m   理論 0.108 N·m
```
300 g は理論とほぼ同じ。軽い箱ほど、理論より弱い力でも落ちない（30 g は 0 まで下げても落ちない）。

## ✏️ 変えてみる 2：指の関節の摩擦
モデルのファイルには、指の関節そのものの摩擦（`frictionloss`、0.052 N·m）が書いてある。モーターの力を抜いても、この分は指が開くのを止める。
**予想**：0 にすると、軽い箱の結果はどう変わる？
"""),
    code("""
print("指の関節の摩擦 0"); table(frictionloss=0.0)
"""),
    md("""
**期待される出力**：30 g 0.005、150 g 0.050、300 g 0.100 N·m。3 つとも理論（0.011 / 0.054 / 0.108）に近づく。

## 5. 掴みに行くと、もっと強い力が要る
今度は作品と同じく、**開いた指を下ろして、τ で閉じて、持ち上げる**。持ち上がった最小の τ を、4 の「落ちた τ」と比べる。
**予想**：4 と同じになる？
"""),
    code("""
def lift_ok(tau, mass):
    grasp(tau, mass); go([RB - JAW, 0, 0.04], CLOSE, 60); go(cur[0], CLOSE, 30)
    z = float(box.get_pos().cpu().numpy()[2])
    return "持てた" if z > 0.025 and tilt() < 10 else ("傾いた" if z > 0.025 else "  --  ")
TRY = [0.15, 0.2, 0.25, 0.3]
print("箱 \\\\ τ   " + "".join(f"{x:>8.2f}" for x in TRY))
for m in [0.03, 0.30]:
    print(f"{m * 1000:4.0f} g   " + "".join(f"{lift_ok(x, m):>6s}" for x in TRY))
"""),
    md("""
**期待される出力**：30 g は 0.20 から、300 g は 0.25 から持てる（300 g・0.30 は傾いてぶら下がる）。4 の落ちた τ（0.000 / 0.110）より、ずっと強い。

**作品では**：試し掴みを 0.2 N·m から始め、持ち上がった τ（力加減）に **余裕（2 倍）** を足して運ぶ。掴みに行くときと、運ぶ途中に揺れるときは、持っているあいだの理論より強い力が要るから。

## 6. 重さを測る：肩の力の増え方
腕を止めて箱を持つと、肩のモーター（`shoulder_lift`、1 番）は、腕の重さ ＋ 箱の重さを支える力を出す。
**同じ姿勢で**、持っているときと、何も持たないときの差 Δτ を取ると、箱の分だけが残る：**Δτ ＝ m g L**（L：肩の軸から箱までの水平の長さ）。
"""),
    code("""
def shoulder(mass, z=0.06, empty=False, moving=False):
    \"\"\"箱を持って（empty なら持たずに）高さ z で止め、肩の力の 1 秒平均と、L を返す。moving：測る間も上下に揺らす\"\"\"
    grasp(0.6, mass)
    if empty:
        box.set_pos(np.array([1.5, 1.5, 0.015]))           # 箱を遠くへ（何も持たない）
    go([RB - JAW, 0, z], CLOSE, 100)
    v = []
    for k_ in range(100):
        if moving:
            q = ik(np.array([RB - JAW, 0, z + 0.02 * math.sin(k_ / 25 * math.pi)])); q[5] = CLOSE; A.control_dofs_position(q)
        sc.step(); v.append(float(A.get_dofs_control_force()[1]))
    L = float(np.linalg.norm((box.get_pos().cpu().numpy()[:2] if not empty else tip()[:2] + np.array([JAW, 0])) - upper.get_pos().cpu().numpy()[:2]))
    return float(np.mean(v)), L

tau_empty, _ = shoulder(0.15, empty=True)
print(f"何も持たない   肩の力 {tau_empty:+.3f} N·m")
for m in [0.03, 0.10, 0.15, 0.30]:
    tau_h, L = shoulder(m)
    print(f"{m * 1000:4.0f} g を持つ   肩の力 {tau_h:+.3f} N·m   Δτ {tau_empty - tau_h:.3f}   L {L * 100:.1f} cm   測った重さ {(tau_empty - tau_h) / (9.81 * L) * 1000:5.0f} g")
"""),
    md("""
**期待される出力**（手元の実測）：何も持たない −0.345 N·m。測った重さは 30 → 30 g、100 → 99 g、150 → 143 g、300 → 288 g。L は 14.9 cm。

## ✏️ 変えてみる 3：基準を決め打ちにする
作品を作る途中では、何も持たないときの肩の力を、別の姿勢で測った決め打ちの値（−0.400 N·m）にしていた。
**予想**：その値を使うと、150 g は何 g と出る？
"""),
    code("""
tau_h, L = shoulder(0.15)
for label, base in [("同じ姿勢で測った基準", tau_empty), ("決め打ち −0.400", -0.400)]:
    print(f"{label:14s}   測った重さ {(base - tau_h) / (9.81 * L) * 1000:5.0f} g（本当は 150 g）")
"""),
    md("""
**期待される出力**：同じ姿勢の基準なら 143 g、決め打ちなら 105 g（45 g も軽く出る）。作品では、箱ごとに同じ姿勢で基準を測り直す形に直した。

## ✏️ 変えてみる 4：動かしながら測る
**予想**：測っている 1 秒のあいだ、上下に揺らすと？（平均と、ばらつき）
"""),
    code("""
for moving in [False, True]:
    vals = []
    for _ in range(1):
        tau_h, L = shoulder(0.15, moving=moving)
        vals.append((tau_empty - tau_h) / (9.81 * L) * 1000)
    grasp(0.6, 0.15); go([RB - JAW, 0, 0.06], CLOSE, 100); v = []
    for k_ in range(100):
        if moving:
            q = ik(np.array([RB - JAW, 0, 0.06 + 0.02 * math.sin(k_ / 25 * math.pi)])); q[5] = CLOSE; A.control_dofs_position(q)
        sc.step(); v.append(float(A.get_dofs_control_force()[1]))
    v = np.array(v)
    print(f"{'揺らしながら' if moving else '止めて':6s}   1 秒平均から {vals[0]:5.0f} g   0.1 秒ごとの値のばらつき {np.std([(tau_empty - x) / (9.81 * L) * 1000 for x in v.reshape(10, 10).mean(axis=1)]):5.0f} g")
"""),
    md("""
**期待される出力**：止めて測ると 140 g 前後で、0.1 秒ごとの値のばらつきは 10 g。揺らしながらだと平均は 161 g、ばらつきは 509 g（加速の分の力が混ざる）。作品では、持ち上げて止めてから 1 秒平均を取る。

## 7. 映像で確かめる
300 g を τ ＝ 0.25 で掴んで持ち上げる（8 step に 1 コマ）。**変えてみる**：`tau=` を 0.15 にして ▶ し直す。
"""),
    code("""
from IPython.display import HTML
import base64, imageio
def film(tau=0.25, mass=0.30, name="h1c.mp4"):
    frames = []
    grab = lambda: frames.append(cam.render(rgb=True)[0]) if len(sc_steps) % 8 == 0 else None
    sc_steps.clear()
    def every():
        sc_steps.append(1); grab()
    start(mass); go([RB - JAW, 0, 0.02], OPEN, 100, every); go(cur[0], PRE, 40, every)
    set_grip(tau); go(cur[0], CLOSE, 200, every); go([RB - JAW, 0, 0.06], CLOSE, 100, every)
    imageio.mimsave(name, frames, fps=12)
    return HTML(f'<video controls width=640 src="data:video/mp4;base64,{base64.b64encode(open(name, "rb").read()).decode()}">')
sc_steps = []
film()
"""),
    md("""
## 確かめる問い
1. 挟む力 F が τ に比例するのはなぜ？ 傾きの逆数は何の長さ？
2. 4 で、軽い箱は τ を 0 近くまで下げても落ちにくかった。なぜ？
3. 作品で、重さの基準（何も持たないときの肩の力）を、箱ごとに同じ姿勢で測り直しているのはなぜ？

<details><summary>答え</summary>

1. 指は閉じきる向きに指示され、力の上限 τ いっぱいで押す。力のモーメント F × r ＝ τ なので F ＝ τ ÷ r。傾きの逆数 r は、指の関節の軸から箱に触れる所までの長さ
2. 指の関節そのものの摩擦（0.052 N·m）が、指が開くのを止める。軽い箱が要る τ（0.01）はこれより小さい。摩擦を 0 にすると、理論どおりに落ちる
3. 腕の姿勢が変わると、腕自身の重さを支える肩の力も変わる。基準を別の姿勢で測ると、その差が箱の重さに混ざる（✏️ 3）
</details>
"""),
]

H2 = [
    md("""
# H2 掴む：形・目印・閉じ方・向き

**ねらい**：箱をうまく掴めるかどうかは、力の大きさの前に、次の 4 つで決まる。作品を作るときに 1 つずつ踏んだ。
| # | 決めたこと | 作品の値 | わざと失敗 |
|---|---|---|---|
| ① | ぶつかり判定のための、形の包み方 | 細かく分ける（0.0） | `convexify` |
| ② | 目印（指先）から箱の中心までのずれ | 1.9 cm | `jaw_center` |
| ③ | 指を速く閉じる 1 段目の止め所 | 0.5 | `pre_close` |
| ④ | 箱の向きに、指を合わせるか | 合わせる | `box_world` |

どれも、値を書き換えて違いを確かめる。
"""),
    md("## 1. 準備"),
    code(SETUP),
    md("""
## 2. 場面：形の包み方が違う SO-101 を 2 台
- **細**：形を細かい凸の部品に分けて包む（`decompose_robot_error_threshold=0.0`。作品）
- **粗**：Genesis の既定（部品ごとに 1 つの凸で包む）

2 台は 50 cm 離して、それぞれの前（根元から 22 cm）に箱を置く。組み立ては 1 回だけ。
"""),
    code("""
import math
OFF = np.array([-0.0079, -0.000218, -0.0981]); FOLD = np.array([0.0, -1.7, 1.6, 1.2, 0.0, 0.0])
OPEN, CLOSE = 1.2, -0.17
def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
def quat_of(R):
    t = np.trace(R)
    if t > 0:
        S = math.sqrt(t + 1.0) * 2; q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.array(q); return q / np.linalg.norm(q)
def Ry(a): return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
def Rz(a): return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

gs.init(backend=gs.cpu, logging_level="warning")
sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
              rigid_options=gs.options.RigidOptions(box_box_detection=True, friction_cone=gs.friction_cone.elliptic, impratio=10.0),
              show_viewer=False)
sc.add_entity(gs.morphs.Plane())
RB, Y2 = 0.22, 0.5
fine = sc.add_entity(gs.morphs.MJCF(file=MODEL, pos=(0, 0, 0), decompose_robot_error_threshold=0.0), material=gs.materials.Rigid(friction=1.0))
coarse = sc.add_entity(gs.morphs.MJCF(file=MODEL, pos=(0, Y2, 0)), material=gs.materials.Rigid(friction=1.0))
mat, look = gs.materials.Rigid(friction=1.0, rho=1000.0), gs.surfaces.Default(color=(0.72, 0.55, 0.36))
box_f = sc.add_entity(gs.morphs.Box(size=(0.03,) * 3, pos=(RB, 0, 0.015)), material=mat, surface=look)
box_c = sc.add_entity(gs.morphs.Box(size=(0.03,) * 3, pos=(RB, Y2, 0.015)), material=mat, surface=look)
cam = sc.add_camera(res=(768, 432), pos=(0.21, -0.30, 0.13), lookat=(0.21, 0.0, 0.04), fov=45, GUI=False)
t0 = time.time(); sc.build(); print(f"build {time.time() - t0:.0f} s")

class Hand:
    \"\"\"真下向きの手先で、自分の前の箱を掴む（作品の共通部品 Arm を、この演習用に小さくしたもの）\"\"\"
    def __init__(self, ent, base_y, box):
        self.e, self.g, self.box, self.y = ent, ent.get_link("gripper"), box, base_y
        self.fixed, self.moving = self.g.idx, ent.get_link("moving_jaw_so101_v1").idx
        ent.set_dofs_position(np.zeros(6)); self.R0 = rot(self.g.get_quat().cpu().numpy())
        self.dyaw, self.cur = 0.0, None
    def Rd(self): return Rz(self.dyaw) @ Ry(math.pi / 2) @ self.R0          # 指を dyaw だけ回して、真下へ
    def ik(self, p):
        R = self.Rd()
        return self.e.inverse_kinematics(link=self.g, pos=p - R @ OFF, quat=quat_of(R), init_qpos=self.e.get_dofs_position(),
                                         max_solver_iters=100).cpu().numpy()
    def tip(self): return self.g.get_pos().cpu().numpy() + rot(self.g.get_quat().cpu().numpy()) @ OFF
    def jaws(self):
        c = self.e.get_contacts(with_entity=self.box)
        la, lb = c["link_a"].cpu().numpy(), c["link_b"].cpu().numpy()
        f = np.linalg.norm(c["force_a"].cpu().numpy(), axis=-1)
        if "valid_mask" in c: f = f * c["valid_mask"].cpu().numpy()
        return float((f * ((la == self.fixed) | (lb == self.fixed))).sum()), float((f * ((la == self.moving) | (lb == self.moving))).sum())
    def set_grip(self, tau):
        lim = np.full(6, 3.35); lim[5] = tau; self.e.set_dofs_force_range(-lim, lim)
    def go(self, p, grip, n, every=None):
        start = self.cur
        for k in range(n):
            s = min(1.0, (k + 1) / (0.7 * n)); s = 3 * s * s - 2 * s ** 3
            if k % 5 == 0 or k == n - 1:
                q = self.ik(start + (np.asarray(p, float) - start) * s); q[5] = grip; self.e.control_dofs_position(q)
            sc.step()
            if every: every()
        self.cur = np.asarray(p, float)

H = {"細": Hand(fine, 0.0, box_f), "粗": Hand(coarse, Y2, box_c)}

def box_state(h):
    p = h.box.get_pos().cpu().numpy(); q = h.box.get_quat().cpu().numpy()
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))
    return p, tilt

def trial(h, tau=0.4, mass=0.15, jaw=0.019, pre=0.5, box_yaw=0.0, align=True, lift=0.04):
    \"\"\"h で、自分の前の箱を掴んで lift だけ持ち上げる。結果の dict を返す。
    jaw：目印から箱の中心までのずれ／pre：速く閉じる 1 段目の止め所／box_yaw：箱の向き [°]／align：指を箱の向きに合わせるか\"\"\"
    sc.reset()
    for x in H.values():
        x.e.set_dofs_position(FOLD); x.e.control_dofs_position(FOLD)
    h.box.set_mass(mass)
    h.box.set_quat(np.array([math.cos(math.radians(box_yaw) / 2), 0, 0, math.sin(math.radians(box_yaw) / 2)]))
    h.dyaw = ((math.radians(box_yaw) + math.pi / 4) % (math.pi / 2) - math.pi / 4) if align else 0.0   # 四角は 90° ごとに同じ → ±45° に
    c = np.array([RB, h.y, 0.0]) - jaw * np.array([math.cos(h.dyaw), math.sin(h.dyaw), 0])          # 目印は、指が閉じる向きに jaw だけ手前
    h.cur = np.array([c[0], c[1], 0.08]); q = h.ik(h.cur); q[5] = OPEN; h.e.set_dofs_position(q); h.e.control_dofs_position(q)
    h.set_grip(3.35)
    for _ in range(30): sc.step()
    peak = [0.0]
    def watch(): peak[0] = max(peak[0], min(h.jaws()))
    h.go([c[0], c[1], 0.02], OPEN, 100, watch)
    h.go(h.cur, pre, 40, watch)                       # 1 段目：力の上限を大きくしたまま、pre まで速く閉じる
    h.set_grip(tau); h.go(h.cur, CLOSE, 200, watch)    # 2 段目：力の上限を tau に絞って閉じきる
    sq = min(h.jaws())
    cc = h.e.get_contacts(with_entity=h.box)                                   # 指が箱に触れている所の高さ（閉じきったとき）
    contact_heights = sorted(round(float(z) * 100, 2) for z in cc["position"].cpu().numpy().reshape(-1, 3)[:, 2])
    grip_q = float(h.e.get_dofs_position()[5])
    h.go([c[0], c[1], 0.02 + lift], CLOSE, 80, watch); h.go(h.cur, CLOSE, 50, watch)
    p, tilt = box_state(h)
    trial.contacts = contact_heights
    return {"挟む力": sq, "閉じる間の最大": peak[0], "箱の高さ": p[2], "傾き": tilt, "上がった": p[2] > 0.015 + lift * 0.5, "指の角度": grip_q}

def show(h=None, title=""):
    sc.visualizer.update(force=True)
    if h is not None:
        cam.set_pose(pos=(0.21, h.y - 0.30, 0.13), lookat=(0.21, h.y, 0.04))
    plt.figure(figsize=(7, 4)); plt.imshow(cam.render(rgb=True)[0]); plt.axis("off"); plt.title(title); plt.show()

def say(label, r):
    print(f"{label:22s} 挟む力 {r['挟む力']:5.2f} N   閉じる間の最大 {r['閉じる間の最大']:5.2f} N   指の角度 {r['指の角度']:4.2f}   箱の高さ {r['箱の高さ'] * 100:4.1f} cm   傾き {r['傾き']:4.1f}°")
"""),
    md("""
## 3. ① 形の包み方
ぶつかり判定は、計算を速くするため、部品の形を **凸の形**（へこみのない形）で包んで行う。
SO-101 の固定側の指は、手首と 1 つの部品になっている。1 つの凸で包むと、**指と手首のあいだのすき間が埋まる**。

**予想**：粗く包んだ腕は、箱を掴める？
"""),
    code("""
for name in ["細", "粗"]:
    r = trial(H[name]); say(f"{name}く包む", r); print("    指が箱に触れている高さ [cm]：", trial.contacts, "（箱は 0〜3 cm）")
    show(H[name], "fine" if name == "細" else "coarse")
"""),
    md("""
**期待される出力**（手元の実測）
```
細く包む   挟む力 5.36 N   閉じる間の最大  7.73 N   指の角度 0.20   箱の高さ 5.5 cm
    指が箱に触れている高さ [cm]： [1.6, 2.12, 2.13, 3.0]
粗く包む   挟む力 5.36 N   閉じる間の最大 10.07 N   指の角度 0.30   箱の高さ 1.5 cm
    指が箱に触れている高さ [cm]： [2.98, 2.99]
```
**挟む力は同じなのに、粗く包んだ腕は持ち上がらない**。触れている所を見ると、箱の上の角（3 cm）だけ。埋まったすき間に邪魔されて指が下りきれず（指の角度 0.30）、角をつまんでいる。
挟む力の数字だけでは分からない。**触れている場所** を見る。

## 4. ② 目印から箱の中心までのずれ
目印 `gripperframe` は、**固定側の指の内側の先端**（指の真ん中ではない）。箱の中心は、目印から 1.9 cm 先（指が閉じる向き）。
**予想**：目印を箱の中心に置く（`jaw=0`）と？
"""),
    code("""
for jaw in [0.019, 0.0]:
    r = trial(H["細"], jaw=jaw); say(f"jaw = {jaw * 100:.1f} cm", r)
show(H["細"], "jaw = 0")
"""),
    md("""
## 5. ③ 指を閉じる 1 段目の止め所
力の上限を小さくすると、指はゆっくりしか閉じない（H1c）。そこで作品は 2 段階で閉じる：**① 力の上限を大きくしたまま、触れる手前（角度 0.5）まで速く閉じる → ② 力の上限を τ に絞って閉じきる**。
3 cm の箱に指が触れるのは、角度 0.20〜0.24。
**予想**：1 段目を 0.3、0.25（触れる直前）、0.2（触れた後）にすると、閉じる間の最大の力は？ 段ボールが潰れるのは 15 N。
"""),
    code("""
for pre in [0.5, 0.3, 0.25, 0.2]:
    r = trial(H["細"], pre=pre); say(f"1 段目 {pre}", r)
"""),
    md("""
**期待される出力**：最大の力は 0.5 で 7.7 N、0.3 で 8.3 N、0.25 で 13.7 N、0.2 で 66 N。触れる直前まで大きな力のまま閉じると、勢いで当たる。最後に挟む力（4.8〜5.4 N）はほとんど変わらないので、**閉じる途中を見ないと気づかない**。

## 6. ④ 箱の向きに指を合わせる
箱が 30° 回って置かれていたら、指も回して、箱の面を挟む。四角は 90° ごとに同じ形なので、回す角度は ±45° に収める。
**予想**：指を合わせないと（`align=False`）、どこを挟む？
"""),
    code("""
for align in [True, False]:
    r = trial(H["細"], box_yaw=30, align=align); say(f"30° の箱・指を{'合わせる' if align else '合わせない'}", r)
    show(H["細"], f"指を{'合わせる' if align else '合わせない'}")
"""),
    md("""
**期待される出力**：合わせると 5.5 cm まで上がる。合わせないと、指は箱の角に当たり、挟む力 0 で上がらない。
"""),
    md("""
## 確かめる問い
1. 粗く包んだ腕が掴めないのはなぜ？ 画面で何が見える？
2. 目印を箱の中心に置くと、何が起きる？
3. 指を閉じる 1 段目を、触れる手前の 0.5 で止めるのはなぜ？

<details><summary>答え</summary>

1. 固定側の指は手首と 1 つの部品。1 つの凸で包むと、指と手首のあいだのすき間が埋まり、指が下りきれない。箱の上の角だけをつまむので、持ち上げると抜ける（見た目の形は同じなので、画面ではすき間があるように見える）
2. 目印は固定側の指の先端なので、そこを箱の中心に置くと、固定側の指が箱の上に下りる
3. 触れる直前まで大きな力のまま閉じると、勢いで当たって、閉じる間の力が箱の強さ（段ボール 15 N）を超える。最後の挟む力は変わらないので、終わった後の値では気づけない
</details>
"""),
]

H3 = [
    md("""
# H3 力加減を探す：試し掴みと潰れの見張り

**ねらい**：作品のその 1 は、箱を **弱い力から少しずつ強めて試し掴み** し、持ち上がった最小の力（力加減）を見つける。
そのあいだ、指が箱を挟む力を見張り、**箱の強さ**（段ボール 15 N、発泡スチロール 6 N）を超えたら「潰れた」とする。
ここでは、
- 試す力の **並び**（弱い順／強めから／いきなり最大）
- 潰れの判定に使う力（**2 本の指の小さいほう** か、足したものか）
- 見つけた力加減のまま運ぶか、**余裕** を足すか

を書き換えて、作品の値の理由を確かめる。H1c の道具をそのまま使う。
"""),
    md("## 1. 準備"),
    code(SETUP),
    md("## 2. 道具と場面（H1c と同じ）"),
    H1C[4],
    md("""
## 3. 試し掴み
`search(重さ, 試す力の並び)`：試す力ごとに「下ろす → 2 段階で閉じる → 2 cm 持ち上げる」。持ち上がって傾きが 10° 未満なら、その力が **力加減**。
持ち上がらなければ、その場で指を開いて次の力へ。毎 step、挟む力（0.05 秒の平均）を見張り、箱の強さを超えたら「潰れた」で止める。
"""),
    code("""
GENTLE = [0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8]     # 壊れやすい箱：弱い順（作品）
QUICK = [0.65, 0.8, 1.2, 1.6]                       # 壊れにくい箱：強めから（作品）
STRONG = [3.35]                                     # いきなり最大（わざと失敗 strong_a）
CARDBOARD, FOAM = 15.0, 6.0                         # 箱の強さ：挟む力 [N] の上限

def search(mass, steps, limit=CARDBOARD, one_side=False):
    \"\"\"力加減を探す。(見つけた力, 試した回数, 挟む力の最大, 潰れた力) を返す。one_side：2 本の指の力を足して判定（わざと失敗）\"\"\"
    start(mass)
    st = {"peak": 0.0, "crushed": None, "recent": [], "tau": None}
    def watch():
        ff, fm = jaws()
        f = ff + fm if one_side else min(ff, fm)
        st["recent"] = (st["recent"] + [f])[-5:]            # 0.05 秒（5 step）の平均
        fm5 = float(np.mean(st["recent"])); st["peak"] = max(st["peak"], fm5)
        if fm5 > limit and st["crushed"] is None:
            st["crushed"] = st["tau"]
    found, tries = None, 0
    for tau in steps:
        st["tau"] = tau; set_grip(3.35)
        go([RB - JAW, 0, 0.08], OPEN, 60, watch); go([RB - JAW, 0, 0.02], OPEN, 80, watch)
        go(cur[0], PRE, 40, watch); set_grip(tau); go(cur[0], CLOSE, 200, watch)
        go([RB - JAW, 0, 0.04], CLOSE, 60, watch); go(cur[0], CLOSE, 30, watch)
        tries += 1
        if st["crushed"] is not None:
            break
        if float(box.get_pos().cpu().numpy()[2]) > 0.025 and tilt() < 10:
            found = tau; break
        set_grip(3.35); go(cur[0], OPEN, 50, watch)       # その場で開いてから、次の力で
    return found, tries, st["peak"], st["crushed"]

def report(label, steps, limit=CARDBOARD, one_side=False, masses=(0.03, 0.15, 0.30)):
    for m in masses:
        f, n, peak, crushed = search(m, steps, limit, one_side)
        res = f"潰れた（{crushed} N·m のとき）" if crushed is not None else (f"力加減 {f} N·m" if f else "持てない")
        print(f"{label:10s} {m * 1000:4.0f} g   {res:22s}   試した回数 {n}   挟む力の最大 {peak:5.1f} N")

report("弱い順", GENTLE)
"""),
    md("""
**期待される出力**（手元の実測）：30 g は 1 回目の 0.2 N·m、150 g と 300 g は 2 回目の 0.25 N·m。挟む力の最大は 3〜5 N で、段ボールの強さ 15 N よりずっと小さい。

## ✏️ 変えてみる 1：いきなり最大の力（わざと失敗 `strong_a`）
**予想**：試さずに、最大の力（3.35 N·m）で握ると？
"""),
    code("""
report("いきなり最大", STRONG)
"""),
    md("""
**期待される出力**：3 つとも 1 回目で「潰れた」。挟む力の最大は 78〜90 N（段ボールの強さ 15 N の 5 倍以上）。

## ✏️ 変えてみる 2：強めから試す
作品では、Gemini が「壊れにくい」と **自信をもって** 答えた箱（金属）だけ、強めの 0.65 N·m から試す（試す回数が減る）。
**予想**：段ボール（15 N）と、発泡スチロール（6 N）では、結果は同じ？
"""),
    code("""
report("強めから", QUICK, limit=CARDBOARD, masses=(0.15,))
report("強めから", QUICK, limit=FOAM, masses=(0.15,))
"""),
    md("""
**期待される出力**：段ボールは 1 回目（0.65 N·m）で持てる（挟む力 11 N）。発泡スチロールの強さ（6 N）では「潰れた」。
**作品では**：これが H5 のわざと失敗 `trust_ai`。Gemini が発泡スチロールを「石・壊れにくい」と見誤り、自信が中くらいなのに信じると、強めから試して潰す。

## 4. 潰れの判定に使う力（わざと失敗 `one_side`）
潰れるのは、両側から挟んだとき。片側の指が触れただけでは、箱は逃げる。だから判定には **2 本の小さいほう** を使う。
**予想**：2 本を足した力で判定すると、発泡スチロール（6 N）を弱い順に試したときは？
"""),
    code("""
report("小さいほう", GENTLE, limit=FOAM, masses=(0.15,))
report("足す", GENTLE, limit=FOAM, one_side=True, masses=(0.15,))
"""),
    md("""
**期待される出力**：小さいほうなら 0.25 N·m で持てる（最大 4 N）。足すと最大 9 N になり、**潰れていないのに「潰れた」と判定** する（判定の誤り）。

## 5. 見つけた力加減のまま運ぶ（わざと失敗 `hold_min`）
力加減は「2 cm を静かに持ち上げられた最小の力」。運ぶときは、手先が動いて揺れる。
**予想**：力加減のまま、左右に 3 cm 揺らしながら 4 秒持つと？ 2 倍の力では？
"""),
    code("""
def carry(mass, tau, seconds=4.0, record=None):
    grasp(tau, mass); go([RB - JAW, 0, 0.06], CLOSE, 100, record)
    for k_ in range(int(seconds * 100)):
        if k_ % 5 == 0:
            q = ik(np.array([RB - JAW + 0.03 * math.sin(k_ / 100 * 2 * math.pi), 0, 0.06])); q[5] = CLOSE; A.control_dofs_position(q)
        sc.step()
        if record: record()
    return float(box.get_pos().cpu().numpy()[2]), tilt()

for m in [0.15, 0.30]:
    f, *_ = search(m, GENTLE)
    for mult in [1.0, 2.0]:
        z, tl = carry(m, f * mult)
        print(f"{m * 1000:4.0f} g   力加減 {f} × {mult:.0f} ＝ {f * mult:.2f} N·m   4 秒後の箱の高さ {z * 100:4.1f} cm   傾き {tl:4.1f}°   {'運べた' if z > 0.04 else '落とした'}")
"""),
    md("""
**期待される出力**：力加減のまま（× 1）では、150 g も 300 g も落とす。2 倍なら運べる。
**作品では**：運ぶ力 ＝ 力加減 × 2。ただし、潰れの上限の 7 割を超えない（段ボールなら 0.7 × 15 N ÷ 14.5 ＝ 0.72 N·m）。余裕をどこまで削れるかは H4 で。

## 6. 映像で確かめる
300 g を力加減のまま揺らして運ぶ（8 step に 1 コマ）。**変えてみる**：`mult=2.0` にして ▶ し直す。
"""),
    code("""
from IPython.display import HTML
import base64, imageio
def film(mass=0.30, mult=1.0, name="h3.mp4"):
    f, *_ = search(mass, GENTLE)
    frames, n = [], [0]
    def rec():
        n[0] += 1
        if n[0] % 8 == 0: frames.append(cam.render(rgb=True)[0])
    carry(mass, f * mult, seconds=3.0, record=rec)
    imageio.mimsave(name, frames, fps=12)
    return HTML(f'<video controls width=640 src="data:video/mp4;base64,{base64.b64encode(open(name, "rb").read()).decode()}">')
film()
"""),
    md("""
## 確かめる問い
1. 弱い順に試すと回数は増える。それでも、作品が段ボールを弱い順に試すのはなぜ？
2. 潰れの判定に、2 本の指の「小さいほう」を使うのはなぜ？
3. 力加減に余裕を足すのはなぜ？ 足しすぎると何が起きる？

<details><summary>答え</summary>

1. 見た目では中身（重さ）が分からず、強めから試すと、壊れやすい箱を潰す。回数より、潰さないことを取る
2. 箱が潰れるのは両側から挟まれたとき。片側だけ触れているときは箱が逃げるので、足すと、潰れていないのに潰れたと判定する
3. 力加減は静かに持ち上げられた最小の力で、運ぶ途中の揺れには足りない。足しすぎると箱の強さを超えて潰すので、潰れの上限の 7 割で抑える
</details>
"""),
]

H4 = [
    md("""
# H4 台座で渡す：情報で持ち、余裕を足して運ぶ

**ねらい**：その 1 が測った **力加減** と **重さ** を、その 2 に渡す。その 2 は、
- **運ぶ力** ＝ 力加減 × 余裕（2 倍。ただし潰れの上限の 7 割まで）で、台座から持つ
- **運ぶ時間** ＝ 1.5 秒 ×（1 ＋ 重さ ÷ 150 g）で、重い箱ほどゆっくり大箱へ運ぶ

余裕と運ぶ時間を書き換えて、**運べた／落とした／潰した** の表を作る。情報を使わずに弱く握る（わざと失敗 `weak_b`）、運ぶ高さも確かめる。
（2 台目の手先の向き `world_tilt` は H1b で済み）

計算を軽くするため、**箱が台座に置かれたところから** 始める。その 1 が見つけた力加減は、H3 の値（300 g は 0.25 N·m）を受け取る。
"""),
    md("## 1. 準備"),
    code(SETUP),
    md("""
## 2. 場面：2 台、台座、大箱
その 1 は原点（たたんだまま）、その 2 は 44 cm 先に 180° 回して置く。台座（高さ 2 cm）は真ん中、大箱（内寸 14 cm、壁の高さ 4 cm）はその 2 から見て +90°。
"""),
    code("""
import math
OFF = np.array([-0.0079, -0.000218, -0.0981]); FOLD = np.array([0.0, -1.7, 1.6, 1.2, 0.0, 0.0])
OPEN, CLOSE, PRE, JAW, HALF = 1.2, -0.17, 0.5, 0.019, 0.6
def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
def quat_of(R):
    t = np.trace(R)
    if t > 0:
        S = math.sqrt(t + 1.0) * 2; q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.array(q); return q / np.linalg.norm(q)
def Ry(a): return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
def Rz(a): return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

gs.init(backend=gs.cpu, logging_level="warning")
sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
              rigid_options=gs.options.RigidOptions(box_box_detection=True, friction_cone=gs.friction_cone.elliptic, impratio=10.0),
              show_viewer=False)
sc.add_entity(gs.morphs.Plane())
D, STAND_H = 0.44, 0.02
entA = sc.add_entity(gs.morphs.MJCF(file=MODEL, pos=(0, 0, 0), decompose_robot_error_threshold=0.0), material=gs.materials.Rigid(friction=1.0))
entB = sc.add_entity(gs.morphs.MJCF(file=MODEL, pos=(D, 0, 0), euler=(0, 0, 180), decompose_robot_error_threshold=0.0), material=gs.materials.Rigid(friction=1.0))
sc.add_entity(gs.morphs.Box(size=(0.05, 0.05, STAND_H), pos=(D / 2, 0, STAND_H / 2), fixed=True), surface=gs.surfaces.Default(color=(0.6, 0.6, 0.65)))
BX, BY = D, -0.22                                    # 大箱の中心（その 2 から見て +90°、22 cm）
for dx, dy, sx, sy in [(0.07, 0, 0.004, 0.144), (-0.07, 0, 0.004, 0.144), (0, 0.07, 0.144, 0.004), (0, -0.07, 0.144, 0.004)]:
    sc.add_entity(gs.morphs.Box(size=(sx, sy, 0.04), pos=(BX + dx, BY + dy, 0.02), fixed=True), surface=gs.surfaces.Default(color=(0.75, 0.6, 0.4)))
box = sc.add_entity(gs.morphs.Box(size=(0.03,) * 3, pos=(D / 2, 0, STAND_H + 0.015)), material=gs.materials.Rigid(friction=1.0, rho=1000.0),
                    surface=gs.surfaces.Default(color=(0.72, 0.55, 0.36)))
cam = sc.add_camera(res=(768, 432), pos=(0.30, -0.55, 0.30), lookat=(0.32, -0.10, 0.03), fov=50, GUI=False)
t0 = time.time(); sc.build(); print(f"build {time.time() - t0:.0f} s")

class Arm:
    \"\"\"根元から見た (方位 az, 半径 r, 高さ z) へ、真下向きの手先を S 字で動かす（作品の共通部品 so101.Arm を小さくしたもの）\"\"\"
    def __init__(self, ent, base, yaw_deg):
        self.e, self.g, self.base, self.yaw = ent, ent.get_link("gripper"), np.array(base, float), math.radians(yaw_deg)
        self.fixed, self.moving = self.g.idx, ent.get_link("moving_jaw_so101_v1").idx
        ent.set_dofs_position(np.zeros(6)); self.R0 = rot(self.g.get_quat().cpu().numpy())
        self.last = None
    def world(self, az, r, z):
        a = self.yaw + az; return np.array([self.base[0] + r * math.cos(a), self.base[1] + r * math.sin(a), z])
    def Rd(self, az):
        return Rz(az) @ Rz(self.yaw) @ Ry(math.pi / 2) @ Rz(-self.yaw) @ self.R0       # 腕自身の向きで真下へ（H1b）
    def tip(self): return self.g.get_pos().cpu().numpy() + rot(self.g.get_quat().cpu().numpy()) @ OFF
    def set_grip(self, tau):
        lim = np.full(6, 3.35); lim[5] = tau; self.e.set_dofs_force_range(-lim, lim)
    def jaws(self):
        c = self.e.get_contacts(with_entity=box)
        la, lb = c["link_a"].cpu().numpy(), c["link_b"].cpu().numpy()
        f = np.linalg.norm(c["force_a"].cpu().numpy(), axis=-1)
        if "valid_mask" in c: f = f * c["valid_mask"].cpu().numpy()
        return float((f * ((la == self.fixed) | (lb == self.fixed))).sum()), float((f * ((la == self.moving) | (lb == self.moving))).sum())
    def move(self, az, r, z, grip, n, every=None):
        tgt = self.world(az, r, z); start = self.last if self.last is not None else self.tip()
        for k in range(n):
            s = min(1.0, (k + 1) / (0.7 * n)); s = 3 * s * s - 2 * s ** 3
            if k % 5 == 0 or k == n - 1:
                R = self.Rd(az); cur = start + (tgt - start) * s
                q = self.e.inverse_kinematics(link=self.g, pos=cur - R @ OFF, quat=quat_of(R), init_qpos=self.e.get_dofs_position(),
                                              max_solver_iters=50).cpu().numpy()
                q[5] = grip; self.e.control_dofs_position(q)
            sc.step()
            if every: every()
        self.last = tgt
A, B = Arm(entA, (0, 0), 0), Arm(entB, (D, 0), 180)
# その 1 は、台座に置いた後、横（方位 −1.2 rad）へ退いた形でたたむ（作品と同じ）。正面でたたむと、指先が台座のすぐ手前にあり、
# 下りてくるその 2 の開いた指とぶつかる（H1b の 📕 2 と同じ）
FOLD_A = FOLD.copy(); FOLD_A[0] = -1.2

def show():
    sc.visualizer.update(force=True); plt.figure(figsize=(8, 4.5)); plt.imshow(cam.render(rgb=True)[0]); plt.axis("off"); plt.show()

def handover(mass, tau_B, carry_s, z_carry=0.07, crush=15.0, every=None):
    \"\"\"台座の上の箱を、その 2 が tau_B で持ち、carry_s 秒で大箱へ運ぶ。結果の dict を返す。\"\"\"
    sc.reset()
    for arm, fold in ((A, FOLD_A), (B, FOLD)):
        arm.e.set_dofs_position(fold); arm.e.control_dofs_position(fold); arm.last = None; arm.set_grip(3.35)
    box.set_mass(mass); box.set_pos(np.array([D / 2, 0, STAND_H + 0.015])); box.set_quat(np.array([1.0, 0, 0, 0]))
    for _ in range(30): sc.step()
    st = {"peak": 0.0, "recent": [], "crushed": None, "phase": ""}
    def watch():
        st["recent"] = (st["recent"] + [min(B.jaws())])[-5:]; f5 = float(np.mean(st["recent"]))
        st["peak"] = max(st["peak"], f5)
        if f5 > crush and st["crushed"] is None: st["crushed"] = st["phase"]
        if every: every()
    r_b, zs = D / 2 - JAW, STAND_H + 0.020
    st["phase"] = "近づく"
    B.move(0.0, 0.26, 0.20, CLOSE, 120, watch)          # 高い所で手先を真下に向けてから
    B.move(0.0, r_b, 0.15, HALF, 100, watch)
    B.move(0.0, r_b, 0.15, OPEN, 40, watch)
    B.move(0.0, r_b, zs, OPEN, 120, watch)
    st["phase"] = "持つ"
    B.move(0.0, r_b, zs, PRE, 40, watch); B.set_grip(tau_B); B.move(0.0, r_b, zs, CLOSE, 200, watch)
    B.move(0.0, r_b, z_carry, CLOSE, 100, watch)
    held = float(box.get_pos().cpu().numpy()[2]) > z_carry - 0.03
    st["phase"] = "運ぶ"
    rel = np.array([BX, BY]) - B.base
    az_d = (math.atan2(rel[1], rel[0]) - B.yaw + math.pi) % (2 * math.pi) - math.pi
    B.move(az_d, float(np.linalg.norm(rel)) - JAW, z_carry, CLOSE, int(carry_s * 100), watch)
    B.move(az_d, float(np.linalg.norm(rel)) - JAW, z_carry, OPEN, 60, watch)
    B.move(az_d, float(np.linalg.norm(rel)) - JAW, 0.15, OPEN, 60, watch)
    p = box.get_pos().cpu().numpy()
    inside = abs(p[0] - BX) < 0.068 and abs(p[1] - BY) < 0.068 and p[2] < 0.06
    res = f"潰した（{st['crushed']}）" if st["crushed"] else ("運べた" if inside else ("持てない" if not held else "落とした"))
    return {"結果": res, "挟む力の最大": st["peak"], "箱": p}

FOUND, MASS = 0.25, 0.30                                    # その 1 から受け取る情報（H3 の 300 g）
def carry_tau(found, margin=2.0, crush=15.0, k=14.5):
    \"\"\"運ぶ力 ＝ 力加減 × 余裕。ただし潰れの上限の 7 割（0.7 × crush ÷ k）を超えない\"\"\"
    return min(found * margin, max(0.7 * crush / k, found))
def carry_time(mass):
    return 1.5 * (1 + mass / 0.15)

r = handover(MASS, carry_tau(FOUND), carry_time(MASS))
print(f"運ぶ力 {carry_tau(FOUND):.2f} N·m   運ぶ時間 {carry_time(MASS):.1f} s   →   {r['結果']}   挟む力の最大 {r['挟む力の最大']:.1f} N")
show()
"""),
    md("""
**期待される出力**：運ぶ力 0.50 N·m、運ぶ時間 4.5 s で「運べた」（挟む力の最大 8 N）。絵で、箱が大箱の中にある。

## ✏️ 変えてみる 1：情報を使わずに弱く握る（わざと失敗 `weak_b`）
**予想**：その 2 が、その 1 の情報を使わずに 0.2 N·m で握ると？
"""),
    code("""
r = handover(MASS, 0.2, carry_time(MASS))
print(f"運ぶ力 0.20 N·m   →   {r['結果']}"); show()
"""),
    md("""
**期待される出力**：「持てない」。手だけが上がり、箱は台座に残る。

## ✏️ 変えてみる 2：余裕 × 運ぶ時間
**予想**：余裕を 1.0 倍（力加減のまま）・1.25 倍・2 倍、運ぶ時間を 0.6 秒（速い）・4.5 秒（重さに合わせる）にすると、6 通りの結果は？
1 通り 1〜2 分かかる（Colab）。
"""),
    code("""
print("余裕 \\\\ 運ぶ時間       0.6 s            4.5 s")
for margin in [1.0, 1.25, 2.0]:
    row = []
    for secs in [0.6, carry_time(MASS)]:
        r = handover(MASS, carry_tau(FOUND, margin), secs)
        row.append(f"{r['結果']}（{r['挟む力の最大']:.0f} N）")
    print(f"{margin:4.2f} 倍（{carry_tau(FOUND, margin):.2f} N·m）   " + "   ".join(f"{c:16s}" for c in row))
"""),
    md("""
**期待される出力**（手元の実測）
```
1.00 倍（0.25 N·m）   落とした（4 N）   落とした（4 N）
1.25 倍（0.31 N·m）   落とした（5 N）   落とした（6 N）
2.00 倍（0.50 N·m）   運べた（11 N）    運べた（8 N）
```
この場面では、**運ぶ時間より余裕が効く**。速く運ぶと、挟む力の最大が 8 N → 11 N に上がる（振られる分、箱が指に押し付けられる）。
作品の通し（5 箱、`line.py`）では、その 1 が見つけた 300 g の力加減が 0.40 N·m で、余裕 1.0 倍でもゆっくりなら運べ、0.6 秒で運ぶと大箱の手前で落とした（動画の場面 4 の失敗）。**力加減の値しだいで、どちらが効くかが変わる**。

## ✏️ 変えてみる 3：運ぶ高さ
**予想**：運ぶ高さを 7 cm から 12 cm にすると？（H1b の届く範囲の表を思い出す）
"""),
    code("""
for z in [0.07, 0.12]:
    r = handover(MASS, carry_tau(FOUND), carry_time(MASS), z_carry=z)
    print(f"運ぶ高さ {z * 100:.0f} cm   →   {r['結果']}   箱 {r['箱'].round(3)}")
"""),
    md("""
**期待される出力**：12 cm でも運べたが、落ちた所は大箱の中心から約 4 cm ずれた（7 cm では 0.3 cm）。

## 📕 作者の失敗の記録：その 2 が箱の上で止まった（2026-10-22）
- **何が起きたか**：この教材を書いたとき、どの条件でも「持てない」（挟む力 0）だった
- **気づいた数字**：その 2 の手先が、目標の高さ 4 cm に対して 5.9 cm で止まっていた
- **切り分け**：その 1 を正面でたたむと、指先が台座のすぐ手前（根元から 16 cm、床から 1.3 cm）にある。その 1 を横へ向けてたたむと、運べた
- **原因**：下りてくるその 2 の開いた指が、たたんだその 1 の指に当たっていた（H1b の 📕 2 と同じ形）
- **対策**：その 1 は、台座に置いた後、横（方位 −1.2 rad）へ退く（作品と同じ）
- **学び**：**2 台が同じ場所に手を伸ばす場面では、待っている側の姿勢も決める**。作品で、その 2 を台座の真上ではなく +60° で待たせたのも同じ理由（その 1 の腕とぶつかり、重さを 1606 g と読んだ）

## 確かめる問い
1. 余裕を足さずに運ぶと落とすのはなぜ？
2. 余裕を足しすぎると、何が起きる？ 作品はどう防いでいる？
3. その 2 がその 1 の情報（力加減）を使わないと、どうなる？

<details><summary>答え</summary>

1. 力加減は、2 cm を静かに持ち上げられた最小の力。運ぶ途中は、加速・減速で箱が振られ、その分の力が足りない
2. 挟む力が箱の強さを超えて潰す（段ボール 15 N）。作品は、運ぶ力を潰れの上限の 7 割（0.72 N·m）で抑える
3. 弱く握ると重い箱が持ち上がらない。強く握ると壊れやすい箱を潰す。ちょうどよい力は、持ってみないと分からない
</details>
"""),
]

BOXES_PNG_B64 = "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"
BOXES_SAVED = '```json\n[\n  {\n    "point": [481, 180],\n    "looks_like": "cardboard",\n    "weight_g": [2, 50],\n    "fragile": true,\n    "confidence": "low"\n  },\n  {\n    "point": [481, 338],\n    "looks_like": "stone",\n    "weight_g": [50, 100],\n    "fragile": false,\n    "confidence": "medium"\n  },\n  {\n    "point": [481, 500],\n    "looks_like": "cardboard",\n    "weight_g": [2, 50],\n    "fragile": true,\n    "confidence": "low"\n  },\n  {\n    "point": [481, 661],\n    "looks_like": "metal",\n    "weight_g": [150, 250],\n    "fragile": false,\n    "confidence": "high"\n  },\n  {\n    "point": [481, 820],\n    "looks_like": "cardboard",\n    "weight_g": [2, 50],\n    "fragile": true,\n    "confidence": "low"\n  }\n]\n```'

H5 = [
    md("""
# H5 見た目で予想する：Gemini（VLM）

**ねらい**：作品の 0 番目の工程。カメラの写真を **VLM**（画像と言葉を両方扱う AI。ここでは Gemini Robotics-ER）に見せ、箱ごとに
**材質・重さ・壊れやすさ・自信** を JSON で答えさせる。その答えで、その 1 の **試し掴みの探し方**（H3：弱い順／強めから）を選ぶ。
- 指示文を書き換えると、答えがどう変わるか
- **自信** を使わずに答えを信じると、何が起きるか（わざと失敗 `trust_ai`）

このノートは Genesis を使わないので、すぐ動く。
"""),
    md("""
## 1. 準備
Gemini のキーは、Colab の左の 🔑（シークレット）に、名前 `GEMINI_API_KEY` で入れて、「ノートブックからのアクセス」をオンにする。
**キーがなくても進められる**（作品で保存した答えを使う）。
"""),
    code("""
import json, base64, io, os
if "google.colab" in str(get_ipython()):
    !pip -q install google-genai==2.29.0
import numpy as np, matplotlib.pyplot as plt
try:
    from google.colab import userdata
    KEY = userdata.get("GEMINI_API_KEY")
except Exception:
    KEY = os.environ.get("GEMINI_API_KEY")
print("キー：", "あり（その場で Gemini を呼ぶ）" if KEY else "なし（保存した答えを使う）")
"""),
    md("""
## 2. 写真を見る
作品のシミュレータで撮った写真。左から、段ボール（30 g）・発泡スチロール（10 g）・段ボール（150 g）・金属（400 g）・段ボール（300 g）。
**段ボールの 3 つは、見た目がまったく同じ**（中身の重さだけ違う）。
"""),
    code("""
PNG_B64 = \"""" + "__PNG__" + """\"
open("boxes.png", "wb").write(base64.b64decode(PNG_B64))
from PIL import Image
img = Image.open("boxes.png"); print(img.size)
plt.figure(figsize=(10, 4)); plt.imshow(img); plt.axis("off"); plt.show()
TRUTH = [("段ボール", 30, True), ("発泡スチロール", 10, True), ("段ボール", 150, True), ("金属", 400, False), ("段ボール", 300, True)]   # 左から（材質, 重さ g, 壊れやすい）
"""),
    md("""
## 3. Gemini に聞く
指示文（英語）は作品と同じ。`point` は箱の中心の位置（0〜1000 に正規化した [y, x]）。
"""),
    code("""
PROMPT = \"\"\"This image shows small boxes (each about 3 cm) on a table, seen by a robot that must pick them up.
For EVERY box, return one JSON object:
- "point": [y, x] of the box center, normalized to 0-1000
- "looks_like": what the box looks like it is made of
- "weight_g": your best guess of its total weight in grams (contents included), as [low, high]
- "fragile": true if squeezing it hard could crush or damage it
- "confidence": "low", "medium" or "high" for the weight guess. If boxes look identical, say so and give a wide range.
Return only a JSON array, no prose.\"\"\"
SAVED = """ + "__SAVED__" + """

def ask(prompt=PROMPT, model="gemini-robotics-er-2-preview"):
    \"\"\"写真と指示文を Gemini に渡し、答えの文字列を返す。キーがなければ、作品で保存した答え\"\"\"
    if not KEY:
        return SAVED
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=KEY, http_options=types.HttpOptions(timeout=120_000, retry_options=types.HttpRetryOptions(attempts=1)))
    f = client.files.upload(file="boxes.png")
    resp = client.interactions.create(model=model,
                                      input=[{"type": "image", "uri": f.uri, "mime_type": f.mime_type}, {"type": "text", "text": prompt}],
                                      generation_config={"thinking_level": "medium"})
    return resp.output_text

def parse(text):
    body = text.split("```json")[-1].split("```")[0] if "```" in text else text
    return sorted(json.loads(body), key=lambda a: a["point"][1])      # 左から順に

def compare(answers):
    print(f"{'#':2s} {'本当':16s} {'Gemini の答え':40s}")
    for i, (a, (mat, g, frag)) in enumerate(zip(answers, TRUTH)):
        print(f"{i:2d} {mat}・{g} g・{'壊れやすい' if frag else '壊れにくい':5s}   {a['looks_like']}・{a['weight_g']} g・fragile={a['fragile']}・自信 {a['confidence']}")

text = ask(); print(text[:400], "..."); answers = parse(text); compare(answers)
"""),
    md("""
**期待される出力**（作品で保存した答え）
- 段ボールの 3 つは、同じ答え（`cardboard`、2〜50 g、壊れやすい、自信 low）。**中身の重さは見た目では分からない** と自分で言っている
- 金属は `metal`、150〜250 g、自信 high（本当は 400 g。半分近くに見積もった）
- 発泡スチロールを **`stone`（石）、壊れにくい、自信 medium** と見誤る

キーがある人は、その場で呼んだ答えが、保存した答えと同じとは限らない（VLM の答えは毎回少し揺れる）。
**2026-10-22 に手元でその場で呼んだら、発泡スチロールを「石・壊れにくい・自信 high」と答えた**（保存した答えは medium）。4 で、これが何を意味するかを見る。

## ✏️ 変えてみる 1：指示文（キーがある人）
**予想**：最後の「If boxes look identical, say so and give a wide range.」を消すと、段ボールの 3 つの答えはどうなる？
"""),
    code("""
if KEY:
    p2 = PROMPT.replace(" If boxes look identical, say so and give a wide range.", "")
    compare(parse(ask(p2)))
else:
    print("キーがないので飛ばす")
"""),
    md("""
**期待される出力**（2026-10-22 に手元で 1 回呼んだ結果）：段ボールの 3 つが「5〜25 g・自信 medium」になった（消す前は 2〜100 g・low）。
「見た目が同じなら幅を広く」と言わないと、**根拠がないのに、幅を狭くして自信を上げる**。指示文の 1 文が、答えの慎重さを決めている。

## 4. 答えで探し方を選ぶ
作品の決まり：**「壊れにくい」かつ「自信 high」の箱だけ、強めから試す**（QUICK）。それ以外は弱い順（GENTLE）。
H3 で確かめたとおり、強めから試すと、挟む力は 11 N。段ボール（15 N）は無事でも、**発泡スチロール（6 N）は潰れる**。
"""),
    code("""
STRENGTH = {"段ボール": 15.0, "発泡スチロール": 6.0, "金属": float("inf")}   # 箱の強さ（挟む力の上限 N）
PEAK = {"GENTLE": 4.3, "QUICK": 10.9}                                         # H3 で測った、挟む力の最大 [N]

def choose(a, trust_ai=False):
    sturdy = (not a.get("fragile", True)) and (a.get("confidence") == "high" or trust_ai)
    return "QUICK" if sturdy else "GENTLE"

def plan(answers, trust_ai=False, title=None):
    print(title or ("わざと失敗 trust_ai（自信によらず信じる）" if trust_ai else "作品の決まり（自信 high のときだけ信じる）"))
    for a, (mat, g, _) in zip(answers, TRUTH):
        way = choose(a, trust_ai)
        crushed = PEAK[way] > STRENGTH[mat]
        print(f"   {mat:8s} {g:4d} g   答え {a['looks_like']:10s}（自信 {a['confidence']:6s}） → {way:6s}   挟む力の最大 {PEAK[way]:4.1f} N   {'潰す' if crushed else '無事'}")

answers = parse(SAVED)          # ここからは、保存した答えで比べる（毎回同じ結果になるように）
plan(answers)
plan(answers, trust_ai=True)
"""),
    md("""
**期待される出力**：作品の決まりでは、金属だけ QUICK、ほかは GENTLE で、5 つとも無事。
`trust_ai` では、発泡スチロール（`stone`、自信 medium）も QUICK になり、**潰す**（動画の場面 5 の失敗）。

**注意**：その場で呼んで「石・自信 high」と返った答え（3 の注）を使うと、**作品の決まりでも潰す**。自信で守る決まりは、答えの揺れで破られることがある。
キーがある人は、`answers = parse(text)`（その場で呼んだ答え）にして `plan(answers)` を ▶ し直して確かめる。

## ✏️ 変えてみる 2：信じる自信の境目
**予想**：`"high"` だけでなく `"medium"` も信じる決まりにすると？ 逆に、誰も信じない（全部 GENTLE）と、何を失う？
"""),
    code("""
def choose(a, trust_ai=False):
    sturdy = (not a.get("fragile", True)) and (a.get("confidence") in ("high", "medium") or trust_ai)   # ← ここを書き換える
    return "QUICK" if sturdy else "GENTLE"
plan(answers, title="書き換えた決まり")
"""),
    md("""
**期待される出力**：medium も信じると、`trust_ai` と同じく発泡スチロールを潰す。全部 GENTLE なら潰さないが、金属（400 g）も 0.2 から順に試すので、試す回数が増える（作品の通しでは、金属は QUICK の 1 回目で持てた）。

## 確かめる問い
1. 段ボールの 3 つで、Gemini が同じ答えを返したのはなぜ？ 作品はそれをどう補っている？
2. 自信が medium の答えを信じないのはなぜ？
3. VLM の答えをロボットの動きに使うとき、気をつけることは？

<details><summary>答え</summary>

1. 見た目が同じで、中身の重さは写真に写らない。作品は、持って確かめる（試し掴みで力加減、肩の力で重さ）
2. 見誤り（発泡スチロールを石）は、自信が中くらいの答えに出た。信じて強めから試すと、壊れやすい箱を潰す。外れたときに取り返しがつかない判断にだけ、高い自信を求める
3. 答えは毎回揺れ、見誤ったまま自信 high と言うこともある（2026-10-22 の発泡スチロール）。自信の決まりだけに頼らず、強めから試すときも、挟む力の見張り（潰れの手前で止める）と、持って確かめる手順を組み合わせる
</details>
"""),
]
for _c in H5:   # 写真と保存した答えを、目印の文字列の所に埋め込む
    _c["source"] = _c["source"].replace("__PNG__", BOXES_PNG_B64).replace("__SAVED__", repr(BOXES_SAVED))

if __name__ == "__main__":
    write("H0_check.ipynb", H0)
    write("H1a_body_and_commands.ipynb", H1A)
    write("H1b_hand_commands.ipynb", H1B)
    write("H1c_moments.ipynb", H1C)
    write("H2_grasp.ipynb", H2)
    write("H3_force_search.ipynb", H3)
    write("H4_handover.ipynb", H4)
    write("H5_gemini.ipynb", H5)
