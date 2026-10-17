"""L1: 指 2 本で持てるかは「2 μ F >= m g」で決まる。シミュレータの数字と理論を 24 マス突き合わせる。

部品はアルミの 4 cm 角（173 g、重さ m g = 1.70 N）。μ を 6 通り × 把持力 F を 4 通り = 24 の環境を並列に回す。
使い方: python lesson/hold.py [--mistake finger_mu | substeps1]
  --mistake finger_mu : 指の μ を 0.01 にしない（Genesis の既定 1.0 のまま）
  --mistake substeps1 : 1 ステップの中の細かい計算（substeps）を 10 → 1 にする
  --mistake slam      : 指をそっと閉じず、いきなり目標の力で閉じる
  --mistake slam_substeps1 : いきなり閉じる ＋ substeps=1
  --mistake jump      : 持ち上げの目標を一度に高さ 30 cm（いまの位置から 16.5 cm 上）に置く（滑らかに動かさない）
  --record out.mp4    : 24 の環境を真横から録画し、表と同じ並び（行 = μ、列 = F）の 1 本にまとめる
  --viewer [--env 5]  : 1 つの環境（番号 = 行 × 4 + 列。既定 5 = μ 0.10・F 5 N）を Genesis の窓で見て、
                        24 の環境の部品の高さと、指に対する部品のずれ（滑り）を rerun へ送る
"""
import argparse

import numpy as np

import common as c

MUS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
FORCES = [2.0, 5.0, 10.0, 20.0]

ap = argparse.ArgumentParser()
ap.add_argument("--mistake", choices=["finger_mu", "substeps1", "jump", "slam", "slam_substeps1"])
ap.add_argument("--record", help="mp4 の保存先（例: outputs/L1.mp4）")
ap.add_argument("--viewer", action="store_true", help="Genesis ビューアと rerun で見る")
ap.add_argument("--env", type=int, default=5, help="窓で見る環境の番号（行 × 4 + 列。0 = μ 0.05・2 N、23 = μ 0.8・20 N）")
args = ap.parse_args()
c.quiet()

grid = [(mu, f) for mu in MUS for f in FORCES]
N = len(grid)
mats = ["aluminium"] * N
true_mu = np.array([mu for mu, _ in grid])
force = np.array([f for _, f in grid])
XY = np.array([0.55, 0.0])
view = c.viewer_view(env=args.env, pos=(0.55, -1.0, 0.35), lookat=(0.55, 0.0, 0.2)) if args.viewer else None
rr = c.rerun_start("L1_hold") if args.viewer else None
if view:
    mu_v, f_v = grid[args.env]
    print(f"viewer: 窓に描くのは環境 {args.env}（mu = {mu_v:.2f}、F = {f_v:.0f} N）")

scene = c.make_scene(substeps=1 if args.mistake in ("substeps1", "slam_substeps1") else 10, record=bool(args.record),
                     viewer=view)
scene.add_entity(c.gs.morphs.Plane())
part = c.add_part(scene, XY)
arm = c.add_franka(scene, finger_mu=1.0 if args.mistake == "finger_mu" else 0.01)
cam = c.add_close_camera(scene, lookat=(*XY, 0.17)) if args.record else None
scene.build(n_envs=N)
m = c.set_parts(part, mats, true_mu)
c.setup_franka(arm, N)
if cam:
    c.start_recording(cam, args.record)

z_grip = c.SIDE / 2 + 0.115  # 手先（hand）から指先の中心までが約 0.115 m


step_no = 0
hand = arm.get_link("hand")


def go(z, steps, f=None):
    """手先を高さ z へ向かわせる指令を出し（f が数値なら、その力で指を閉じる）、時計を steps 回進める。"""
    global step_no
    c.command(arm, [*XY, z], N, force=f)
    for _ in range(steps):
        scene.step()
        step_no += 1
        if rr:  # 24 の環境の、部品の高さ [cm] と、指に対する部品のずれ [cm]（つかんだ後に増えたら、滑っている）
            rr.set_time("sim", duration=step_no * c.DT)
            pz = part.get_pos()[:, 2].cpu().numpy()
            gap = hand.get_pos()[:, 2].cpu().numpy() - pz
            for i, (mu, fo) in enumerate(grid):
                rr.log(f"part_height_cm/mu={mu:.2f}/F={fo:.0f}N", rr.Scalars(pz[i] * 100))
                rr.log(f"hand_minus_part_cm/mu={mu:.2f}/F={fo:.0f}N", rr.Scalars(gap[i] * 100))
            tl = c.tilt_deg(part)  # 部品の傾き [度]。指先に角で引っかかると大きくなる
            for i, (mu, fo) in enumerate(grid):
                rr.log(f"part_tilt_deg/mu={mu:.2f}/F={fo:.0f}N", rr.Scalars(tl[i]))


def phase(name):
    """いまの段階の名前を、ターミナルと rerun の記録（文字の欄）に残す。"""
    if rr:
        rr.set_time("sim", duration=step_no * c.DT)
        rr.log("phase", rr.TextLog(name))


phase("部品の上へ")
go(0.25, 150)              # 部品の上へ
phase("下ろす")
go(z_grip, 150)            # 下ろす
if args.mistake in ("slam", "slam_substeps1"):  # いきなり目標の力で閉じる
    phase("いきなり閉じる")
    go(z_grip, 100, force)
else:
    phase("そっと閉じる")
    go(z_grip, 50, np.minimum(force, 5.0))  # そっと閉じる（5 N まで）
    phase("目標の力で握る")
    go(z_grip, 50, force)                   # それから目標の力
phase("持ち上げる")
# ゆっくり持ち上げる。目標を一度に 16.5 cm 上に置くと、ロボットは急加速して「重さ＋加速の分」が要る
# （わざと失敗 jump で確かめる）。ここでは 2 秒かけて滑らかに動かし、静止して持つ条件だけを見る
if args.mistake == "jump":
    go(0.30, 200, force)
else:
    for k in range(200):
        s_ = (k + 1) / 200
        go(z_grip + (0.30 - z_grip) * (3 * s_**2 - 2 * s_**3), 1, force)  # 滑らかな S 字（始めと終わりの速さが 0）
go(0.30, 50, force)
z = part.get_pos()[:, 2].cpu().numpy()

tilt = c.tilt_deg(part)
lifted = z > 0.10           # 高さ 10 cm より上にある
upright = tilt < 10.0       # 傾き 10 度未満（指先に角で引っかかって、傾いてぶら下がっていない）
held = lifted & upright     # 持てた = 持ち上がっていて、まっすぐ（L1-c：高さだけで判定すると、ぶら下がりも「持てた」になる）
if cam:
    c.stop_recording(cam, args.record, cols=len(FORCES), labels=[f"mu={mu:.2f} F={f:.0f}N" for mu, f in grid], ok=held)
theory = 2 * true_mu * force >= m[0] * c.G
print(f"part: aluminium {m[0] * 1000:.0f} g (m g = {m[0] * c.G:.2f} N)   mistake={args.mistake}")
print("simulator (OK = held, TL = lifted but tilted, -- = dropped)   theory 2 mu F >= m g")
print("mu \\ F  " + "".join(f"{f:>6.0f}N" for f in FORCES) + "   " + "".join(f"{f:>6.0f}N" for f in FORCES))
match = int((held == theory).sum())
for i, mu in enumerate(MUS):
    cell = lambda k: "OK" if held[k] else ("TL" if lifted[k] else "--")  # noqa: E731
    s = "".join(f"{cell(i * len(FORCES) + j):>7s}" for j in range(len(FORCES)))
    t = "".join(f"{'OK' if theory[i * len(FORCES) + j] else '--':>7s}" for j in range(len(FORCES)))
    print(f"{mu:5.2f}  {s}   {t}")
print(f"match {match}/{N}")
for k in np.where(lifted & ~upright)[0]:
    print(f"tilted: mu={grid[k][0]:.2f} F={grid[k][1]:.0f}N  {tilt[k]:.0f} deg（持ち上がったが、傾いてぶら下がっている）")
if view:
    c.hold_viewer(scene)
