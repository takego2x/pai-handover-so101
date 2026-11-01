"""潰れる箱（重さ・強度・摩擦がランダム）を、1 台目が触って測り、2 台目が情報で素早く運ぶ。

  1 台目（--first 1〜3 台）：
    ゆるい力（τ 0.25）から 0.05 ずつ強めて試し掴み → 持ち上がった最小の力（力加減。重さと摩擦を含む）
    → 宙で τ 0.4 で握り、肩の力から重さ、指の角度から硬さ → 強度を推定（箱の幅は 3 cm と決まっているので、角度＝潰れ具合）
    → 台座に置き、情報（力加減・重さ・推定強度）を渡す。
    強度が分かるまでは τ 0.4（約 5.6 N）を超えない（どの箱の強度より小さい、と箱の側で決めてある）。
    分かった後は、推定強度の 6 割を超えない。
  見る：作業台を真上から見る固定カメラ（ロボットには付けない）の深度から、箱の位置と向きを出す（handover/vision.py）。
    掴みに行く前と、掴み損ねて手を退かせた後に見直す。2 台目は台座の上の箱を見てから持つ。
    手順の判断には、シミュレータの箱の座標を使わない（使うのはカメラ、自分の関節の角度と力だけ）。答え合わせにだけ使う。
  --camera：同じ固定カメラの画像を Gemini に見せ、箱を指し示させ、見た目から「壊れにくいか」を予想させる。
    壊れにくい箱は大きく刻んで試す（試す回数を減らす）。強度は必ず触って測る。
  2 台目：力加減 × 2（ただし推定強度の 6 割まで）で 1 回で持ち、重さに合わせた速さで大箱へ。

  4 台を同時に動かすため、腕ごとの手順を 1 step ずつ進む形（generator）で書く（line3.py と同じ）。

使い方（答え合わせ用リポジトリの直下で）:
  docker compose run --rm sim python handover/soft_line.py --first 3 --seed 1 [--per 2] [--camera] [--record out.mp4 --cam over]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101 as s  # noqa: E402
import softbox as sb  # noqa: E402
import vision as vi  # noqa: E402
import carry_policy as cp  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent / "spikes"))
from vlm_softboxes import surface  # noqa: E402   見た目の絵（Gemini に見せた写真と同じ質感）

ap = argparse.ArgumentParser()
ap.add_argument("--first", type=int, default=1, choices=[1, 2, 3])
ap.add_argument("--per", type=int, default=2, help="1 台目 1 台あたりの箱の数")
ap.add_argument("--total", type=int, help="箱の総数（省略時は first × per）。台数を変えて同じ箱を比べるとき")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--camera", action="store_true")
ap.add_argument("--policy", action="store_true", help="運び方（握る力・速さ）を、学習した方策で決める（handover/data/carry_policy.json）")
ap.add_argument("--record")
ap.add_argument("--cam", default="over")
ap.add_argument("--max-steps", type=int, default=40000)
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]

ROBOT_MU = 0.1
POLICY = json.loads(cp.W_PATH.read_text()) if args.policy else None
SQ_PER_TAU = 14.5
TAU0, TAU_STEP, TAU_MEAS = 0.2, 0.025, 0.4      # 試し始める力、強める幅、強度を測る力（約 5.6 N。どの箱の強度 7.5 N より小さい）
# 見た目の手がかりがないときは、強度が分からないので小さく刻む（τ 0.2 から 0.025 ずつ）。
# カメラ（Gemini）が「壊れにくい」と答えた箱は、大きく刻む（τ 0.25 から 0.05 ずつ）。Gemini は材質は当てるが、重さは当てられない
# （段ボールも発泡スチロールも数十 g 以下と答え、本当は 143〜231 g の箱もあった）ので、重さの予想は使わない。
# 刻み方を変えても、試す力の上限 τ 0.4（約 5.6 N）は、どの箱の強度（7.5 N 以上）より小さいので潰さない
CAM_TAU0, CAM_STEP = 0.25, 0.05
SAFE = 0.6                                      # 推定強度のこの割合までしか握らない（推定の誤差が 2 割ほどあるため）
MARGIN = 3.0                                    # 力加減の何倍で握って運ぶか（強度の上限まで）。2.5 では、試しで 20 mm のうち 16.7 mm しか上がっていない箱
                                                #   （手首カメラの見た目では滑りが見えない）を運び始めに滑らせ、台座に横倒しで置いた（12 箱の box7、2026-10-27）
SLOW_REF = 2.5                                  # 余裕がこれに届かない箱は、足りない分だけゆっくり運ぶ（振られる力は速さの 2 乗）。2.0 では 205 g の段ボールを 1 台目が台座へ運ぶ途中で落とした（当ててから閉じる形にして、力加減が床で滑らせる分だけ小さく＝正しく出るようになった後。2026-10-27）
MIN_MARGIN = 1.8                                # 余裕がこれに届かない箱（重さに対して弱すぎる）は運ばない（1.57 倍の箱を 1.6 倍ゆっくり運んでも、運び始めに落とした。
                                                #   1.62 倍の箱を 2.4 倍ゆっくり運んでも、2 台目が途中で落とした。1.91 倍は運べた。12 箱の box9）
D, R, STAND_H = 0.44, 0.22, 0.02
# 1 台目が床の箱を掴む高さ zg は 1.5 cm（箱 3 cm の下半分を挟む）。2.0 cm では指が箱の上 1 cm にしか触れず、試し掴みで少し滑ると
# 上の端だけをつまむ形になり、運ぶ途中に回って抜けた（214 g・摩擦 0.62 の箱、2026-10-27）
Z_CARRY_A, Z_CARRY_B, LIFT_DZ, zg = 0.06, 0.08, 0.02, float(os.environ.get("SOFT_ZG", "0.015"))   # その 2 は 8 cm（7 cm では箱の底と大箱の壁の上端のすき間が 1 cm しかない）
zs = STAND_H + 0.020
HALF = 0.6
# 校正（handover/spikes/soft_mu.py、板 2 枚の箱、100 g、宙で τ 0.4）：指の角度 → 硬さ k。重い箱は角度が 0.037 rad/kg だけ大きく出るので差し引く
CAL_Q = np.array([0.1682, 0.1761, 0.1821, 0.1878, 0.1911, 0.1948])
CAL_K = np.array([2500.0, 3500.0, 5000.0, 8000.0, 12000.0, 20000.0])
Q_PER_KG = 0.037
# 1 台目の下ろし方：固定側の指を 4 mm 外に構えて下ろし（箱の上に乗らないように）、閉じる前に箱の側面へ 3 mm 押し当てる。
# 当てずに閉じると、動く側の指だけが箱を押し、箱は床に押し付けられて滑らず固定側の指に届かない。届くまで力を上げるので、
# 力加減に「箱を床で滑らせる力」が混ざっていた（τ 0.2〜0.25 は動く側の指だけが触れ、0.275 でやっと両側。2026-10-27 実測）
CLEAR_DOWN, PRESS = 0.004, -0.003
HOLDING_Q = 0.05                                 # 指の角度がこれより大きければ、何かを挟んでいる（空なら閉じきって −0.17）

STATIONS = {1: [0], 2: [0, -100], 3: [0, 100, -100]}[args.first]
SIDE_OF = {0: 1, 100: -1, -100: 1}                # 台ごとに、箱を置く側（隣の台と大箱を避ける）
BIG_AZ = 50
BOX_AZ = [40, 64, 88, 28, 52, 76]                  # 箱を置く向き（腕の正面から）。4 個目からは外の列
BOX_R = [R, R, R, 0.285, 0.285, 0.285]            # 腕の根元からの距離。外の列は互い違い（となりの箱まで 8.8 cm 以上）。内の列（14.5 cm）は、構えた手の下で
                                                  # 動き出しにはじき（28°）、102° は根元の回転の限界 110° に当たった（2026-10-27）

specs = sb.random_boxes(args.total or args.first * args.per, seed=args.seed)
TOP_KEY = f"top_first{args.first}_n{len(specs)}_seed{args.seed}"     # 固定カメラの画像と、Gemini の答えの名前
TOP_PROMPT = """This top-down image shows a robot work table. There are small boxes (each about 3 cm) on the table and on small gray stands.
Ignore the yellow robot arms, the gray stands and the large open brown tray.
For EVERY small box, return one JSON object:
- "point": [y, x] of the box center, normalized to 0-1000
- "looks_like": what the box looks like it is made of
- "weight_g": your best guess of its total weight in grams (contents included), as [low, high]
- "fragile": true if squeezing it hard could crush or damage it
- "confidence": "low", "medium" or "high" for the weight guess
Return only a JSON array, no prose."""
sc = s.make_scene()
sc.add_entity(s.gs.morphs.Plane())
ARM_MODEL = vi.camera_model(s.MODEL)       # 手首のカメラ付きの公式モデル（カメラと取り付け金具に、重さと当たり判定がある）
entB = s.add_so101(sc, (0, 0), 0, friction=ROBOT_MU, model=ARM_MODEL)
entA = [s.add_so101(sc, (D * math.cos(math.radians(a)), D * math.sin(math.radians(a))), a + 180, friction=ROBOT_MU, model=ARM_MODEL)
        for a in STATIONS]
for a in STATIONS:
    sc.add_entity(s.gs.morphs.Box(size=(0.05, 0.05, STAND_H), pos=(R * math.cos(math.radians(a)), R * math.sin(math.radians(a)), STAND_H / 2),
                                  fixed=True), surface=s.gs.surfaces.Default(color=(0.6, 0.6, 0.65)))
bx, by = R * math.cos(math.radians(BIG_AZ)), R * math.sin(math.radians(BIG_AZ))
for dx, dy, sx, sy in [(0.07, 0, 0.004, 0.144), (-0.07, 0, 0.004, 0.144), (0, 0.07, 0.144, 0.004), (0, -0.07, 0.144, 0.004)]:
    sc.add_entity(s.gs.morphs.Box(size=(sx, sy, 0.04), pos=(bx + dx, by + dy, 0.02), fixed=True),
                  surface=s.gs.surfaces.Default(color=(0.75, 0.6, 0.4)))
boxes = []
for j, sp in enumerate(specs):
    i = j % len(STATIONS)
    slot = j // len(STATIONS)
    a = STATIONS[i]
    base, yaw = np.array([D * math.cos(math.radians(a)), D * math.sin(math.radians(a))]), math.radians(a + 180)
    w = yaw + SIDE_OF[a] * math.radians(BOX_AZ[slot])
    f = sb.xml(ROOT / "outputs" / "handover" / "softbox" / f"seed{args.seed}_{sp['name']}.xml", sp["k"], sp["mass"], sp["mu"], sp["look"])
    ent = sc.add_entity(s.gs.morphs.MJCF(file=f, pos=(base[0] + BOX_R[slot] * math.cos(w), base[1] + BOX_R[slot] * math.sin(w), 0), euler=(0, 0, math.degrees(w))),
                        surface=surface(s.gs, sp["look"]))
    boxes.append({**sp, "ent": ent, "owner": i})
CAMS = {"over": ((0.08, -0.34, 1.05), (0.08, -0.03, 0.0), 55),
        "one": ((0.24, -0.42, 0.62), (0.24, -0.02, 0.0), 55),
        "one_b": ((0.26, -0.42, 0.62), (0.26, -0.02, 0.0), 55)}     # one を 2 cm ずらしたもの（one で描画が止まったとき）       # 1 台目 1 台：真上寄りから、1 台目・台座・2 台目・大箱が全部見える
cams = {c: sc.add_camera(res=(1280, 720), pos=CAMS[c][0], lookat=CAMS[c][1], fov=CAMS[c][2], GUI=False)
        for c in args.cam.split(",")} if args.record else {}
# 作業台を真上から見る固定カメラ（高さ 1.8 m、作業台全体 1.3 m 角を 1280 画素で。1 画素 ≒ 1 mm）
topcam = vi.add_top_camera(sc, center=(0.05, 0.0), height=1.8, fov=40, res=(1280, 1280))
vi.add_top_camera_rig(sc, center=(0.05, 0.0), height=1.8)      # 支柱・腕・本体（見た目だけ。真上のカメラの画像には写らない）
handcams = [vi.add_hand_camera(sc) for _ in range(1 + len(STATIONS))]     # 手首のカメラ（その 2、その 1 の順）
sc.build(n_envs=1)
for b in boxes:
    sb.hold_spring(b["ent"])
B = s.Arm(sc, entB, (0, 0), 0, 1)
A = [s.Arm(sc, e, (D * math.cos(math.radians(a)), D * math.sin(math.radians(a))), a + 180, 1) for e, a in zip(entA, STATIONS)]
for arm, hc in zip([B] + A, handcams):
    arm.hand = vi.attach_hand(hc, arm.e)
for _ in range(30):
    sc.step()

# ---- 最初に見る：固定カメラで箱を見つけ、近い 1 台目に割り当てる ----
STAND_XY = [np.array([R * math.cos(math.radians(a)), R * math.sin(math.radians(a))]) for a in STATIONS]
seen0 = [c for c in vi.look(topcam) if min(np.linalg.norm(np.array(c[:2]) - sxy) for sxy in STAND_XY) > 0.04]
for b in boxes:          # 答え合わせ用：見つけた箱が、どの本当の箱かの対応（手順の判断には使わない）
    p = b["ent"].get_pos().cpu().numpy()[0, :2]
    c = vi.nearest(seen0, p, max_d=0.03)
    b["seen"] = c
    b["owner"] = None if c is None else int(np.argmin([np.linalg.norm(np.array(c[:2]) - a_.base) for a_ in A]))
    if c is None:
        print(f"[ カメラ ] {b['name']} が見つからない", flush=True)
print(f"[ カメラ ] 箱 {len(seen0)} 個を見つけた", flush=True)
GEMINI = {}
if args.camera:
    import imageio
    img_f, ans_f = ROOT / "handover" / "data" / f"{TOP_KEY}.png", ROOT / "handover" / "data" / f"{TOP_KEY}.gemini.txt"
    rgb, depth, _, _ = topcam.render(rgb=True, depth=True)
    if not ans_f.exists():      # まだ聞いていなければ、その場で Gemini に聞いて保存（以後は使い回す）
        imageio.imwrite(img_f, np.asarray(rgb))
        sys.path.insert(0, str(ROOT / "sim"))
        import vlm_material as vm
        vm.PROMPT = TOP_PROMPT
        ans_f.write_text(vm.ask_gemini(img_f))
    txt = ans_f.read_text()
    answers = json.loads(txt.split("```json")[-1].split("```")[0] if "```" in txt else txt)
    P = vi._points(topcam, np.asarray(depth))
    H_, W_ = P.shape[:2]
    for a_ in answers:            # Gemini が指した点 → 深度で 3 次元の位置 → 一番近い、カメラで見つけた箱
        u, v = min(W_ - 1, int(a_["point"][1] / 1000 * W_)), min(H_ - 1, int(a_["point"][0] / 1000 * H_))
        c = vi.nearest(seen0, P[v, u, :2], max_d=0.05)
        if c is not None:
            GEMINI[id(c)] = a_
    print(f"[ Gemini ] 箱 {len(answers)} 個を指し、カメラで見つけた箱 {len(GEMINI)} 個と結び付けた", flush=True)
rec = Path(args.record) if args.record else None
if cams:
    rec.parent.mkdir(parents=True, exist_ok=True)
    for c, cam in cams.items():
        cam.start_recording(save_to_filename=str(rec.with_name(f"{rec.stem}_{c}.mp4")), fps=25)
t = [0]
marks = []
hand_writers = {}
if cams and not os.environ.get("NO_HAND"):      # 手首のカメラの映像も保存する（4 step に 1 コマ＝25 fps。録画と同じ）
    import imageio
    for nm, arm in [("hand_2", B)] + [(f"hand_1-{i}", a_) for i, a_ in enumerate(A)]:
        hand_writers[nm] = (arm, imageio.get_writer(str(rec.with_name(f"{rec.stem}_{nm}.mp4")), fps=25, macro_block_size=8))


def mark(who, name, what):
    marks.append({"step": t[0], "who": who, "box": name, "what": what})
    print(f"[{t[0] / 100:6.1f} s] {who:4s} {name:6s} {what}", flush=True)


def move(arm, az, r, z, g, n):
    az = np.broadcast_to(np.asarray(az, float), (1,))
    tgt = arm.world(az, np.broadcast_to(r, (1,)), np.broadcast_to(z, (1,)))
    start = arm.last if arm.last is not None else arm.tip()
    for k in range(n):
        f = min(1.0, (k + 1) / (0.7 * n))
        f = 3 * f ** 2 - 2 * f ** 3
        if k % 5 == 0 or k == n - 1:
            cur = start + (tgt - start) * f
            Rd = arm.Rd(az[0], arm.dyaw[0])
            q = arm.e.inverse_kinematics(link=arm.g, pos=(cur[0] - Rd @ s.OFF)[None], quat=s.quat_of(Rd)[None],
                                         init_qpos=arm.e.get_dofs_position(), max_solver_iters=50).cpu().numpy()
            q[:, 5] = g
            arm.e.control_dofs_position(q)
        yield
    arm.last = tgt


def aim(arm, seen, clearance=0.0):
    """カメラで見た箱（位置と向き）から、目印を置く (方位, 半径) と指を回す角度。四角なので向きは 90° ごとに同じ → ±45° に収める
    （箱は 2 つの向きに板があるので、どちらの向きで挟んでも硬さを測れる）"""
    cxy, yaw = np.array(seen[:2], float), float(seen[3])
    tip = cxy.copy()
    for _ in range(3):
        rel = tip - arm.base
        az = math.atan2(rel[1], rel[0]) - arm.yaw
        dyaw = (yaw - (arm.yaw + az) + math.pi / 4) % (math.pi / 2) - math.pi / 4
        c = arm.yaw + az + dyaw
        tip = cxy - (s.JAW + clearance) * np.array([math.cos(c), math.sin(c)])
    rel = tip - arm.base
    az = (math.atan2(rel[1], rel[0]) - arm.yaw + math.pi) % (2 * math.pi) - math.pi
    return az, float(np.linalg.norm(rel)), dyaw


def relook(xy, near=0.04, top_min=None):
    """固定カメラで見直し、xy に一番近い箱（near 以内）を返す。見えなければ None（手や腕に隠れたとき）"""
    cands = vi.look(topcam)
    if top_min is not None:
        cands = [c for c in cands if c[2] > top_min]
    return vi.nearest(cands, xy, max_d=near)


def holding(arm):
    """指の角度で、何かを挟んでいるか（ロボット自身の関節の値）"""
    return float(arm.e.get_dofs_position().cpu().numpy()[0, 5]) > HOLDING_Q


# 持てたかの基準（手首のカメラの色の画像＋肩のトルク。どちらも実機の SO-101 で取れる値）
SURE_MAD = 12.0                     # 画像：1 台目はこれ以下で「持てた」（しっかり持てた 1〜12、滑りながら上がった 19〜26。2026-10-27 実測）
MAD_MAX = 25.0                      # 画像：持ち上げる前後で、指のあいだの見た目の平均の差（持てた 1〜23、置き去り 27〜75。2026-10-27 実測）
M_MIN = 0.015                       # 肩のトルク：手に掛かった重さが 15 g 以上（いちばん軽い箱は 30 g）
# 画像の一致度（NCC）は使わない：2 台目が台座から持ち上げると、箱の縁の後ろの台座と床の見え方が変わり、持てていても 0.57〜0.71 になった。
# 白い発泡のように模様のない面でも当てにならない


def shoulder_tau(arm, n=60):
    """肩の関節の力 [N·m]（落ち着いた後の 30 step の平均）"""
    v = []
    for _ in range(n):
        yield
        v.append(float(arm.e.get_dofs_control_force().cpu().numpy()[0, 1]))
    return float(np.mean(v[-30:]))


def lever_of(arm, az):
    """肩から箱の中心までの水平の距離 [m]（自分の手先の位置から計算）"""
    cdir = arm.yaw + az + float(arm.dyaw[0])
    box_xy = arm.tip()[0, :2] + s.JAW * np.array([math.cos(cdir), math.sin(cdir)])
    return float(np.linalg.norm(box_xy - arm.e.get_link("upper_arm").get_pos().cpu().numpy()[0, :2]))


def lifted(arm, img_before, tau_ref, lever):
    """持ち上げた後に、持てたか。①指が何かを挟んでいる ②手首のカメラの画像で、指のあいだの見た目が持ち上げる前と変わらない
    （箱が手と一緒に上がった）③肩のトルクで、手に箱の重さが掛かっている。の全部。
    指の角度だけでは、箱が指のあいだに残ったまま床に置き去りになったのを見分けられなかった（重さを 0〜5 g と測った）。
    前は手首のカメラの深度で見ていたが、実機の手首カメラ（公式キットの UVC）には深度がない。深度では、滑りながら 1.4 cm しか
    上がらなかった箱を「持てた」とし、力加減を小さく見積もって台座の手前で落とした"""
    tau_now = yield from shoulder_tau(arm)
    img_after = vi.hand_rgb(arm.hand)
    ncc, mad = vi.same_view(vi.roi_of(img_before), vi.roi_of(img_after))
    dy, dx, _ = vi.box_shift(img_before, img_after)
    m = (tau_ref - tau_now) / (9.81 * lever)
    return holding(arm) and mad <= MAD_MAX and m >= M_MIN, (ncc, mad, m, dy, dx)


def z_of(e):
    return float(e.get_pos().cpu().numpy()[0, 2])


def tilt_of(e):
    q = e.get_quat().cpu().numpy()[0]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))


def k_from_angle(q, m):
    """宙で τ 0.4 で握ったときの指の角度（重さで補正）→ 硬さ k（校正の表を内挿。表の外は端の値）"""
    qc = q - Q_PER_KG * (m - 0.1)
    return float(np.exp(np.interp(qc, CAL_Q, np.log(CAL_K))))


def watch():
    if os.environ.get("SOFT_DEBUG"):            # 答え合わせ：高く上がった箱が床の近くまで落ちた瞬間
        for b in boxes:
            z = z_of(b["ent"])
            if z > 0.045:
                b["_zmax"] = max(b.get("_zmax", 0.0), z)
                if not b.get("_slip_shown") and z < b["_zmax"] - 0.004:      # 高い所で下がり始めた瞬間
                    b["_slip_shown"] = True
                    c = b["ent"].get_contacts()
                    nm = {}
                    for e_ in sc.entities:
                        for l_ in getattr(e_, "links", []):
                            nm[l_.idx] = f"{e_.idx}:{l_.name}"
                    own = {l_.idx for l_ in b["ent"].links}
                    pr = sorted({(nm.get(int(u), u), nm.get(int(v), v), round(float(np.linalg.norm(f)), 2)) for u, v, f, m in
                                 zip(c["link_a"].cpu().numpy()[0], c["link_b"].cpu().numpy()[0], c["force_a"].cpu().numpy()[0], c["valid_mask"].cpu().numpy()[0])
                                 if m and not (int(u) in own and int(v) in own)})
                    print(f"      DEBUG 下がり始めた {b['name']} {t[0] / 100:.2f} s: 高さ {z:.3f}（最高 {b['_zmax']:.3f}） 触れている {pr}", flush=True)
            # 答え合わせ：床にあって、まだ誰も手を付けていない箱が動いた瞬間（ほかの箱を扱う腕にはじかれた）
            if not b.get("_busy") and t[0] % 20 == 0:
                p_ = b["ent"].get_pos().cpu().numpy()[0, :2]
                if "_p0" not in b:
                    b["_p0"] = p_
                elif not b.get("_bumped") and np.linalg.norm(p_ - b["_p0"]) > 0.005:
                    b["_bumped"] = True
                    near = sorted((float(np.linalg.norm(a_.tip()[0, :2] - p_)), nm) for nm, a_ in [("2", B)] + [(f"1-{k}", x) for k, x in enumerate(A)])[0]
                    print(f"      DEBUG はじかれた {b['name']} {t[0] / 100:.2f} s: {np.round(b['_p0'], 3).tolist()} → {np.round(p_, 3).tolist()}  "
                          f"いちばん近い手 {near[1]}（{near[0] * 100:.1f} cm）  傾き {tilt_of(b['ent']):.0f}°", flush=True)
            if z > 0.045:
                b["_high"] = True
            elif z < 0.02 and b.get("_high"):
                b["_high"] = False
                hold_arms = [nm for nm, a_ in [("2", B)] + [(f"1-{k}", x) for k, x in enumerate(A)] if holding(a_)]
                print(f"      DEBUG 落ちた {b['name']} {t[0] / 100:.2f} s: 箱 {np.round(b['ent'].get_pos().cpu().numpy()[0], 3).tolist()} 傾き {tilt_of(b['ent']):.0f}°  "
                      f"何かを挟んでいる腕 {hold_arms}  札 {locks}", flush=True)
    for b in boxes:
        if not b.get("crushed") and sb.crushed(b["ent"]):
            b["crushed"] = True
            mark("!", b["name"], "潰れた")
            if os.environ.get("SOFT_DEBUG"):
                c = b["ent"].get_contacts()
                names = {}
                for e in sc.entities:
                    for l in getattr(e, "links", []):
                        names[l.idx] = f"{e.idx}:{l.name}"
                pairs = [(names.get(int(u), u), names.get(int(v), v), round(float(np.linalg.norm(f)), 1)) for u, v, f, m in
                         zip(c["link_a"].cpu().numpy()[0], c["link_b"].cpu().numpy()[0], c["force_a"].cpu().numpy()[0], c["valid_mask"].cpu().numpy()[0]) if m]
                print(f"      DEBUG 潰れた瞬間 {b['name']}: 箱 {np.round(b['ent'].get_pos().cpu().numpy()[0], 3).tolist()} 触れている {pairs}  B の挟む力 {float(s.squeeze(B, b['ent'])[0]):.1f}", flush=True)


stands = [{"box": None, "info": None, "ready_at": None, "busy": False, "blocked": False} for _ in STATIONS]
GRIP_OK = (0.10, 0.23)        # 3 cm の箱を挟んだときの指の角度の範囲（0.17〜0.20 前後）。外れたら掴み損ね
# 台座ごとの「使用中」の札。1 台目は台座へ近づいてから退くまで、2 台目は台座から持つときと、運ぶ途中にその台座の上を通るとき、
# 札を取ってから動く（取れなければ待つ）。札がないと、2 台目が −100° の台座から大箱（+50°）へ運ぶ途中に 0° の台座の上を通り、
# 置きに来た 1 台目の指が、2 台目の持つ箱の板を 18 N で押して潰した
locks = [None] * len(STATIONS)


def take(idxs, who):
    """札をまとめて取る（全部取れるまで待つ）"""
    while any(locks[j] not in (None, who) for j in idxs):
        yield
    for j in idxs:
        locks[j] = who


def give(idxs, who):
    for j in idxs:
        if locks[j] == who:
            locks[j] = None


def crossed(a0, a1):
    """方位 a0 → a1（度）へ回る途中に通る台座"""
    lo, hi = min(a0, a1), max(a0, a1)
    return [j for j, a in enumerate(STATIONS) if lo <= a <= hi]


def tip_for(arm, center_xy):
    """箱の中心を center_xy に置くための、目印の (方位, 半径)。指を回した角度 dyaw の分も含めて逆算する
    （回した角度を見ずに目印を台座の中心から JAW 手前に置くと、箱の中心が最大 1.7 cm ずれ、2 台目が掴み損ねた）"""
    tip = np.array(center_xy, float)
    for _ in range(3):
        rel = tip - arm.base
        az = math.atan2(rel[1], rel[0]) - arm.yaw
        c = arm.yaw + az + float(arm.dyaw[0])
        tip = np.array(center_xy, float) - s.JAW * np.array([math.cos(c), math.sin(c)])
    rel = tip - arm.base
    return (math.atan2(rel[1], rel[0]) - arm.yaw + math.pi) % (2 * math.pi) - math.pi, float(np.linalg.norm(rel))


def first(i):
    arm = A[i]
    a = STATIONS[i]
    mine = [x for x in boxes if x["owner"] == i]
    mine.sort(key=lambda x: math.atan2(x["seen"][1] - arm.base[1], x["seen"][0] - arm.base[0]))   # 見つけた順（方位の順）
    for b in mine:
        name, ent, seen = b["name"], b["ent"], b["seen"]
        b["_busy"] = True
        give([i], f"1-{i}")
        # 台座のとなりの箱を試して測る間は、その台座の札を持つ（2 台目が台座の箱を取りに来ると、腕どうしが触れて肩の力がずれ、
        # 重さを −1 kg・1.6 kg と測った。3 台の構成の 1-1、seed 333・118、2026-10-27）
        if math.hypot(seen[0] - STAND_XY[i][0], seen[1] - STAND_XY[i][1]) < 0.17:
            yield from take([i], f"1-{i}")
        c = relook(seen[:2])                     # 近づく前に見直す（ほかの箱を処理している間に動いたかもしれない）
        seen = c if c is not None else seen
        ans = GEMINI.get(id(b["seen"])) if GEMINI else None
        sturdy = ans is not None and not ans.get("fragile", True)
        tau, step = (CAM_TAU0, CAM_STEP) if sturdy else (TAU0, TAU_STEP)
        mark(f"1-{i}", name, f"試し始め τ {tau:.2f}、{step} ずつ" +
             (f"（カメラ：{ans['looks_like']}・{'壊れにくい' if sturdy else '壊れやすい'}・{ans['weight_g']} g）" if ans else "（カメラなし）"))
        # 空の手の肩の力は、押し当てた位置（持った後に重さを測る位置と同じ）の上で測る。外に構えた位置で測り、持った後に
        # 手を 7 mm 横へ戻して測ると、重くて滑りやすい箱（box4）がその動きで指から抜けた（重さ 7 g と測り、床の箱を置き直しで潰した）
        az, r, arm.dyaw[:] = aim(arm, seen, clearance=PRESS)
        arm.set_grip_tau(s.TAU_MAX)
        yield from move(arm, az, r, Z_CARRY_A, s.CLOSE, 150)
        v = []
        for _ in range(100):
            yield
            v.append(arm.e.get_dofs_control_force().cpu().numpy()[0, 1])
        tau_empty = float(np.mean(v))
        az, r, arm.dyaw[:] = aim(arm, seen, clearance=CLEAR_DOWN)
        yield from move(arm, az, r, 0.08, s.OPEN, 100)
        yield from move(arm, az, r, zg + LIFT_DZ, s.OPEN, 60)     # 空の手の肩の力を、持ち上げる高さで測っておく（持てたかの比べる相手）
        tau_lift_ref = yield from shoulder_tau(arm)
        # 基準は、宙で測った空の手の力とほぼ同じになるはず（高さが 2.5 cm 違うだけ。ふだんの差は 0.01 N·m）。大きくずれたら、
        # 腕が何かに押されている（隣の腕が手首のカメラに触れ、−0.43 → −2.42 N·m。重さを −1 kg と測り続け、強い金属の箱を
        # 「安全な力では持てない」とした。2026-10-27、seed 333）。少し待って測り直す
        for _ in range(5):
            if abs(tau_lift_ref - tau_empty) <= 0.1:
                break
            for _ in range(100):
                yield
            tau_lift_ref = yield from shoulder_tau(arm)
        if os.environ.get("SOFT_DEBUG"):
            c_ = arm.e.get_contacts()
            nm_ = {l.idx: l.name for l in arm.e.links}
            for x in boxes:
                for l in x["ent"].links: nm_[l.idx] = f"{x['name']}:{l.name}"
            pr_ = sorted({(nm_.get(int(u), int(u)), nm_.get(int(v), int(v))) for u, v, m_ in zip(c_["link_a"].cpu().numpy()[0], c_["link_b"].cpu().numpy()[0], c_["valid_mask"].cpu().numpy()[0]) if m_})
            print(f"      DEBUG 空の手の基準 {name}: 宙 {tau_empty:.3f}  持ち上げる高さ {tau_lift_ref:.3f} N·m  触れている {pr_}", flush=True)
        yield from move(arm, az, r, zg, s.OPEN, 60)
        if os.environ.get("SOFT_DEBUG"):
            c = arm.e.get_contacts()
            names = {l.idx: l.name for l in arm.e.links}
            for x in boxes:
                for l in x["ent"].links: names[l.idx] = f"{x['name']}:{l.name}"
            pairs = sorted({(names.get(int(u), int(u)), names.get(int(v), int(v))) for u, v, m in zip(c["link_a"].cpu().numpy()[0], c["link_b"].cpu().numpy()[0], c["valid_mask"].cpu().numpy()[0]) if m})
            print(f"      DEBUG 下ろした後 {name}: 目印 {np.round(arm.tip()[0], 3).tolist()} 関節 {np.round(arm.e.get_dofs_position().cpu().numpy()[0], 2).tolist()} 触れている {pairs}", flush=True)
        yield from move(arm, az, r, zg, s.PRE, 40)
        az, r, _ = aim(arm, seen, clearance=PRESS)                  # 固定側の指を箱の側面に当ててから閉じる
        yield from move(arm, az, r, zg, s.PRE, 40)
        cap, strength, found, tries, lost = TAU_MEAS, None, None, 0, False
        while tau <= cap + 1e-9:
            arm.set_grip_tau(tau)
            yield from move(arm, az, r, zg, s.CLOSE, 200)
            img_before = vi.hand_rgb(arm.hand)
            z_before = z_of(ent)
            if os.environ.get("SOFT_DEBUG"):
                c = arm.e.get_contacts()
                nm = {l.idx: l.name for l in arm.e.links}
                for x in boxes:
                    for l in x["ent"].links: nm[l.idx] = f"{x['name']}:{l.name}"
                pr = sorted({(nm.get(int(u), int(u)), nm.get(int(v), int(v))) for u, v, m in zip(c["link_a"].cpu().numpy()[0], c["link_b"].cpu().numpy()[0], c["valid_mask"].cpu().numpy()[0])
                             if m and not (int(u) in nm and int(v) in nm and ":" not in nm[int(u)] and ":" not in nm[int(v)])})
                cdir = arm.yaw + az + float(arm.dyaw[0])
                want = arm.tip()[0, :2] + s.JAW * np.array([math.cos(cdir), math.sin(cdir)])
                print(f"      DEBUG 閉じた {name} τ {tau:.3f}: 指 {float(arm.e.get_dofs_position().cpu().numpy()[0, 5]):.3f}  見た箱 {np.round(seen[:2], 3).tolist()} 向き {math.degrees(seen[3]):.0f}°  "
                      f"本当 {np.round(ent.get_pos().cpu().numpy()[0], 3).tolist()} 傾き {tilt_of(ent):.0f}°  指のあいだの中心 {np.round(want, 3).tolist()}  指の回転 {math.degrees(float(arm.dyaw[0])):.0f}°  触れている {pr}", flush=True)
            yield from move(arm, az, r, zg + LIFT_DZ, s.CLOSE, 60)
            tries += 1
            ok, (ncc, mad, m_try, dy, dx) = yield from lifted(arm, img_before, tau_lift_ref, lever_of(arm, az))
            if os.environ.get("SOFT_DEBUG"):
                print(f"      DEBUG 試し {name} τ {tau:.3f}: 指 {float(arm.e.get_dofs_position().cpu().numpy()[0, 5]):.3f}  画像 MAD {mad:.1f} ずれ dy {dy:+d} dx {dx:+d}  "
                      f"手の重さ {m_try * 1000:.0f} g  箱が上がった {(z_of(ent) - z_before) * 1000:.1f} mm（手は {LIFT_DZ * 1000:.0f} mm、答え合わせ）", flush=True)
            # 1 台目は、しっかり持てたとき（見た目の差が SURE_MAD 以下）だけを「持てた」とする。少しでも滑ったら、いったん置いて
            # 1 段強い力で掴み直す。滑りながら上がった箱（τ 0.20、205 g、2 cm 上げた手に 1.1 cm しか付いてこず見た目の差 21）は、
            # 上の端だけをつまむ持ち方のまま 10〜17° 傾いてぶら下がり、力を 2.4 倍にしても 8 秒かけて回って抜け落ちた
            # 安全な上限の力の回は、2 台目と同じ基準（見た目の差 MAD_MAX 以下）で決める。上限では掴み直す先がなく、
            # 持ち上がった箱（手 20 mm に箱 20.3 mm、274 g）を見た目の差 14 で「滑った」とし、「安全な力では持てない」と取り違えた
            # （試しの途中で箱が裏返り、見た目が変わっていた。2026-10-27、12 箱の box4）。余裕が足りなければ、この後の判定で人の確認へ
            last_try = tau + step > cap + 1e-9
            if ok and (mad <= SURE_MAD or last_try):   # 箱が手と一緒に上がった（手首のカメラの見た目が変わらず、肩に箱の重さが掛かっている）
                found = tau
                break
            if ok and os.environ.get("SOFT_DEBUG"):
                print(f"      DEBUG {name} τ {tau:.3f}: 持ち上がったが少し滑った（見た目の差 {mad:.0f}）→ 置いて掴み直す", flush=True)
            # 持ち上がらなかった：その場で指を開き、手を根元の側へ退かせてから、固定カメラで箱を見直して下ろし直す
            # （閉じたまま下ろすと、指のあいだから抜け落ちた箱の上に、閉じきった指が乗るだけになった。真上から見るので、
            #   手が箱の上にあると箱が隠れる）
            arm.set_grip_tau(s.TAU_MAX)
            yield from move(arm, az, r, zg + LIFT_DZ, s.OPEN, 50)
            yield from move(arm, az, r, 0.10, s.OPEN, 50)        # 真上へ上げてから根元へ退く（斜めに退くと、開いた指が内側の列の箱をはじいた。12 箱の box10）
            yield from move(arm, az, r - 0.08, 0.10, s.OPEN, 40)
            c = relook(seen[:2])
            if c is None:
                # 見えない：自分の腕が箱の上を覆っている（8 cm 退いただけでは、2 回目の試しで 2 cm 押した箱が隠れたまま、
                # 古い位置を 7 回空振りして「安全な力では持てない」と取り違えた。2026-10-27、seed 22 の box4）。根元へ大きく引いて見直す
                yield from move(arm, az, 0.12, 0.20, s.OPEN, 60)
                c = relook(seen[:2])
            if os.environ.get("SOFT_DEBUG"):
                print(f"      DEBUG 見直し {name}: {'見えない' if c is None else [round(c[0], 3), round(c[1], 3), round(c[2], 3)]}", flush=True)
            if c is None:
                lost = True
                break
            seen = c
            az, r, arm.dyaw[:] = aim(arm, seen, clearance=CLEAR_DOWN)
            yield from move(arm, az, r, 0.08, s.OPEN, 60)
            yield from move(arm, az, r, zg, s.OPEN, 60)
            yield from move(arm, az, r, zg, s.PRE, 40)
            az, r, _ = aim(arm, seen, clearance=PRESS)
            yield from move(arm, az, r, zg, s.PRE, 40)
            tau = round(tau + step, 3)
        if found is None:
            b.update(tries=tries)
            mark(f"1-{i}", name, f"固定カメラで見失った（{tries} 回目の後）→ 人の確認へ" if lost else f"安全な力の範囲では持てない（{tries} 回）→ 渡さない")
            arm.set_grip_tau(s.TAU_MAX)
            yield from move(arm, az, r, 0.12, s.OPEN, 80)
            continue
        # 宙で：重さ（肩の力）と、強度（指の角度）を同時に測る
        meas_tau = max(found, TAU_MEAS)       # 宙で τ 0.4 で握り、指の角度から強度を測る（found ≦ 0.4 なので、0.4 に上げる）
        arm.set_grip_tau(meas_tau)
        yield from move(arm, az, r, Z_CARRY_A, s.CLOSE, 100)
        taus, qs = [], []
        for _ in range(100):
            yield
            taus.append(arm.e.get_dofs_control_force().cpu().numpy()[0, 1])
            qs.append(arm.e.get_dofs_position().cpu().numpy()[0, 5])
        cdir = arm.yaw + az + float(arm.dyaw[0])          # 箱の中心＝自分の手先（関節の角度から計算）から、指の閉じる向きに JAW
        box_xy = arm.tip()[0, :2] + s.JAW * np.array([math.cos(cdir), math.sin(cdir)])
        lever = float(np.linalg.norm(box_xy - arm.e.get_link("upper_arm").get_pos().cpu().numpy()[0, :2]))
        m_hat = max((tau_empty - float(np.mean(taus))) / (9.81 * lever), 0.0)
        if m_hat < M_MIN or not holding(arm):
            # 測る途中で落とした（手に重さが掛かっていない）。ここで「置き直し」に進むと、床の箱の上で指を閉じて潰した。
            # 指を開いて離れ、人の確認へ回す
            b.update(found=found, tries=tries, set_aside=True)
            mark(f"1-{i}", name, f"重さを測る途中で落とした（{m_hat * 1000:.0f} g）→ 人の確認へ")
            arm.set_grip_tau(s.TAU_MAX)
            yield from move(arm, az, r, Z_CARRY_A, s.OPEN, 40)
            yield from move(arm, az, r - 0.08, 0.12, s.OPEN, 80)
            continue
        if strength is None:
            strength = sb.strength_of(k_from_angle(float(np.mean(qs[-30:])), m_hat))
        cap = SAFE * strength / SQ_PER_TAU
        if POLICY:      # 学習した方策：予測の成功率が 95% 以上で、いちばん速い運び方（届かなければ None ＝ 運ばない）
            ch = cp.choose(POLICY, found, m_hat, strength)
            hold = ch[0] if ch else found
        else:
            hold = min(found * MARGIN, max(cap, found))
        b.update(found=found, m_hat=m_hat, strength_est=strength, tries=tries)
        mark(f"1-{i}", name, f"力加減 {found:.2f}（{tries} 回）、重さ {m_hat * 1000:.0f} g（本当 {b['mass'] * 1000:.0f}）、"
                              f"強度 {strength:.1f} N（本当 {b['strength']:.1f}）、摩擦 {b['mu']}")
        arm.set_grip_tau(hold)
        if (ch is None) if POLICY else (hold / found < MIN_MARGIN):
            # 重さに対して弱すぎる：潰さない力では、運ぶ余裕が足りない（余裕 1.27 倍では、ゆっくり運んでも台座の手前で落とした）。
            # 運ばずに元の場所へ置き直し、人の確認に回す。強度を測らずに運べば、潰すか落とすかしていた箱
            mark(f"1-{i}", name, f"重さに対して弱すぎる（余裕 {hold / found:.2f} 倍）→ 運ばずに置き直し、人の確認へ")
            b["set_aside"] = True
            yield from move(arm, az, r, zg + 0.002, s.CLOSE, 120)
            arm.set_grip_tau(s.TAU_MAX)
            yield from move(arm, az, r, zg + 0.002, s.OPEN, 50)
            yield from move(arm, az, r, 0.12, s.OPEN, 80)
            continue
        give([i], f"1-{i}")             # 台座が空くのを待つ前に返す（持ったままだと、2 台目が台座の箱を取りに来られない）
        while stands[i]["box"] is not None and not stands[i]["blocked"]:
            yield
        if stands[i]["blocked"]:     # その 2 が持てなかった箱が台座に残っている：重ねて置くと箱どうしがぶつかって潰れた
            mark(f"1-{i}", name, "台座がふさがっている → 置かずに人の確認へ")
            b["set_aside"] = True
            yield from move(arm, az, r, zg + 0.002, s.CLOSE, 120)
            arm.set_grip_tau(s.TAU_MAX)
            yield from move(arm, az, r, zg + 0.002, s.OPEN, 50)
            yield from move(arm, az, r, 0.12, s.OPEN, 80)
            continue
        yield from take([i], f"1-{i}")
        slow = max(1.0, (SLOW_REF / (hold / found)) ** 2)    # 強度の上限で余裕を 2 倍まで足せない箱は、足りない分だけゆっくり（その 2 と同じ）
        n_carry = int(150 * (1 + m_hat / 0.15) * slow)
        if POLICY:
            n_carry, slow = int(150 * ch[1]), 1.0
            mark(f"1-{i}", name, f"方策：握る力 {hold:.2f}（力加減の {hold / found:.1f} 倍）、運ぶ {n_carry} step（成功の予測 {ch[2]:.0%}）")
        if slow > 1.0:
            mark(f"1-{i}", name, f"余裕が {hold / found:.2f} 倍しかない → {slow:.1f} 倍の時間をかけて台座へ")
        az_p, r_p = tip_for(arm, (R * math.cos(math.radians(a)), R * math.sin(math.radians(a))))
        yield from move(arm, az_p, r_p, Z_CARRY_A, s.CLOSE, n_carry)
        yield from move(arm, az_p, r_p, zs + 0.002, s.CLOSE, 120)
        yield from move(arm, az_p, r_p, zs + 0.002, s.OPEN, 60)
        yield from move(arm, az_p, r_p, 0.15, s.OPEN, 80)
        yield from move(arm, 1.2 * SIDE_OF[a], 0.18, 0.15, HALF, 100)   # 箱のある側へ退く（台座の上を横切らない）
        if os.environ.get("SOFT_DEBUG"):
            print(f"      DEBUG 置いて退いた後 {name}: 箱 {np.round(ent.get_pos().cpu().numpy()[0], 3).tolist()} 傾き {tilt_of(ent):.0f}°", flush=True)
        on_stand = relook(STAND_XY[i], near=0.025, top_min=STAND_H + 0.022)   # 固定カメラで、台座の上に箱があるか
        give([i], f"1-{i}")
        if on_stand is None:  # 置けていない箱は、その 2 に渡さない（台座の外へ落ちた箱を、その 2 が空振りした）
            mark(f"1-{i}", name, "台座の上に箱が見えない → 渡さない")
            b["dropped_by_A"] = True
            continue
        stands[i].update(box=b, info=(found, m_hat, strength), ready_at=t[0])
        mark(f"1-{i}", name, f"台座 {i} に置いた → 情報を渡す")
    give([i], f"1-{i}")
    mark(f"1-{i}", "-", "終わり")


def second():
    k = 0
    yield from move(B, math.radians(BIG_AZ), 0.15, 0.15, s.CLOSE, 100)
    while True:
        ready = [j for j, st in enumerate(stands) if st["box"] is not None and not st["busy"]]
        if not ready:
            if all(g is None for g in gens[1:]) and not any(st["box"] for st in stands):
                return
            yield
            continue
        j = min(ready, key=lambda j: stands[j]["ready_at"])
        st = stands[j]
        st["busy"] = True
        b, (found, m_hat, strength) = st["box"], st["info"]
        cap = SAFE * strength / SQ_PER_TAU
        ch_B = cp.choose(POLICY, found, m_hat, strength) if POLICY else None
        tau_B = ch_B[0] if ch_B else min(found * MARGIN, max(cap, found))
        mark("2", b["name"], f"台座 {j} から、情報で 1 回で持つ（τ {tau_B:.2f}）")
        path = sorted(set(crossed(STATIONS[j], BIG_AZ)) | {j})     # 持つ台座と、運ぶ途中に上を通る台座
        yield from take(path, "2")
        # 台座の上の箱を、近づく前に固定カメラで見る（近づいた後だと、自分の腕で台座が隠れた）
        seen_b = relook(STAND_XY[j], near=0.03, top_min=STAND_H + 0.022)
        got = False
        for attempt in range(2):          # 掴み損ねたら、離れて見直し、もう 1 回だけやり直す
            B.set_grip_tau(s.TAU_MAX)
            B.dyaw[:] = 0.0
            yield from move(B, math.radians(STATIONS[j]), 0.15, 0.18, s.CLOSE, 100)
            if seen_b is None:
                break
            # 固定側の指と箱の側面のすき間は 4 mm しかない。カメラの誤差（1〜2 mm）と、横の台座（±100°）へ腕を大きく回したときの
            # 手先の誤差（3〜5 mm）が重なると、固定側の指が箱の上に乗った（指先が 1.5 cm 高いまま閉じ、持ち上がらない／落とす）。
            # 指を外側に構えて下ろし、閉じながら箱を固定側の指へ寄せる
            az_b, r_b, B.dyaw[:] = aim(B, seen_b, clearance=0.004 + 0.002 * attempt)
            if os.environ.get("SOFT_DEBUG"):
                print(f"      DEBUG 2 が見た箱 {b['name']}: カメラ {np.round(seen_b[:3], 3).tolist()} 向き {math.degrees(seen_b[3]):.0f}°  "
                      f"本当 {np.round(b['ent'].get_pos().cpu().numpy()[0], 3).tolist()}  方位 {math.degrees(az_b):.0f}° 半径 {r_b:.3f} 指の回転 {math.degrees(float(B.dyaw[0])):.0f}°", flush=True)
            yield from move(B, az_b, r_b, 0.15, HALF, 80)
            yield from move(B, az_b, r_b, 0.15, s.OPEN, 30)
            yield from move(B, az_b, r_b, zs + 0.02, s.OPEN, 80)      # 空の手の肩の力を、持ち上げる高さで測っておく
            tau_ref_b = yield from shoulder_tau(B)
            yield from move(B, az_b, r_b, zs, s.OPEN, 40)
            yield from move(B, az_b, r_b, zs, s.PRE, 40)
            # 下ろしきってから、固定側の指を箱の側面まで寄せる（構えた分を戻す）。閉じながら箱を固定側の指まで滑らせる形だと、
            # 摩擦の大きい柔らかい箱（box3：摩擦 0.77・k 約 5000）は、弧を描いて閉じる動く側の指に台座へ押し付けられて傾き、
            # 板の縁が台座に食い込んで滑らず、固定側の指に届かなかった（台座からの力 8 N、箱の重さの 4 倍）
            az_b, r_b, _ = aim(B, seen_b, clearance=0.0)
            yield from move(B, az_b, r_b, zs, s.PRE, 40)
            B.set_grip_tau(tau_B)
            yield from move(B, az_b, r_b, zs, s.CLOSE, 200)
            q_grip = float(B.e.get_dofs_position().cpu().numpy()[0, 5])
            img_b = vi.hand_rgb(B.hand)
            zb_before = z_of(b["ent"])
            if os.environ.get("SOFT_DEBUG"):
                c_ = B.e.get_contacts()
                names_ = {}
                for e_ in sc.entities:
                    for l_ in getattr(e_, "links", []):
                        names_[l_.idx] = f"{e_.idx}:{l_.name}"
                own = {l_.idx for l_ in B.e.links}
                pairs_ = sorted({(names_.get(int(u), u), names_.get(int(v), v)) for u, v, m in zip(c_["link_a"].cpu().numpy()[0], c_["link_b"].cpu().numpy()[0], c_["valid_mask"].cpu().numpy()[0])
                                 if m and not (int(u) in own and int(v) in own)})
                print(f"      DEBUG 2 の触れている物 {b['name']}: {pairs_}  台座 j={j}  本当の箱の向き {math.degrees(2*math.atan2(float(b['ent'].get_quat().cpu().numpy()[0,3]), float(b['ent'].get_quat().cpu().numpy()[0,0]))):.0f}°", flush=True)
                print(f"      DEBUG 2 が閉じた {b['name']}: 挟む力 {float(s.squeeze(B, b['ent'])[0]):.2f} N  指 {q_grip:.2f}  目印 {np.round(B.tip()[0], 3).tolist()}  "
                      f"箱 {np.round(b['ent'].get_pos().cpu().numpy()[0], 3).tolist()}", flush=True)
                cb = b["ent"].get_contacts()
                box_own = {l_.idx for l_ in b["ent"].links}
                pairs_b = sorted({(names_.get(int(u), u), names_.get(int(v), v), round(float(np.linalg.norm(f)), 2)) for u, v, f, m in
                                  zip(cb["link_a"].cpu().numpy()[0], cb["link_b"].cpu().numpy()[0], cb["force_a"].cpu().numpy()[0], cb["valid_mask"].cpu().numpy()[0])
                                  if m and not (int(u) in box_own and int(v) in box_own)})
                print(f"      DEBUG 箱が触れている物 {b['name']}: {pairs_b}  傾き {tilt_of(b['ent']):.1f}°  板の沈み {np.round(b['ent'].get_dofs_position().cpu().numpy()[0, -2:] * 1000, 1).tolist()} mm  "
                      f"固定側の指 {np.round(B.g.get_pos().cpu().numpy()[0], 3).tolist()}  動く側の指 {np.round(B.e.get_link('moving_jaw_so101_v1').get_pos().cpu().numpy()[0], 3).tolist()}", flush=True)
            if GRIP_OK[0] < q_grip < GRIP_OK[1]:
                yield from move(B, az_b, r_b, zs + 0.02, s.CLOSE, 40)
                ok_b, (ncc, mad, m_b, dy, dx) = yield from lifted(B, img_b, tau_ref_b, lever_of(B, az_b))
                if os.environ.get("SOFT_DEBUG"):
                    print(f"      DEBUG 2 が持ち上げた {b['name']}: 画像 MAD {mad:.1f} ずれ dy {dy:+d} dx {dx:+d}  手の重さ {m_b * 1000:.0f} g  "
                          f"箱が上がった {(z_of(b['ent']) - zb_before) * 1000:.1f} mm（手は 20 mm、答え合わせ）", flush=True)
                if ok_b:
                    got = True
                    break
            mark("2", b["name"], f"掴み損ねた（指の角度 {q_grip:.2f}）→ 離れて見直す" if attempt == 0 else f"掴み損ねた（指の角度 {q_grip:.2f}）")
            B.set_grip_tau(s.TAU_MAX)
            yield from move(B, az_b, r_b, zs + 0.02, s.OPEN, 40)
            yield from move(B, az_b, r_b, 0.15, s.OPEN, 60)
            yield from move(B, math.radians(BIG_AZ), 0.15, 0.15, s.CLOSE, 80)     # 台座から離れてから見直す（腕で隠れないように）
            seen_b = relook(STAND_XY[j], near=0.03, top_min=STAND_H + 0.022)
        if not got:   # 持てなかった：台座はふさがったまま（1 台目は重ねて置かない）。人の確認へ
            mark("2", b["name"], "持ち上げられない → 台座はふさがったまま、人の確認へ")
            st.update(busy=False, blocked=True, box=None, info=None)
            give(path, "2")
            continue
        yield from move(B, az_b, r_b, Z_CARRY_B, s.CLOSE, 60)
        st.update(box=None, info=None, ready_at=None, busy=False)
        off = [(-0.03, -0.03), (0.03, -0.03), (-0.03, 0.03), (0.03, 0.03), (0.0, 0.0), (0.0, -0.035), (0.035, 0.0)][k % 7]
        k += 1
        tgt = np.array([bx + off[0], by + off[1]]) - B.base
        az_d = (math.atan2(tgt[1], tgt[0]) - B.yaw + math.pi) % (2 * math.pi) - math.pi
        B.dyaw[:] = 0.0
        # 運ぶ時間：重さに合わせる。さらに、強度の上限で余裕を 2 倍まで足せなかった箱は、足りない分だけゆっくり（振られる力は速さの 2 乗）
        margin_eff = tau_B / found
        slow = max(1.0, (SLOW_REF / margin_eff) ** 2)
        n_carry = int(150 * (1 + m_hat / 0.15) * slow)
        if ch_B:
            n_carry, slow = int(150 * ch_B[1]), 1.0
        if slow > 1.0:
            mark("2", b["name"], f"余裕が {margin_eff:.2f} 倍しかない → {slow:.1f} 倍の時間をかけて運ぶ")
        drop = {"z": None}
        def carry_watch():
            if drop["z"] is None and z_of(b["ent"]) < Z_CARRY_B - 0.03:
                drop["z"] = t[0]
                mark("2", b["name"], f"運ぶ途中で落とした（{np.round(b['ent'].get_pos().cpu().numpy()[0], 3).tolist()}）")
        for _ in move(B, az_d, float(np.linalg.norm(tgt)) - s.JAW, Z_CARRY_B, s.CLOSE, n_carry):
            carry_watch()
            yield
        yield from move(B, az_d, float(np.linalg.norm(tgt)) - s.JAW, Z_CARRY_B, s.OPEN, 50)
        give(path, "2")
        p = b["ent"].get_pos().cpu().numpy()[0]
        b["inside"] = abs(p[0] - bx) < 0.068 and abs(p[1] - by) < 0.068 and p[2] < 0.06
        mark("2", b["name"], f"大箱へ（{'入った' if b['inside'] else '外れた ' + str(np.round(p, 3).tolist())}）")
        yield from move(B, az_d, 0.15, 0.15, s.CLOSE, 60)


gens = [second()] + [first(i) for i in range(len(A))]
while any(g is not None for g in gens) and t[0] < args.max_steps:
    for i, g in enumerate(gens):
        if g is None:
            continue
        try:
            next(g)
        except StopIteration:
            gens[i] = None
    sc.step()
    watch()
    t[0] += 1
    if hand_writers and t[0] % 4 == 0:
        for arm, w in hand_writers.values():
            w.append_data(vi.hand_rgb(arm.hand))

if cams:
    for cam in cams.values():
        cam.stop_recording()
    for _, w in hand_writers.values():
        w.close()
summary = [{k: b.get(k) for k in ("name", "look", "mass", "k", "strength", "mu", "found", "m_hat", "strength_est", "tries", "crushed", "inside", "set_aside", "dropped_by_A")}
           for b in boxes]
if rec:
    rec.with_suffix(".json").write_text(json.dumps({"fps": 25, "steps": t[0], "args": vars(args), "marks": marks, "boxes": summary},
                                                   ensure_ascii=False, indent=1, default=float))
print(f"steps {t[0]} ({t[0] / 100:.1f} s)  first={args.first} camera={args.camera} seed={args.seed}  {'時間切れ' if t[0] >= args.max_steps else ''}")
print(" box    look       重さ g(測)   強度 N(推定)   摩擦  力加減  回数  結果")
for b in boxes:
    res = "潰した" if b.get("crushed") else ("OK" if b.get("inside") else ("外れた" if "inside" in b else ("台座の手前で落とした" if b.get("dropped_by_A") else ("人の確認へ" if b.get("set_aside") else "渡さず"))))
    print(f" {b['name']:6s} {b['look']:9s} {b['mass'] * 1000:4.0f}({(b.get('m_hat') or 0) * 1000:4.0f})  {b['strength']:5.1f}({b.get('strength_est') or 0:5.1f})"
          f"   {b['mu']:.2f}   {b.get('found') or 0:.2f}   {b.get('tries') or 0:2d}   {res}")
