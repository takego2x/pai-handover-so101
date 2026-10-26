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
sys.path.insert(0, str(Path(__file__).resolve().parent / "spikes"))
from vlm_softboxes import surface  # noqa: E402   見た目の絵（Gemini に見せた写真と同じ質感）

ap = argparse.ArgumentParser()
ap.add_argument("--first", type=int, default=1, choices=[1, 2, 3])
ap.add_argument("--per", type=int, default=2, help="1 台目 1 台あたりの箱の数")
ap.add_argument("--total", type=int, help="箱の総数（省略時は first × per）。台数を変えて同じ箱を比べるとき")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--camera", action="store_true")
ap.add_argument("--record")
ap.add_argument("--cam", default="over")
ap.add_argument("--max-steps", type=int, default=40000)
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]

ROBOT_MU = 0.1
SQ_PER_TAU = 14.5
TAU0, TAU_STEP, TAU_MEAS = 0.2, 0.025, 0.4      # 試し始める力、強める幅、強度を測る力（約 5.6 N。どの箱の強度 7.5 N より小さい）
# 見た目の手がかりがないときは、強度が分からないので小さく刻む（τ 0.2 から 0.025 ずつ）。
# カメラ（Gemini）が「壊れにくい」と答えた箱は、大きく刻む（τ 0.25 から 0.05 ずつ）。Gemini は材質は当てるが、重さは当てられない
# （段ボールも発泡スチロールも数十 g 以下と答え、本当は 143〜231 g の箱もあった）ので、重さの予想は使わない。
# 刻み方を変えても、試す力の上限 τ 0.4（約 5.6 N）は、どの箱の強度（7.5 N 以上）より小さいので潰さない
CAM_TAU0, CAM_STEP = 0.25, 0.05
SAFE = 0.6                                      # 推定強度のこの割合までしか握らない（推定の誤差が 2 割ほどあるため）
MARGIN = 2.0
MIN_MARGIN = 1.5                                # 余裕がこれに届かない箱（重さに対して弱すぎる）は運ばない
D, R, STAND_H = 0.44, 0.22, 0.02
Z_CARRY_A, Z_CARRY_B, LIFT_DZ, zg = 0.06, 0.08, 0.02, 0.020   # その 2 は 8 cm（7 cm では箱の底と大箱の壁の上端のすき間が 1 cm しかない）
zs = STAND_H + 0.020
HALF = 0.6
# 校正（handover/spikes/soft_mu.py、板 2 枚の箱、100 g、宙で τ 0.4）：指の角度 → 硬さ k。重い箱は角度が 0.037 rad/kg だけ大きく出るので差し引く
CAL_Q = np.array([0.1682, 0.1761, 0.1821, 0.1878, 0.1911, 0.1948])
CAL_K = np.array([2500.0, 3500.0, 5000.0, 8000.0, 12000.0, 20000.0])
Q_PER_KG = 0.037
HOLDING_Q = 0.05                                 # 指の角度がこれより大きければ、何かを挟んでいる（空なら閉じきって −0.17）

STATIONS = {1: [0], 2: [0, -100], 3: [0, 100, -100]}[args.first]
SIDE_OF = {0: 1, 100: -1, -100: 1}                # 台ごとに、箱を置く側（隣の台と大箱を避ける）
BIG_AZ = 50
BOX_AZ = [40, 64, 88]

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
entB = s.add_so101(sc, (0, 0), 0, friction=ROBOT_MU)
entA = [s.add_so101(sc, (D * math.cos(math.radians(a)), D * math.sin(math.radians(a))), a + 180, friction=ROBOT_MU) for a in STATIONS]
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
    ent = sc.add_entity(s.gs.morphs.MJCF(file=f, pos=(base[0] + R * math.cos(w), base[1] + R * math.sin(w), 0), euler=(0, 0, math.degrees(w))),
                        surface=surface(s.gs, sp["look"]))
    boxes.append({**sp, "ent": ent, "owner": i})
CAMS = {"over": ((0.08, -0.34, 1.05), (0.08, -0.03, 0.0), 55),
        "one": ((0.24, -0.42, 0.62), (0.24, -0.02, 0.0), 55)}       # 1 台目 1 台：真上寄りから、1 台目・台座・2 台目・大箱が全部見える
cams = {c: sc.add_camera(res=(1280, 720), pos=CAMS[c][0], lookat=CAMS[c][1], fov=CAMS[c][2], GUI=False)
        for c in args.cam.split(",")} if args.record else {}
# 作業台を真上から見る固定カメラ（高さ 1.8 m、作業台全体 1.3 m 角を 1280 画素で。1 画素 ≒ 1 mm）
topcam = vi.add_top_camera(sc, center=(0.05, 0.0), height=1.8, fov=40, res=(1280, 1280))
handcams = [vi.add_hand_camera(sc) for _ in range(1 + len(STATIONS))]     # 手首のカメラ（その 2、その 1 の順）
sc.build(n_envs=1)
for b in boxes:
    sb.hold_spring(b["ent"])
B = s.Arm(sc, entB, (0, 0), 0, 1)
A = [s.Arm(sc, e, (D * math.cos(math.radians(a)), D * math.sin(math.radians(a))), a + 180, 1) for e, a in zip(entA, STATIONS)]
for arm, hc in zip([B] + A, handcams):
    arm.hand = vi.attach_hand(hc, arm.g)
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
if cams:      # 手首のカメラの映像も保存する（4 step に 1 コマ＝25 fps。録画と同じ）
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


def lifted(arm, d_before):
    """持ち上げた後、手首のカメラで見た箱までの距離が、持ち上げる前と変わらないか（箱が手と一緒に上がった）。
    指の角度だけでは、箱が指のあいだに残ったまま床に置き去りになったのを見分けられなかった（重さを 0〜5 g と測った）"""
    d_after = vi.hand_depth(arm.hand)
    return holding(arm) and (d_after - d_before) < 0.008, d_after - d_before


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
        c = relook(seen[:2])                     # 近づく前に見直す（ほかの箱を処理している間に動いたかもしれない）
        seen = c if c is not None else seen
        ans = GEMINI.get(id(b["seen"])) if GEMINI else None
        sturdy = ans is not None and not ans.get("fragile", True)
        tau, step = (CAM_TAU0, CAM_STEP) if sturdy else (TAU0, TAU_STEP)
        mark(f"1-{i}", name, f"試し始め τ {tau:.2f}、{step} ずつ" +
             (f"（カメラ：{ans['looks_like']}・{'壊れにくい' if sturdy else '壊れやすい'}・{ans['weight_g']} g）" if ans else "（カメラなし）"))
        az, r, arm.dyaw[:] = aim(arm, seen)
        arm.set_grip_tau(s.TAU_MAX)
        yield from move(arm, az, r, Z_CARRY_A, s.CLOSE, 150)
        v = []
        for _ in range(100):
            yield
            v.append(arm.e.get_dofs_control_force().cpu().numpy()[0, 1])
        tau_empty = float(np.mean(v))
        yield from move(arm, az, r, 0.08, s.OPEN, 100)
        yield from move(arm, az, r, zg, s.OPEN, 80)
        if os.environ.get("SOFT_DEBUG"):
            c = arm.e.get_contacts()
            names = {l.idx: l.name for l in arm.e.links}
            for x in boxes:
                for l in x["ent"].links: names[l.idx] = f"{x['name']}:{l.name}"
            pairs = sorted({(names.get(int(u), int(u)), names.get(int(v), int(v))) for u, v, m in zip(c["link_a"].cpu().numpy()[0], c["link_b"].cpu().numpy()[0], c["valid_mask"].cpu().numpy()[0]) if m})
            print(f"      DEBUG 下ろした後 {name}: 目印 {np.round(arm.tip()[0], 3).tolist()} 関節 {np.round(arm.e.get_dofs_position().cpu().numpy()[0], 2).tolist()} 触れている {pairs}", flush=True)
        yield from move(arm, az, r, zg, s.PRE, 40)
        cap, strength, found, tries = TAU_MEAS, None, None, 0
        while tau <= cap + 1e-9:
            arm.set_grip_tau(tau)
            yield from move(arm, az, r, zg, s.CLOSE, 200)
            d_before = vi.hand_depth(arm.hand)
            yield from move(arm, az, r, zg + LIFT_DZ, s.CLOSE, 60)
            yield from move(arm, az, r, zg + LIFT_DZ, s.CLOSE, 30)
            tries += 1
            ok, dd = lifted(arm, d_before)
            if ok:                    # 箱が手と一緒に上がった（手首のカメラで、箱までの距離が変わらない）
                found = tau
                break
            # 持ち上がらなかった：その場で指を開き、手を根元の側へ退かせてから、固定カメラで箱を見直して下ろし直す
            # （閉じたまま下ろすと、指のあいだから抜け落ちた箱の上に、閉じきった指が乗るだけになった。真上から見るので、
            #   手が箱の上にあると箱が隠れる）
            arm.set_grip_tau(s.TAU_MAX)
            yield from move(arm, az, r, zg + LIFT_DZ, s.OPEN, 50)
            yield from move(arm, az, r - 0.08, 0.10, s.OPEN, 60)
            c = relook(seen[:2])
            seen = c if c is not None else seen
            az, r, arm.dyaw[:] = aim(arm, seen)
            yield from move(arm, az, r, 0.08, s.OPEN, 60)
            yield from move(arm, az, r, zg, s.OPEN, 60)
            yield from move(arm, az, r, zg, s.PRE, 40)
            tau = round(tau + step, 3)
        if found is None:
            mark(f"1-{i}", name, "安全な力の範囲では持てない → 渡さない")
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
        if strength is None:
            strength = sb.strength_of(k_from_angle(float(np.mean(qs[-30:])), m_hat))
        cap = SAFE * strength / SQ_PER_TAU
        hold = min(found * MARGIN, max(cap, found))
        b.update(found=found, m_hat=m_hat, strength_est=strength, tries=tries)
        mark(f"1-{i}", name, f"力加減 {found:.2f}（{tries} 回）、重さ {m_hat * 1000:.0f} g（本当 {b['mass'] * 1000:.0f}）、"
                              f"強度 {strength:.1f} N（本当 {b['strength']:.1f}）、摩擦 {b['mu']}")
        arm.set_grip_tau(hold)
        if hold / found < MIN_MARGIN:
            # 重さに対して弱すぎる：潰さない力では、運ぶ余裕が足りない（余裕 1.27 倍では、ゆっくり運んでも台座の手前で落とした）。
            # 運ばずに元の場所へ置き直し、人の確認に回す。強度を測らずに運べば、潰すか落とすかしていた箱
            mark(f"1-{i}", name, f"重さに対して弱すぎる（余裕 {hold / found:.2f} 倍）→ 運ばずに置き直し、人の確認へ")
            b["set_aside"] = True
            yield from move(arm, az, r, zg + 0.002, s.CLOSE, 120)
            arm.set_grip_tau(s.TAU_MAX)
            yield from move(arm, az, r, zg + 0.002, s.OPEN, 50)
            yield from move(arm, az, r, 0.12, s.OPEN, 80)
            continue
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
        slow = max(1.0, (MARGIN / (hold / found)) ** 2)    # 強度の上限で余裕を 2 倍まで足せない箱は、足りない分だけゆっくり（その 2 と同じ）
        n_carry = int(150 * (1 + m_hat / 0.15) * slow)
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
        tau_B = min(found * MARGIN, max(cap, found))
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
            yield from move(B, az_b, r_b, zs, s.OPEN, 100)
            yield from move(B, az_b, r_b, zs, s.PRE, 40)
            B.set_grip_tau(tau_B)
            yield from move(B, az_b, r_b, zs, s.CLOSE, 200)
            q_grip = float(B.e.get_dofs_position().cpu().numpy()[0, 5])
            d_before_b = vi.hand_depth(B.hand)
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
            if GRIP_OK[0] < q_grip < GRIP_OK[1]:
                yield from move(B, az_b, r_b, zs + 0.02, s.CLOSE, 40)
                ok_b, _ = lifted(B, d_before_b)
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
        slow = max(1.0, (MARGIN / margin_eff) ** 2)
        n_carry = int(150 * (1 + m_hat / 0.15) * slow)
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
summary = [{k: b.get(k) for k in ("name", "look", "mass", "k", "strength", "mu", "found", "m_hat", "strength_est", "tries", "crushed", "inside")}
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
