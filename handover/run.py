"""作品の本体：段ボールの小箱を、SO-101 その 1 が持って力加減と重さを測り、台座に置く。その 2 がその情報で持ち、大箱に入れる。

  1. その 1：弱い力から少しずつ強めて試し掴み（持ち上がる最小の力 = 力加減）。その力で持って止め、肩の関節の力から重さを出す
  2. その 1：台座に置く
  3. その 2：その 1 の情報（力加減）に余裕を足した力で、台座から持つ
  4. その 2：大箱に入れる
  段ボールは、指が押す力（閉じた瞬間の勢いを含む）が CRUSH_N を超えたら「潰れた」とする（変形は計算しない、決まりで判定）。

使い方（答え合わせ用リポジトリの直下で）:
  docker compose run --rm -v <SO-ARM100>:/m:ro sim python handover/run.py [--masses 30,150,300] [--b data|weak|strong] [--a search|strong]
  --a search : その 1 は弱い力から強める（既定）   --a strong : その 1 はいきなり最大の力で握る（わざと失敗：潰す）
  --b data   : その 2 はその 1 の力加減を使う（既定） --b weak : 情報なしで弱く握る（わざと失敗：重い箱を落とす）
  --b strong : 情報なしで最大の力で握る（わざと失敗：潰す）
  --record outputs/handover/run.mp4 [--env 0] : 環境 1 つを録画する
  --mistake <名前> : 作るときに実際に起きた失敗を再現する（--mistake list で一覧）
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101 as s  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--masses", default="30,150,300", help="小箱の中身の重さ [g]（環境ごと）")
ap.add_argument("--a", choices=["search", "strong"], default="search")
ap.add_argument("--b", choices=["data", "weak", "strong"], default="data")
ap.add_argument("--record")
ap.add_argument("--env", type=int, default=0)
MISTAKES = {
    "convexify": "形を単純化したまま → 指のすき間が埋まって掴めない",
    "jaw_center": "目印を指の真ん中だと思い込む → 固定側の指が小箱の上に乗る",
    "jump": "目標へ一気に動かす → 行き過ぎて、小箱の上に着地する・押しのける",
    "pre_close": "速く閉じる 1 段目を触れる直前（0.25）まで → 閉じた勢いで潰す（2026-10-18 の確認では潰れず、150 g を落とす形になった）",
    "box_world": "小箱を世界の軸のまま置き、向きを見ない → 角を挟んで、待つ間に滑り落ちる",
    "no_reaim": "小箱の位置を見直さずに決め打ちの場所へ下りる → 落とした後の試し掴みがずれる（単独では再現しない：小箱を斜めに置いていた不具合と重なって起きていた。記録のみ）",
    "hold_min": "力加減が見つかっても、ぎりぎりの力のまま待つ → 待つ間に滑り落ちる",
    "one_side": "片側の指の力でも潰れと判定する → 触れただけで「潰れた」と出る（判定の誤り）",
    "carry_high": "高く持ち上げて運ぶ → 手首が可動範囲の端に当たり、重い箱が傾いて落ちる（単独では再現しない：他の不具合を直した後は 300 g も運べた。記録のみ）",
    "small_margin": "運ぶ余裕を小さく（1.25 倍）→ 運ぶ途中で重い箱を落とす",
    "world_tilt": "180° 回したその 2 を世界の軸で下に向ける → 手先が横を向いて小箱を払う",
    "open_from_fold": "その 2 がたたんだ姿勢のまま指を開いて近づく → 開いた指が台座の小箱を払う（単独では再現しない：他の不具合と重なって起きていた。記録のみ）",
}
ap.add_argument("--mistake", help="わざと失敗の名前（list で一覧）")
args = ap.parse_args()
if args.mistake == "list":
    for k, v in MISTAKES.items():
        print(f"  {k:15s} {v}")
    raise SystemExit
if args.mistake and args.mistake not in MISTAKES:
    raise SystemExit(f"知らないわざと失敗: {args.mistake}（--mistake list で一覧）")
MIS = args.mistake
s.MISTAKE = MIS
if MIS == "jaw_center":
    s.JAW = 0.0

MASSES = np.array([float(x) / 1000 for x in args.masses.split(",")])
N = len(MASSES)
CRUSH_N = 15.0                       # 段ボールが潰れる、指が押す力 [N]
TAU_STEPS = [0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8]  # その 1 が試す力の設定 [N·m]（弱い順）
MARGIN = 1.25 if MIS == "small_margin" else 2.0                         # 運ぶときは、力加減にこの倍率の余裕を足す
SQ_PER_TAU = 14.5                    # 挟む力 [N] ÷ 指の力の設定 [N·m]（handover/spikes/grip_window.py で測った比例）
TAU_SAFE = 0.7 * CRUSH_N / SQ_PER_TAU   # 潰れの上限の 7 割の力。余裕を足しても、ここを超えない
                                        # （8 割では、300 g を持ち上げて止めている間の揺れで 15.7 N になった）
WEAK, STRONG = 0.2, s.TAU_MAX
D, R, SIDE, STAND_H = 0.44, 0.22, 0.03, 0.02
az0 = math.radians(-45)              # 床の小箱の方位（その 1 から見て）
LIFT_DZ = 0.02                       # 試し掴みで持ち上げる高さ
Z_CARRY_A, Z_CARRY_B = (0.10, 0.10) if MIS == "carry_high" else (0.06, 0.07)    # 運ぶ高さ（目印）。根元から 20 cm で 8 cm 以上に上げると、手首が可動範囲の端に張り付いて
                                     # 手先を真下に向けきれず、小箱が傾いて重い箱が滑り落ちた

sc = s.make_scene(args.env)
sc.add_entity(s.gs.morphs.Plane())
entA = s.add_so101(sc, (0, 0), 0)
entB = s.add_so101(sc, (D, 0), 180)
sc.add_entity(s.gs.morphs.Box(size=(0.05, 0.05, STAND_H), pos=(D / 2, 0, STAND_H / 2), fixed=True),
              surface=s.gs.surfaces.Default(color=(0.6, 0.6, 0.65)))
bx, by = D + R * math.cos(math.radians(225)), R * math.sin(math.radians(225))   # 大箱（その 2 から見て +45°）
for dx, dy, sx, sy in [(0.05, 0, 0.004, 0.10), (-0.05, 0, 0.004, 0.10), (0, 0.05, 0.10, 0.004), (0, -0.05, 0.10, 0.004)]:
    sc.add_entity(s.gs.morphs.Box(size=(sx, sy, 0.04), pos=(bx + dx, by + dy, 0.02), fixed=True),
                  surface=s.gs.surfaces.Default(color=(0.75, 0.6, 0.4)))
box = sc.add_entity(s.gs.morphs.Box(size=(SIDE,) * 3, pos=(R * math.cos(az0), R * math.sin(az0), SIDE / 2),
                                    euler=(0, 0, 0.0 if MIS == "box_world" else math.degrees(az0))),   # その 1 に面を向けて置く（世界の軸のままだと対角線を挟む）
                    material=s.gs.materials.Rigid(friction=1.0, rho=1000.0),
                    surface=s.gs.surfaces.Default(color=(0.72, 0.55, 0.36)))   # 段ボールの色。見た目は全部同じ
cam = None
if args.record:
    cam = sc.add_camera(res=(1280, 720), pos=(D / 2, -0.42, 0.20), lookat=(D / 2, 0.0, 0.05), fov=50, GUI=False)
sc.build(n_envs=N)
box.set_mass(torch.tensor(MASSES, dtype=s.gs.tc_float))
A = s.Arm(sc, entA, (0, 0), 0, N)
B = s.Arm(sc, entB, (D, 0), 180, N)
for _ in range(30):
    sc.step()
if cam:
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    cam.start_recording(save_to_filename=args.record, fps=25)

peak = np.zeros(N)       # 指が押した力の最大 [N]（両腕・全工程）
crushed = np.zeros(N, bool)
PHASE = "start"
crush_at = [""] * N      # 潰れた工程（どこで潰したかを記録する）
recent = []              # 直近 5 step の挟む力（潰れは 0.05 秒の平均で判定する）


def watch(k):
    """毎 step、指が小箱を挟む力を見張る。0.05 秒の平均が CRUSH_N を超えたら潰れた（以後ずっと潰れたまま）。"""
    global peak, crushed
    f = np.maximum(s.squeeze(A, box), s.squeeze(B, box))
    recent.append(f)
    del recent[:-5]
    fm = np.mean(recent, axis=0)  # 0.05 秒の平均。滑り落ちる瞬間の 1 step だけの計算の飛びは、潰れに数えない
    peak = np.maximum(peak, fm)
    for i in np.where((fm > CRUSH_N) & ~crushed)[0]:
        crush_at[i] = f"{PHASE} step {k} ({fm[i]:.1f} N)"
    crushed |= fm > CRUSH_N


def box_z():
    return box.get_pos().cpu().numpy()[:, 2]


def aim(arm):
    """小箱の実際の位置と向きを見て、arm の目印を置く (方位, 半径) と、指を回す角度を環境ごとに返す（作品ではカメラで見る役目。
    ここではシミュレータの位置と向きで代える）。落としたあとは位置も向きも変わるので、試すたびに見直す。
    指は小箱の面に合わせて回す（四角なので 90° ごとに同じ → ±45° に収める）。目印は、回した指が閉じる向きに JAW だけ手前"""
    cp = box.get_pos().cpu().numpy()
    q = box.get_quat().cpu().numpy()
    box_yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    if MIS == "box_world":  # 向きを見ない：指は根元からの方位のまま、目印は方位に沿って JAW 手前
        box_yaw = np.full(arm.n, np.nan)
    tip = cp[:, :2].copy()
    for _ in range(3):  # 方位と指の角度は互いに決まるので、3 回くり返して落ち着かせる
        rel = tip - arm.base
        az = np.arctan2(rel[:, 1], rel[:, 0]) - arm.yaw
        d = box_yaw - (arm.yaw + az)
        dyaw = np.nan_to_num((d + math.pi / 4) % (math.pi / 2) - math.pi / 4, nan=0.0)
        close_dir = arm.yaw + az + dyaw  # 指が閉じる向き（根元から外向き）
        tip = cp[:, :2] - s.JAW * np.stack([np.cos(close_dir), np.sin(close_dir)], axis=1)
    rel = tip - arm.base
    az = (np.arctan2(rel[:, 1], rel[:, 0]) - arm.yaw + math.pi) % (2 * math.pi) - math.pi
    return az, np.linalg.norm(rel, axis=1), dyaw


def box_tilt():
    """小箱の傾き [度]（上の面が真上からどれだけ倒れたか）。指先に角で引っかかってぶら下がると大きくなる（L1-c と同じ）。"""
    q = box.get_quat().cpu().numpy()
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))


# ---- 1. その 1：試し掴みで力加減を探し、重さを測る ----
zg = 0.020
s.move(sc, [(A, az0, R - s.JAW, 0.10, s.OPEN)], 150, watch)
s.move(sc, [(A, az0, R - s.JAW, zg, s.OPEN)], 150, watch)
found = np.full(N, np.nan)                  # 持ち上がった最小の力の設定
hold_tau = np.zeros(N)                      # 見つかった環境が、持って待つあいだの力の設定（運ぶ力）


def carry_tau(f):
    """運ぶ力 = 力加減 × 余裕。ただし潰れの上限の手前（TAU_SAFE）を超えない。重い箱ほど、余裕を足せる幅が狭い。"""
    return np.minimum(f * MARGIN, np.maximum(TAU_SAFE, f))
tries = np.zeros(N, int)
steps = [STRONG] if args.a == "strong" else TAU_STEPS
az_a, r_a = np.full(N, az0), np.full(N, R - s.JAW)    # その 1 の目印を置く所（環境ごと）
for tau in steps:
    PHASE = f"1-try tau {tau}"
    todo = np.isnan(found) & ~crushed
    if not todo.any():
        break
    t_now = np.where(todo, tau, hold_tau)   # 見つかった環境は、運ぶ力で持ったまま待つ
    if MIS != "no_reaim":  # 毎回（1 回目も）：小箱の位置を見直して、その真上から下り直す（持っている環境は動かさない）
        # 1 回目を決め打ちの場所に下ろすと、数 mm ずれて小箱の上の角を挟み（指の角度 0.37）、持って待つ間に滑り落ちた
        az_new, r_new, dy_new = aim(A)
        az_a, r_a = np.where(todo, az_new, az_a), np.where(todo, r_new, r_a)
        A.dyaw = np.where(todo, dy_new, A.dyaw)
        s.move(sc, [(A, az_a, r_a, np.where(todo, 0.08, zg + LIFT_DZ), np.where(todo, s.OPEN, s.CLOSE))], 80, watch)
        s.move(sc, [(A, az_a, r_a, np.where(todo, zg, zg + LIFT_DZ), np.where(todo, s.OPEN, s.CLOSE))], 80, watch)
    PHASE = f"1-close tau {tau}"
    s.close_gently(sc, A, az_a, r_a, np.where(todo, zg, zg + LIFT_DZ), t_now, watch, mask=todo)
    PHASE = f"1-lift tau {tau}"
    s.move(sc, [(A, az_a, r_a, zg + LIFT_DZ, s.CLOSE)], 60, watch)
    s.move(sc, [(A, az_a, r_a, zg + LIFT_DZ, s.CLOSE)], 30, watch)
    up = (box_z() > SIDE / 2 + LIFT_DZ * 0.5) & (box_tilt() < 10.0)  # 持ち上がって、かつ傾いていない
    tries += todo
    found = np.where(todo & up, tau, found)
    newly = todo & up                        # 見つかったら、すぐ運ぶ力に上げる（ぎりぎりの力で長く待つと滑り落ちる）
    hold_tau = np.where(newly, np.nan_to_num(found, nan=tau) if MIS == "hold_min" else carry_tau(np.nan_to_num(found, nan=tau)), hold_tau)
    A.set_grip_tau(np.where(todo & ~up, tau, hold_tau))
    retry = todo & ~up                       # 持ち上がらなかった環境は、下ろして開き、次の力で
    PHASE = f"1-release tau {tau}"
    if retry.any():  # その場で指を開いてから下ろす（下ろしながら開くと、床に戻った小箱を挟んだまま押し付ける）
        A.set_grip_tau(np.where(retry, s.TAU_MAX, hold_tau))   # 持って待っている環境は、運ぶ力のまま
        s.move(sc, [(A, az_a, r_a, zg + LIFT_DZ, np.where(retry, s.OPEN, s.CLOSE))], 50, watch)
print("[1] force search:", ", ".join(f"{m * 1000:.0f} g -> {'CRUSHED' if c else (f'{f:.2f} N·m' if not np.isnan(f) else 'not lifted')}"
                                   f" ({t} tries)" for m, f, c, t in zip(MASSES, found, crushed, tries)))
# その 1 自身も、見つけた力に余裕を足して運ぶ（試し掴みの 2 cm を静かに、でぎりぎり持てた力のままだと、高く速く運ぶと落ちる）
A.set_grip_tau(np.where(np.isnan(found), WEAK, hold_tau))
s.move(sc, [(A, az_a, r_a, zg + LIFT_DZ, s.CLOSE)], 40, watch)
PHASE = "1-weigh"
# 重さ：持ち上げて止め、肩の関節の力を 1 秒平均。腕だけのときの力との差を、てこの長さ × g で割る
s.move(sc, [(A, az_a, r_a, Z_CARRY_A, s.CLOSE)], 100, watch)
taus = []
for _ in range(100):
    sc.step()
    watch(0)
    taus.append(entA.get_dofs_control_force().cpu().numpy()[:, 1])
tau_hold = np.mean(taus, axis=0)
lever = np.linalg.norm(box.get_pos().cpu().numpy()[:, :2] - entA.get_link("upper_arm").get_pos().cpu().numpy()[:, :2], axis=1)
TAU_EMPTY = -0.400                           # 手に何も持たないとき（handover/spikes/grasp_weigh.py の直線の切片）
m_hat = (TAU_EMPTY - tau_hold) / (9.81 * lever)
held_A = box_z() > Z_CARRY_A - 0.03
print("[1] weight:", ", ".join(f"{m * 1000:.0f} g -> {mh * 1000:.0f} g" if h else f"{m * 1000:.0f} g -> (not held)"
                               for m, mh, h in zip(MASSES, m_hat, held_A)))

# 運ぶ速さ：重い箱ほどゆっくり（揺れと振られる力を小さくする）。測った重さで決める。全部の環境で同じ時間なので、いちばん重い箱に合わせる
n_carry = int(150 * (1 + np.nanmax(np.where(held_A, m_hat, 0.0)) / 0.15))
print(f"[1] carry time {n_carry / 100:.1f} s (heaviest measured {np.nanmax(np.where(held_A, m_hat, 0.0)) * 1000:.0f} g)")

# ---- 2. その 1：台座に置く ----
PHASE = "2-place"
zs = STAND_H + 0.020
s.move(sc, [(A, 0.0, D / 2 - s.JAW, Z_CARRY_A, s.CLOSE)], n_carry, watch)
s.move(sc, [(A, 0.0, D / 2 - s.JAW, zs + 0.002, s.CLOSE)], 120, watch)
s.move(sc, [(A, 0.0, D / 2 - s.JAW, zs + 0.002, s.OPEN)], 60, watch)
s.move(sc, [(A, 0.0, D / 2 - s.JAW, 0.12, s.OPEN)], 100, watch)
on_stand = (box_z() > STAND_H + SIDE / 2 - 0.005) & (np.linalg.norm(box.get_pos().cpu().numpy()[:, :2] - [D / 2, 0], axis=1) < 0.03)
print("[2] on stand:", on_stand.astype(int).tolist(), " box", box.get_pos().cpu().numpy().round(3).tolist())

# ---- 3. その 2：台座から持つ（力はその 1 の情報で決める） ----
PHASE = "3-B pick"
if args.b == "data":
    tau_B = carry_tau(np.nan_to_num(found, nan=WEAK))
else:
    tau_B = np.full(N, WEAK if args.b == "weak" else STRONG)
az_b, r_b, B.dyaw = aim(B)   # 台座の上の小箱の位置と向きを見て、その 2 の指を合わせる
# 退く・近づくは、高い所（15 cm）を通る。運ぶ途中の指は半分だけ開く（大きく開いた指が、向きを変えながら台座の小箱を払った）
HALF = 0.6
s.move(sc, [(A, 0.0, D / 2 - s.JAW, 0.15, s.OPEN)], 60, watch)
print("[3] before approach box", box.get_pos().cpu().numpy().round(3).tolist())
# その 2 のたたんだ姿勢は、指先が台座のほうへ水平に向き、高さも小箱と同じ。そこで指を開くと、動く側の指が水平に振れて
# 台座の小箱を払う。指は閉じたまま、台座から離れた高い所で手先を真下に向けてから、開いて真上へ動く
if MIS == "open_from_fold":  # わざと失敗：たたんだ姿勢のまま指を開いて、まっすぐ近づく
    s.move(sc, [(A, -1.2, 0.18, 0.15, HALF), (B, az_b, r_b, 0.15, s.OPEN)], 150, watch)
else:
    s.move(sc, [(A, -1.2, 0.18, 0.15, HALF), (B, 0.0, 0.26, 0.20, s.CLOSE)], 150, watch)
    s.move(sc, [(B, az_b, r_b, 0.15, HALF)], 120, watch)
print("[3] after approach box", box.get_pos().cpu().numpy().round(3).tolist())
s.move(sc, [(B, az_b, r_b, 0.15, s.OPEN)], 40, watch)
s.move(sc, [(B, az_b, r_b, zs, s.OPEN)], 150, watch)
s.close_gently(sc, B, az_b, r_b, zs, tau_B, watch)
print("[3] B after close: grip q", entB.get_dofs_position().cpu().numpy()[:, 5].round(2), " squeeze", s.squeeze(B, box).round(1),
      " box", box.get_pos().cpu().numpy().round(3).tolist(), " B tip", B.tip().round(3).tolist())
s.move(sc, [(B, az_b, r_b, Z_CARRY_B, s.CLOSE)], 120, watch)
held_B = box_z() > Z_CARRY_B - 0.03
print("[3] B grip tau:", tau_B.round(2).tolist(), " held:", held_B.astype(int).tolist())

# ---- 4. その 2：大箱に入れる ----
PHASE = "4-B carry"
s.move(sc, [(B, math.radians(45), R - s.JAW, Z_CARRY_B, s.CLOSE)], n_carry, watch)
s.move(sc, [(B, math.radians(45), R - s.JAW, Z_CARRY_B, s.OPEN)], 60, watch)   # 大箱の壁（4 cm）の上で離す
s.move(sc, [(B, math.radians(45), R - s.JAW, 0.12, s.OPEN)], 80, watch)
if cam:
    cam.stop_recording()
p = box.get_pos().cpu().numpy()
inside = (np.abs(p[:, 0] - bx) < 0.046) & (np.abs(p[:, 1] - by) < 0.046) & (p[:, 2] < 0.04)
print(f"a={args.a} b={args.b} mistake={MIS}  crush limit {CRUSH_N:.0f} N")
for i, m in enumerate(MASSES):
    if crushed[i]:
        print(f"   {m * 1000:.0f} g crushed at: {crush_at[i]}")
print(" box    peak squeeze   crushed  in big box  result")
for i, m in enumerate(MASSES):
    ok = inside[i] and not crushed[i]
    print(f"{m * 1000:4.0f} g   {peak[i]:6.1f} N      {'YES' if crushed[i] else 'no ':3s}      {'yes' if inside[i] else 'no '}     "
          f"{'OK' if ok else ('CRUSHED' if crushed[i] else 'DROPPED / MISSED')}")
