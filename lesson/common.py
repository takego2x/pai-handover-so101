"""教材の共通部品。どのレッスンも、ここにある定数と関数を使う。

読む順番:
  1. 定数（重力、部品の大きさ、材質の密度、余裕 15%）
  2. sample_parts : 部品の材質と「本当の μ」を決める（ロボットには見せない正解）
  3. make_scene / add_franka : シミュレータの世界と、ロボットを作る
  4. ik / hand_quat : 手先の位置と向き → 関節の角度
  5. trapezoid : 加速 → 一定速 → 減速 の動き
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import genesis as gs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
from mu_table import load, mu_range  # noqa: E402  文献値の表（data/friction_literature.csv）

G = 9.81                 # 重力加速度 [m/s^2]
SIDE = 0.04              # 部品はすべて 4 cm 角の立方体
DENSITY = {"POM": 1410.0, "ABS": 1050.0, "aluminium": 2700.0, "steel": 7850.0, "NBR": 1200.0}  # [kg/m^3]
POOL = list(DENSITY)     # 第一層に来る材質の候補
SAFETY = 0.85            # 使える摩擦力のうち、計画に使う割合（15% の余裕を残す）
A_CAP = 9.0              # 運ぶときの加速度の上限 [m/s^2]
V_MAX = 1.5              # 運ぶときの速さの上限 [m/s]
F_MAX = 60.0             # グリッパが出せる把持力の上限 [N]（指 1 本）
TRUE_WIDEN = (0.7, 1.3)  # 本当の μ は、文献値の幅を 0.7〜1.3 倍に広げた範囲から抽選する
DT = 0.01                # シミュレーションの 1 ステップ [s]
# ロボットの「構えの姿勢」（関節 7 本＋指 2 本）。全環境をここから始める
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
MOTORS, FINGERS = np.arange(7), np.arange(7, 9)


def mass_of(material):
    return DENSITY[material] * SIDE**3


def sample_parts(n, seed0=1000, material=None, stream=0):
    """部品 n 個の材質と本当の μ を決める。material を渡せば全部その材質、None ならランダム。
    本当の μ は文献値の幅の外も含む（文献値を信じても外れる部品を作るため）。"""
    rows = load()
    mats, mus = [], []
    for s in range(seed0, seed0 + n):
        rng = np.random.default_rng(s + 100_000 * stream)
        m = material or POOL[rng.integers(len(POOL))]
        lo, hi, _ = mu_range(m, rows=rows)
        mats.append(m)
        mus.append(float(rng.uniform(lo * TRUE_WIDEN[0], hi * TRUE_WIDEN[1])))
    return mats, np.array(mus)


def make_scene(substeps=10):
    if not getattr(gs, "_initialized", False):  # 1 つのプロセスで場面を 2 つ作るとき（L8）、起動は 1 回だけ
        gs.init(backend=gs.cpu, logging_level="warning")
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=substeps),
        # 楕円錐の摩擦（既定の多角錐だと効く μ が下がり、じわじわ滑る）。batch_links_info は環境ごとに重さを変えるため
        rigid_options=gs.options.RigidOptions(box_box_detection=True, batch_links_info=True,
                                              friction_cone=gs.friction_cone.elliptic, impratio=10.0),
        show_viewer=False)


def add_part(scene, xy, yaw_deg=0.0):
    """部品（4 cm 角）。重さと μ は build の後に環境ごとに設定する（set_parts）。"""
    return scene.add_entity(gs.morphs.Box(size=(SIDE,) * 3, pos=(*xy, SIDE / 2), euler=(0, 0, yaw_deg)),
                            material=gs.materials.Rigid(friction=1.0, rho=1000.0))


def set_parts(part, mats, true_mu):
    """build の後に呼ぶ。環境ごとに、部品の重さ（材質の密度から）と μ を設定し、設定できたかを確かめる。"""
    mass = np.array([mass_of(m) for m in mats])
    part.set_mass(torch.tensor(mass, dtype=gs.tc_float, device=gs.device))
    part.set_friction_ratio(torch.tensor(true_mu, dtype=gs.tc_float, device=gs.device).unsqueeze(1),
                            links_idx_local=[0])
    got = part.get_mass().cpu().numpy()
    assert np.allclose(got, mass, rtol=0.005), f"重さが設定されていない: {got} vs {mass}"
    return mass


def add_franka(scene, xy=(0.0, 0.0), yaw_deg=0.0, finger_mu=0.01):
    """Franka。指の μ は 0.01（接触の μ は「2 物体の大きいほう」なので、部品の μ がそのまま効く）。"""
    return scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(*xy, 0), euler=(0, 0, yaw_deg)),
                            material=gs.materials.Rigid(friction=finger_mu))


def setup_franka(arm, n_envs, home=True):
    arm.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    arm.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    arm.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                             np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
    if home:  # 全環境を同じ姿勢から始める（L3 の「わざと失敗」で、外したときに何が起きるかを見る）
        arm.set_dofs_position(np.tile(HOME, (n_envs, 1)))
        arm.control_dofs_position(np.tile(HOME, (n_envs, 1)))


def hand_quat(yaw, n_envs):
    """手先を真下に向け、指を閉じる向きを yaw [rad] だけ回した姿勢（クォータニオン）。yaw はスカラーか (N,)。"""
    y = np.broadcast_to(np.asarray(yaw, float), (n_envs,))
    return np.column_stack([np.zeros(n_envs), np.cos(y / 2), np.sin(y / 2), np.zeros(n_envs)])


def command(arm, pos, n_envs, force=None, yaw=0.0, seed=None):
    """手先を pos へ向かわせる指令を出す（まだ動かない。scene.step() で動く）。
    force=None なら指を開く。数値か (N,) なら、その力 [N] で指を閉じる。
    seed は IK の出発点の関節角度。None なら「いまの姿勢」から解く。"""
    pos = np.broadcast_to(np.asarray(pos, float), (n_envs, 3))
    q = arm.inverse_kinematics(link=arm.get_link("hand"), pos=pos, quat=hand_quat(yaw, n_envs),
                               init_qpos=arm.get_dofs_position() if seed is None else seed)
    arm.control_dofs_position(q[:, :-2], MOTORS)
    if force is None:
        arm.control_dofs_position(np.full((n_envs, 2), 0.04), FINGERS)
    else:
        f = np.broadcast_to(np.asarray(force, float), (n_envs,))
        arm.control_dofs_force(np.column_stack([-f, -f]), FINGERS)


def trapezoid(dist, a, v=V_MAX):
    """距離 dist を、加速度 a・最高速度 v の台形の速さで動く。時刻 → 進んだ距離 の関数と、かかる時間を返す。"""
    if a <= 0:
        return (lambda t: 0.0), math.inf
    t_acc = v / a
    if a * t_acc**2 >= dist:  # 最高速度に届かない（三角形）
        t_acc, v, t_flat = math.sqrt(dist / a), math.sqrt(dist * a), 0.0
    else:
        t_flat = (dist - a * t_acc**2) / v
    total = 2 * t_acc + t_flat

    def pos(t):
        t = min(max(t, 0.0), total)
        if t < t_acc:
            return 0.5 * a * t * t
        if t < t_acc + t_flat:
            return 0.5 * a * t_acc**2 + v * (t - t_acc)
        return dist - 0.5 * a * (total - t) ** 2

    return pos, total


def carry_force(m, mu, a=A_CAP):
    """重さ m と μ から、加速度 a で持ち上げても余裕 15% で落とさない最小の把持力（指 1 本）。上限 F_MAX。
    保持の条件  m (g + a) <= SAFETY * 2 μ F   を F について解いたもの。"""
    return np.minimum(F_MAX, m * (G + a) / (SAFETY * 2 * mu))


def accel_for(m, mu, F):
    """把持力 F で、余裕 15% を残して出せる加速度（上限 A_CAP）。0 なら静止して持つこともできない。"""
    return np.clip(SAFETY * 2 * mu * F / m - G, 0.0, A_CAP)


def quiet():
    """Genesis の警告を減らす（エラーではないので、結果の行だけを読みやすくする）。"""
    import warnings
    warnings.filterwarnings("ignore")
