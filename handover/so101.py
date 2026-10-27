"""SO-101 を Genesis で動かす共通部品（受け渡しの作品）。

試し（handover/spikes/）で分かったこと：
  - 形は細かく分けて包む（decompose_robot_error_threshold=0.0）。単純化すると指のすき間が埋まる
  - 目印 gripperframe は、固定側の指の内側の先端。小箱の中心は、目印から腕の外向きに JAW
  - モーターは強いばね・弱い減衰（kp 998、kv 2.7）。目標を一度に遠くへ置くと行き過ぎるので、S 字で少しずつ動かす
  - 真下向きで届くのは、根元から 16〜28 cm、低い所
  - 指の力の上限を小さくすると閉じるのが遅いので、速く閉じてから力を絞る（close_gently）
"""
import math
import os

import numpy as np
import torch
import genesis as gs

MODEL = os.environ.get("SO101_MODEL", "/work/third_party/SO-ARM100/Simulation/SO101/so101_new_calib.xml")  # TheRobotStudio/SO-ARM100 a758567
OFF = np.array([-0.0079, -0.000218, -0.0981])  # gripper リンクから gripperframe まで
JAW = 0.019                                    # 目印から小箱の中心まで（腕の外向き）
OPEN, CLOSE = 1.2, -0.17                       # 指の角度（開く／閉じきる）
TAU_MAX = 2.94                                 # モーターの力の上限 [N·m]
FOLD = np.array([0.0, -1.7, 1.6, 1.2, 0.0, 0.0])  # たたんだ姿勢
PRE = 0.5                                      # 速く閉じる 1 段目の止め所（指の角度）
MISTAKE = None  # わざと失敗（run.py の --mistake）。共通部品の中で起きる失敗は、ここで切り替える


def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_of(Rm):
    """回転の行列 → クォータニオン (w, x, y, z)。どの向きでも 0 で割らない形。"""
    t = np.trace(Rm)
    if t > 0:
        S = math.sqrt(t + 1.0) * 2
        q = [0.25 * S, (Rm[2, 1] - Rm[1, 2]) / S, (Rm[0, 2] - Rm[2, 0]) / S, (Rm[1, 0] - Rm[0, 1]) / S]
    elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
        S = math.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
        q = [(Rm[2, 1] - Rm[1, 2]) / S, 0.25 * S, (Rm[0, 1] + Rm[1, 0]) / S, (Rm[0, 2] + Rm[2, 0]) / S]
    elif Rm[1, 1] > Rm[2, 2]:
        S = math.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
        q = [(Rm[0, 2] - Rm[2, 0]) / S, (Rm[0, 1] + Rm[1, 0]) / S, 0.25 * S, (Rm[1, 2] + Rm[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
        q = [(Rm[1, 0] - Rm[0, 1]) / S, (Rm[0, 2] + Rm[2, 0]) / S, (Rm[1, 2] + Rm[2, 1]) / S, 0.25 * S]
    q = np.array(q)
    return q / np.linalg.norm(q)


def Ry(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def Rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def make_scene(env=0):
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu, logging_level="warning")
    return gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
                    rigid_options=gs.options.RigidOptions(box_box_detection=True, friction_cone=gs.friction_cone.elliptic,
                                                          impratio=10.0, batch_links_info=True, batch_dofs_info=True),
                    vis_options=gs.options.VisOptions(rendered_envs_idx=[env]), show_viewer=False)  # 描くのは環境 env だけ


def add_so101(scene, xy=(0, 0), yaw_deg=0.0, friction=1.0, model=None):
    """friction：ロボットの表面の摩擦係数。Genesis では、触れた 2 つの物の摩擦係数の大きいほうが使われるので、
    箱ごとの摩擦の違いを効かせたいときは、ロボットのほうを小さくする（潰れる箱の作品では 0.1）
    model：別のモデルのパス（手首のカメラ付きの公式モデルなど。vision.camera_model）"""
    # わざと失敗 convexify：形を細かく分けずに単純化したまま（固定側の指は手首と 1 つの部品なので、包むとすき間が埋まる）
    opts = {} if MISTAKE == "convexify" else {"decompose_robot_error_threshold": 0.0}
    return scene.add_entity(gs.morphs.MJCF(file=model or MODEL, pos=(*xy, 0), euler=(0, 0, yaw_deg), **opts),
                            material=gs.materials.Rigid(friction=friction))


class Arm:
    """真下向きの手先を、根元から見た (方位 az, 半径 r, 高さ z) へ、S 字で少しずつ動かす。build の後に作る。"""

    def __init__(self, scene, ent, base_xy, base_yaw_deg, n_envs):
        self.sc, self.e, self.g, self.n = scene, ent, ent.get_link("gripper"), n_envs
        self.base, self.yaw = np.array(base_xy, float), math.radians(base_yaw_deg)
        ent.set_dofs_kp(np.full(6, 998.22))
        ent.set_dofs_kv(np.full(6, 2.731))
        self.set_grip_tau(TAU_MAX)
        ent.set_dofs_position(np.zeros((n_envs, 6)))
        self.R0 = rot(self.g.get_quat().cpu().numpy()[0])  # 関節 0 での手先の向き（step 前に読む）
        ent.set_dofs_position(np.tile(FOLD, (n_envs, 1)))
        ent.control_dofs_position(np.tile(FOLD, (n_envs, 1)))
        self.last = None
        self.dyaw = np.zeros(n_envs)   # 指の向きを、方位からさらに回す角度（小箱の向きに合わせる）

    def set_grip_tau(self, tau):
        """指のモーターの力の上限（= 握る力の設定）[N·m]。数値か (環境の数,)。"""
        tau = np.broadcast_to(np.asarray(tau, float), (self.n,))
        lim = np.tile(np.full(6, TAU_MAX), (self.n, 1))
        lim[:, 5] = tau
        self.e.set_dofs_force_range(-lim, lim)

    def world(self, az, r, z):
        a = self.yaw + np.asarray(az, float)
        return np.stack([self.base[0] + r * np.cos(a), self.base[1] + r * np.sin(a), np.broadcast_to(z, np.shape(a))], axis=-1)

    def Rd(self, az, dyaw=0.0):
        # 「手先を下へ倒す」回転は、腕自身の向き（yaw）で見た横の軸のまわりに回す（世界の y 軸で回すと、
        # 180° 回して置いた腕では手先が横を向く）
        if MISTAKE == "world_tilt":  # わざと失敗：世界の y 軸で倒す（180° 回して置いたその 2 では、手先が横を向く）
            return Rz(az + dyaw) @ Ry(math.pi / 2) @ self.R0
        return Rz(az + dyaw) @ Rz(self.yaw) @ Ry(math.pi / 2) @ Rz(-self.yaw) @ self.R0

    def tip(self):
        P, Q = self.g.get_pos().cpu().numpy(), self.g.get_quat().cpu().numpy()
        return np.array([P[i] + rot(Q[i]) @ OFF for i in range(self.n)])


def move(scene, plan, n, every=None):
    """plan = [(腕, 方位, 半径, 高さ, 指の角度)]。方位・半径・高さ・指は数値か (環境の数,)。全部の腕を同時に n step で動かす。
    every(k) を渡すと、step ごとに呼ぶ（rerun への記録や、潰れの見張り）。"""
    st = []
    for arm, az, r, z, g in plan:
        az = np.broadcast_to(np.asarray(az, float), (arm.n,))
        tgt = arm.world(az, np.broadcast_to(r, (arm.n,)), np.broadcast_to(z, (arm.n,)))
        start = arm.last if arm.last is not None else arm.tip()
        st.append((arm, az, start, tgt, np.broadcast_to(np.asarray(g, float), (arm.n,))))
    for k in range(n):
        s = min(1.0, (k + 1) / (0.7 * n))
        s = 3 * s ** 2 - 2 * s ** 3
        if MISTAKE == "jump":  # わざと失敗：目標を一度に置く（強いばね・弱い減衰のモーターは行き過ぎる）
            s = 1.0
        for arm, az, start, tgt, g in st:
            if k % 5 == 0 or k == n - 1:
                cur = start + (tgt - start) * s
                pos = np.zeros((arm.n, 3))
                quat = np.zeros((arm.n, 4))
                for i in range(arm.n):
                    Rd = arm.Rd(az[i], arm.dyaw[i])
                    pos[i] = cur[i] - Rd @ OFF
                    quat[i] = quat_of(Rd)
                q = arm.e.inverse_kinematics(link=arm.g, pos=pos, quat=quat, init_qpos=arm.e.get_dofs_position(),
                                             max_solver_iters=50).cpu().numpy()
                q[:, 5] = g
                arm.e.control_dofs_position(q)
        scene.step()
        if every:
            every(k)
    for arm, az, start, tgt, g in st:
        arm.last = tgt


def close_gently(scene, arm, az, r, z, tau, every=None, mask=None):
    """指を 2 段階で閉じる：①力の上限を大きくして、触れる手前（角度 PRE）まで速く閉じる ②力の上限を tau に絞って閉じきる。
    3 cm の小箱に指が触れるのは角度 0.20〜0.24。PRE を 0.25 にすると、1 段目の勢いで当たって押す力が跳ね上がる（潰れる）。
    mask を渡すと、mask が True の環境だけ閉じ直す。False の環境は、いまの力の設定のまま持ち続ける
    （持って待っている環境に 1 段目の最大の力をかけると、持っている小箱を強く挟み直して潰す）"""
    tau = np.broadcast_to(np.asarray(tau, float), (arm.n,))
    mask = np.ones(arm.n, bool) if mask is None else np.asarray(mask, bool)
    pre = 0.25 if MISTAKE == "pre_close" else PRE  # わざと失敗：触れる直前まで速く閉じる（勢いで当たって潰す）
    arm.set_grip_tau(np.where(mask, TAU_MAX, tau))
    move(scene, [(arm, az, r, z, np.where(mask, pre, CLOSE))], 40, every)
    arm.set_grip_tau(tau)
    move(scene, [(arm, az, r, z, CLOSE)], 200, every)  # 力の設定が小さいと閉じるのが遅いので、2 秒待つ


def squeeze(arm, obj):
    """2 本の指で物を挟んでいる力 [N] = 固定側の指が押す力と、動く側の指が押す力の小さいほう。形は (環境の数,)。
    片側の指が触れただけ（下ろす途中に側面をかすめる など）では物は逃げるので、挟む力には数えない。
    段ボールが潰れるのは、両側から挟んだとき。"""
    c = arm.e.get_contacts(with_entity=obj)
    la, lb = c["link_a"].cpu().numpy(), c["link_b"].cpu().numpy()
    f = np.linalg.norm(c["force_a"].cpu().numpy(), axis=-1) * c["valid_mask"].cpu().numpy()
    fixed, moving = arm.g.idx, arm.e.get_link("moving_jaw_so101_v1").idx
    f_fixed = (f * ((la == fixed) | (lb == fixed))).sum(axis=1)
    f_moving = (f * ((la == moving) | (lb == moving))).sum(axis=1)
    if MISTAKE == "one_side":  # わざと失敗：両指の力を足す（片側の指がかすめただけでも「挟んだ」と数える）
        return f_fixed + f_moving
    return np.minimum(f_fixed, f_moving)
