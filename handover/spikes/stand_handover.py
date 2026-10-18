"""試し：置き台を使う受け渡し（SO-101 × 2 台）。

A（根元が原点）：床のキューブ（A から見て -45°、22 cm）を掴み、持ち上げて止めて重さを測り、置き台（真ん中）に置く。
B（根元が x=0.44、A に向けて 180° 回す）：置き台から掴み、B から見て +45° の大箱（上の開いた箱）へ入れる。
握る力は、指のモーターの力の上限（GRIP_TAU [N·m]）で決める。指がキューブを押す力は接触の情報から読む。
指の当たり方（handover/spikes/jaw_geometry.py で測った）：目印 gripperframe は固定側の指の内側の先端。
キューブの中心は、目印から腕の外向き（根元から遠ざかる向き）に 1.9 cm。

使い方（答え合わせ用リポジトリの直下で）:
  docker compose run --rm -v <SO-ARM100 の置き場>:/m:ro sim python handover/spikes/stand_handover.py [質量g ...]
動画: outputs/handover/stand.mp4（環境 0）
"""
import math
import sys

import numpy as np
import torch
import genesis as gs

MASSES = [float(x) / 1000 for x in sys.argv[1:]] or [0.03, 0.15]
GRIP_TAU = float(__import__('os').environ.get('GRIP_TAU', 0.5))                # 指のモーターの力の上限 [N·m]（= 握る力の設定）
N = len(MASSES)
MODEL = "/m/Simulation/SO101/so101_new_calib.xml"
D = 0.44                      # 2 台の根元の距離
R = 0.22                      # 掴む・置く半径（真下向きで届く 16〜28 cm の真ん中）
SIDE = 0.03
STAND_H = 0.02
OFF = np.array([-0.0079, -0.000218, -0.0981])  # gripper リンクから gripperframe まで
JAW = 0.019                   # 目印からキューブの中心まで（腕の外向き）
OPEN, CLOSE = 1.2, -0.17

gs.init(backend=gs.cpu, logging_level="warning")
sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
              rigid_options=gs.options.RigidOptions(box_box_detection=True, friction_cone=gs.friction_cone.elliptic,
                                                    impratio=10.0, batch_links_info=True),
              vis_options=gs.options.VisOptions(rendered_envs_idx=[0]), show_viewer=False)
sc.add_entity(gs.morphs.Plane())
armA = sc.add_entity(gs.morphs.MJCF(file=MODEL, pos=(0, 0, 0), decompose_robot_error_threshold=0.0),
                     material=gs.materials.Rigid(friction=1.0))
armB = sc.add_entity(gs.morphs.MJCF(file=MODEL, pos=(D, 0, 0), euler=(0, 0, 180), decompose_robot_error_threshold=0.0),
                     material=gs.materials.Rigid(friction=1.0))
sc.add_entity(gs.morphs.Box(size=(0.05, 0.05, STAND_H), pos=(D / 2, 0, STAND_H / 2), fixed=True),
              surface=gs.surfaces.Default(color=(0.6, 0.6, 0.65)))
az0 = math.radians(-45)
cube = sc.add_entity(gs.morphs.Box(size=(SIDE,) * 3, pos=(R * math.cos(az0), R * math.sin(az0), SIDE / 2)),
                     material=gs.materials.Rigid(friction=1.0, rho=1000.0),
                     surface=gs.surfaces.Default(color=(0.9, 0.35, 0.15)))
# 大箱：B から見て +45°、半径 R。上の開いた箱（底と 4 枚の壁、壁の高さ 4 cm）
bx, by = D + R * math.cos(math.pi + math.radians(45)), R * math.sin(math.pi + math.radians(45))
for dx, dy, sx, sy in [(0.05, 0, 0.004, 0.10), (-0.05, 0, 0.004, 0.10), (0, 0.05, 0.10, 0.004), (0, -0.05, 0.10, 0.004)]:
    sc.add_entity(gs.morphs.Box(size=(sx, sy, 0.04), pos=(bx + dx, by + dy, 0.02), fixed=True),
                  surface=gs.surfaces.Default(color=(0.75, 0.6, 0.4)))
cam = sc.add_camera(res=(1280, 720), pos=(D / 2, -0.42, 0.20), lookat=(D / 2, 0.0, 0.05), fov=50, GUI=False)
sc.build(n_envs=N)
cube.set_mass(torch.tensor(MASSES, dtype=gs.tc_float))


def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_of(Rm):
    """回転の行列 → クォータニオン (w, x, y, z)。どの向きでも 0 で割らない形（対角の大きい成分から求める）。"""
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


class Arm:
    """真下向きの手先を、根元から見た方位 az・半径 r・高さ z へ、S 字で少しずつ動かす。"""

    def __init__(self, ent, base_xy, base_yaw):
        self.e, self.g = ent, ent.get_link("gripper")
        self.base, self.yaw = np.array(base_xy, float), base_yaw
        ent.set_dofs_kp(np.full(6, 998.22))
        ent.set_dofs_kv(np.full(6, 2.731))
        lim = np.array([2.94, 2.94, 2.94, 2.94, 2.94, GRIP_TAU])  # 指だけ、力の上限を握る力の設定にする
        ent.set_dofs_force_range(-lim, lim)
        ent.set_dofs_position(np.zeros((N, 6)))
        self.R0 = rot(self.g.get_quat().cpu().numpy()[0])  # 関節 0 での手先の向き（step 前に読む）
        self.last = None
        self.grip = OPEN

    def world(self, az, r, z):  # 根元から見た (方位, 半径, 高さ) → 世界の座標
        a = self.yaw + az
        return np.array([self.base[0] + r * math.cos(a), self.base[1] + r * math.sin(a), z])

    def Rd(self, az):  # 真下向き、根元から見た方位 az を向く
        # 「手先を下へ倒す」回転は、腕自身の向き（yaw）で見た横の軸のまわりに回す。世界の y 軸で回すと、
        # 180° 回して置いた腕では逆向きに倒れて、手先が横を向く（その 2 の指が小箱を払い飛ばした）
        return Rz(az) @ Rz(self.yaw) @ Ry(math.pi / 2) @ Rz(-self.yaw) @ self.R0

    def tip(self):
        P, Q = self.g.get_pos().cpu().numpy(), self.g.get_quat().cpu().numpy()
        return np.array([P[i] + rot(Q[i]) @ OFF for i in range(N)])


A = Arm(armA, (0, 0), 0.0)
B = Arm(armB, (D, 0), math.pi)
FOLD = np.array([0.0, -1.7, 1.6, 1.2, 0.0, 0.0])
for arm in (A, B):
    arm.e.set_dofs_position(np.tile(FOLD, (N, 1)))
    arm.e.control_dofs_position(np.tile(FOLD, (N, 1)))
for _ in range(30):
    sc.step()


def move(plan, n):
    """plan = [(腕, 方位, 半径, 高さ, 指)]。全部の腕を同時に、n step かけて動かす。"""
    starts = {}
    for arm, az, r, z, g in plan:
        tgt = arm.world(az, r, z)
        start = arm.last if arm.last is not None else arm.tip()[0]  # 最初の 1 回も、いまの手先から少しずつ動かす
        starts[id(arm)] = (start, tgt, az, g)
    for k in range(n):
        s = min(1.0, (k + 1) / (0.7 * n))
        s = 3 * s ** 2 - 2 * s ** 3
        for arm, *_ in plan:
            st, tgt, az, g = starts[id(arm)]
            if k % 5 == 0 or k == n - 1:
                cur = st + (tgt - st) * s
                Rd = arm.Rd(az)
                q = arm.e.inverse_kinematics(link=arm.g, pos=np.tile(cur - Rd @ OFF, (N, 1)), quat=np.tile(quat_of(Rd), (N, 1)),
                                             init_qpos=arm.e.get_dofs_position(), max_solver_iters=50).cpu().numpy()
                q[:, 5] = g
                arm.e.control_dofs_position(q)
        sc.step()
    for arm, az, r, z, g in plan:
        arm.last = arm.world(az, r, z)


def close_gently(arm, plan_pose, tau):
    """指を 2 段階で閉じる：①力の上限を大きくして、触れる手前（角度 0.25）まで速く閉じる ②力の上限を tau に絞って閉じきる。
    力の上限が小さいままだと、指の関節の減衰で閉じるのが遅い（0.5 N·m では 0.8 秒で半分しか閉じない）。"""
    ent = arm.e
    lim = np.array([2.94] * 6)
    ent.set_dofs_force_range(-lim, lim)
    move([plan_pose[:4] + (0.25,)], 40)
    lim[5] = tau
    ent.set_dofs_force_range(-lim, lim)
    move([plan_pose[:4] + (CLOSE,)], 80)


def squeeze(arm_ent):
    """指がキューブを押している力 [N]（接触の力の大きさの合計を、左右の指で半分に）。形は (環境の数,)。"""
    c = arm_ent.get_contacts(with_entity=cube)
    f = np.linalg.norm(c["force_a"].cpu().numpy(), axis=-1) * c["valid_mask"].cpu().numpy()
    return f.sum(axis=1) / 2


cam.start_recording(save_to_filename="/work/outputs/handover/stand.mp4", fps=25)
zg = 0.020                                    # 床のキューブを掴む高さ（目印）
zs = STAND_H + 0.020                          # 置き台の上のキューブを掴む高さ
hold = (A, az0, R - JAW, 0.10, CLOSE)
# A：床のキューブを掴む
move([(A, az0, R - JAW, 0.10, OPEN)], 150)
move([(A, az0, R - JAW, zg, OPEN)], 150)
print("A tip after down", A.tip().round(3), " cube", cube.get_pos().cpu().numpy().round(3), " target", A.world(az0, R - JAW, zg).round(3))
close_gently(A, (A, az0, R - JAW, zg, CLOSE), GRIP_TAU)
print("A grip q after close", armA.get_dofs_position().cpu().numpy()[:, 5].round(2), " squeeze", squeeze(armA).round(1))
move([hold], 150)
move([hold], 50)
taus = []
for _ in range(100):  # 止めて 1 秒、肩の関節の力を平均する
    sc.step()
    taus.append(armA.get_dofs_control_force().cpu().numpy()[:, 1])
tau = np.mean(taus, axis=0)
print("A lifted cube z", cube.get_pos().cpu().numpy()[:, 2].round(3), " shoulder tau", tau.round(4), " squeeze N", squeeze(armA).round(1))
# A：置き台へ（置き台の中心は A から見て方位 0、半径 D/2）
move([(A, 0.0, D / 2 - JAW, 0.10, CLOSE)], 150)
move([(A, 0.0, D / 2 - JAW, zs + 0.002, CLOSE)], 120)
move([(A, 0.0, D / 2 - JAW, zs + 0.002, OPEN)], 60)
move([(A, 0.0, D / 2 - JAW, 0.12, OPEN)], 100)
print("on stand cube", cube.get_pos().cpu().numpy().round(3))
# B：置き台の上の小箱の、実際の位置を目指す（作品ではカメラで見る役目。ここではシミュレータの位置で代える）
cp = cube.get_pos().cpu().numpy()[0]
rel = cp[:2] - B.base
az_b = math.atan2(rel[1], rel[0]) - B.yaw
az_b = (az_b + math.pi) % (2 * math.pi) - math.pi
r_b = float(np.linalg.norm(rel)) - JAW
print(f"B aims at cube: az {math.degrees(az_b):.1f} deg, tip radius {r_b:.3f}")
move([(A, -1.2, 0.18, 0.15, OPEN), (B, az_b, r_b, 0.10, OPEN)], 150)
move([(B, az_b, r_b, zs, OPEN)], 120)
close_gently(B, (B, az_b, r_b, zs, CLOSE), GRIP_TAU)
move([(B, az_b, r_b, 0.10, CLOSE)], 120)
print("B lifted cube z", cube.get_pos().cpu().numpy()[:, 2].round(3), " squeeze N", squeeze(armB).round(1))
move([(B, math.radians(45), R - JAW, 0.10, CLOSE)], 150)
move([(B, math.radians(45), R - JAW, 0.07, CLOSE)], 120)   # 大箱の壁（4 cm）の上から
move([(B, math.radians(45), R - JAW, 0.07, OPEN)], 60)
move([(B, math.radians(45), R - JAW, 0.12, OPEN)], 80)
cam.stop_recording()
goal = B.world(math.radians(45), R, SIDE / 2)
p = cube.get_pos().cpu().numpy()
for i, m in enumerate(MASSES):
    inside = abs(p[i, 0] - bx) < 0.046 and abs(p[i, 1] - by) < 0.046 and p[i, 2] < 0.04
    print(f"{m * 1000:5.0f} g  final {p[i].round(3)}  {'IN BIG BOX' if inside else 'outside'}")
