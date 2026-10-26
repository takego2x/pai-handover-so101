"""試し：手首のカメラ（グリッパの部品に取り付け、指先の方向を見る）に、箱がどう写るか。持ち上がった／抜け落ちたで深度がどう変わるか。"""
import math, sys
from pathlib import Path
import numpy as np, imageio
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import so101 as s, softbox as sb  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_softboxes import surface  # noqa
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/handover/handcam"; OUT.mkdir(parents=True, exist_ok=True)
R = 0.22
sc = s.make_scene(); sc.add_entity(s.gs.morphs.Plane())
ent = s.add_so101(sc, (0, 0), 0, friction=0.1)
cases = [("light", 0.06, 0.8), ("heavy", 0.25, 0.6)]
boxes = [sc.add_entity(s.gs.morphs.MJCF(file=sb.xml(ROOT / f"outputs/handover/softbox/hc_{n}.xml", 8000, m, mu, "cardboard"),
                                        pos=(R, 0.4 + 0.2 * i, 0)), surface=surface(s.gs, "cardboard")) for i, (n, m, mu) in enumerate(cases)]
hc = sc.add_camera(res=(160, 120), fov=70, near=0.005, GUI=False)
sc.build(n_envs=1)
OFFS = {"center": (0.0, 0.0, -0.035), "side": (0.0, -0.035, -0.04)}
off = OFFS[sys.argv[1] if len(sys.argv) > 1 else "side"]
T = np.eye(4); T[:3, 3] = off
hc.attach(ent.get_link("gripper"), T)
A = s.Arm(sc, ent, (0, 0), 0, 1)
def snap(tag):
    hc.move_to_attach()
    Tc = np.asarray(hc.transform) if not callable(getattr(hc, "transform", None)) else np.asarray(hc.transform())
    tip = A.tip()[0]
    print(f"  {tag}: カメラ位置 {np.round(np.asarray(hc.pos), 3).tolist()} 見ている先 {np.round(np.asarray(hc.lookat), 3).tolist()} 指先 {np.round(tip, 3).tolist()}")
    rgb, depth, _, _ = hc.render(rgb=True, depth=True)
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    imageio.imwrite(OUT / f"{tag}.png", rgb)
    h, w = depth.shape
    c = depth[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8]
    print(f"  {tag:22s} 中央の深度 中央値 {np.median(c) * 100:5.1f} cm  最小 {c.min() * 100:5.1f} cm")
for (n, m, mu), b in zip(cases, boxes):
    sc.reset()
    for bb in boxes: sb.hold_spring(bb)
    b.set_pos(np.array([[R, 0, 0]]))
    ent.set_dofs_position(np.tile(s.FOLD, (1, 1))); ent.control_dofs_position(np.tile(s.FOLD, (1, 1))); A.last = None
    for _ in range(30): sc.step()
    A.set_grip_tau(s.TAU_MAX)
    s.move(sc, [(A, 0.0, R - s.JAW, 0.08, s.OPEN)], 120); snap(f"{n}_above")
    s.move(sc, [(A, 0.0, R - s.JAW, 0.02, s.OPEN)], 100); s.move(sc, [(A, 0.0, R - s.JAW, 0.02, s.PRE)], 40)
    A.set_grip_tau(0.2); s.move(sc, [(A, 0.0, R - s.JAW, 0.02, s.CLOSE)], 200); snap(f"{n}_closed")
    s.move(sc, [(A, 0.0, R - s.JAW, 0.04, s.CLOSE)], 60); s.move(sc, [(A, 0.0, R - s.JAW, 0.04, s.CLOSE)], 30); snap(f"{n}_lifted")
    print(f"  {n}: 箱の高さ（答え合わせ）{float(b.get_pos().cpu().numpy()[0, 2]) * 100:.1f} cm  指の角度 {float(ent.get_dofs_position()[0, 5]):.2f}")
