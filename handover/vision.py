"""固定カメラで箱を見つける：作業台を真上から見るカメラのカラー画像と深度から、箱の位置と向きを出す。

  ロボットにカメラは付けない。カメラはシミュレータの中の決まった場所に置き、Genesis で描いた画像と深度だけを使う
  （シミュレータの箱の座標は見ない）。
  1. 深度 → 画素ごとの 3 次元の点（カメラの位置・向き・画角から、ピンホールの式で逆算）
  2. 床からの高さが「箱の上の面」の帯にある点を残し、ロボットの色（黄・黒）の画素を除く
  3. つながった点のかたまり（大きさが 3 cm の箱の上の面くらい）を箱とみなし、
     中心 → 位置、外形を囲む最小の長方形の傾き → 向き（四角なので 90° ごとに同じ）
"""
import math

import numpy as np
from scipy import ndimage

SIDE = 0.03


def add_top_camera(sc, center=(0.0, 0.0), height=1.2, fov=40, res=(1280, 1280)):
    """作業台を真上から見る固定カメラ。up は +y（画像の上が +y）"""
    cam = sc.add_camera(res=res, pos=(center[0], center[1], height), lookat=(center[0], center[1], 0.0),
                        up=(0.0, 1.0, 0.0), fov=fov, GUI=False)
    cam._top = {"center": np.array(center, float), "height": height, "fov": fov, "res": res}
    return cam


def _points(cam, depth):
    """深度（光軸方向の距離）→ 世界座標の点 (H, W, 3)。真上から見下ろすカメラ：画像の右＝+x、上＝+y"""
    t = cam._top
    w, h = t["res"]
    f = (h / 2) / math.tan(math.radians(t["fov"]) / 2)        # 縦の画角から焦点距離 [px]
    u = np.arange(w) - (w - 1) / 2
    v = np.arange(h) - (h - 1) / 2
    uu, vv = np.meshgrid(u, v)
    z = np.asarray(depth, float)
    x = t["center"][0] + uu * z / f
    y = t["center"][1] - vv * z / f
    return np.stack([x, y, t["height"] - z], axis=-1)


def look(cam, z_band=(0.022, 0.062), area_cm2=(4.0, 14.0)):
    """箱の候補の一覧 [(x, y, 上の面の高さ, 向き [rad], 面積 [cm²], 平均の色)]。手や腕に隠れた箱は出ない"""
    rgb, depth, _, _ = cam.render(rgb=True, depth=True)
    rgb = np.asarray(rgb)
    P = _points(cam, np.asarray(depth))
    zz = P[..., 2]
    r, g, b = (rgb[..., k].astype(int) for k in range(3))
    yellow = (r > 140) & (g > 110) & (b < 90)
    black = (r < 45) & (g < 45) & (b < 45)
    mask = (zz > z_band[0]) & (zz < z_band[1]) & ~yellow & ~black
    lab, n = ndimage.label(mask)
    px_area = (cam._top["height"] * 2 * math.tan(math.radians(cam._top["fov"]) / 2) / cam._top["res"][1]) ** 2   # 1 画素の面積 [m²]（床の高さで）
    out = []
    for k in range(1, n + 1):
        sel = lab == k
        area = sel.sum() * px_area * 1e4
        if not (area_cm2[0] <= area <= area_cm2[1]):
            continue
        pts = P[sel]
        top = float(np.percentile(pts[:, 2], 90))
        xy = pts[:, :2]
        c = xy.mean(axis=0)
        out.append((float(c[0]), float(c[1]), top, _yaw(xy - c), float(area), rgb[sel].mean(axis=0)))
    return out


def _yaw(xy):
    """点の集まりを囲む最小の長方形の傾き（0〜90°）。1° 刻みで回して、外形の面積がいちばん小さい角度"""
    best, ang = None, 0.0
    for d in np.arange(0, 90, 1.0):
        a = math.radians(d)
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        q = xy @ R.T
        s = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if best is None or s < best:
            best, ang = s, a
    return ang


def nearest(cands, xy, max_d=0.05):
    """xy に一番近い候補（max_d より遠ければ None）"""
    if not cands:
        return None
    d = [math.hypot(c[0] - xy[0], c[1] - xy[1]) for c in cands]
    i = int(np.argmin(d))
    return cands[i] if d[i] <= max_d else None


def add_top_camera_rig(sc, center=(0.0, 0.0), height=1.2, pole_xy=(0.75, 0.0)):
    """真上のカメラの見た目（支柱・腕・本体）。見た目だけで、ぶつからない（collision=False）。build の前に呼ぶ。
    支柱は真上のカメラの写る範囲（床で 1.3 m 角）の外に立て、腕と本体はカメラより上に置く（カメラの画像に写らない）"""
    gray, dark = (0.55, 0.56, 0.58), (0.12, 0.12, 0.13)
    px, py = pole_xy
    cx, cy = center
    top = height + 0.08
    box = lambda size, pos, color: sc.add_entity(s_gs().morphs.Box(size=size, pos=pos, fixed=True, collision=False),
                                                surface=s_gs().surfaces.Default(color=color))
    box((0.04, 0.04, top), (px, py, top / 2), gray)                                         # 支柱
    L = math.hypot(cx - px, cy - py)
    ang = math.atan2(cy - py, cx - px)
    sc.add_entity(s_gs().morphs.Box(size=(L, 0.03, 0.03), pos=((px + cx) / 2, (py + cy) / 2, top), euler=(0, 0, math.degrees(ang)),
                                    fixed=True, collision=False), surface=s_gs().surfaces.Default(color=gray))  # 腕
    box((0.10, 0.08, 0.05), (cx, cy, height + 0.035), dark)                                 # 本体（下の面がカメラより 1 cm 上）


def s_gs():
    import genesis as gs
    return gs


# ---- 手首のカメラ（公式モデル so101_new_calib_camera.xml の取り付け位置。SO-ARM100 a758567） ----
# 公式の部品 wrist_camera は 32 mm 角の UVC カメラ。手を真下に向けたとき、指先の方を向くのは部品の +z
# （実測：指先への向きと cos 0.981、handover/spikes/wrist_camera.py）。指先から 10.5 cm 上・横に 7 cm の所から、
# 斜めに指のあいだを見下ろす。前は手首のモーターの内側（グリッパの中央）に付けていたが、実機ではそこに付けられない
CAMERA_MODEL = "so101_new_calib_camera.xml"
WRIST_LENS = (0.0, 0.0, 0.0275)        # 部品の +z の面（0.0263）の少し前
# 箱が写る範囲（画像の縦・横の割合）。閉じた指のあいだの箱は、いつもここに写る（グリッパに対して箱の位置が決まっているため）
BOX_ROI = (0.583, 0.75, 0.47, 0.625)


def camera_model(base_model):
    """SO-101 のモデルのパスから、同じフォルダの公式のカメラ付きモデルのパス"""
    from pathlib import Path
    return str(Path(base_model).with_name(CAMERA_MODEL))


def add_hand_camera(sc, res=(320, 240), fov=70):
    """build の前に作る。手前の限界 near は 5 mm（既定の 10 cm では、手を下ろすと箱も床も描かれなかった）"""
    return sc.add_camera(res=res, fov=fov, near=0.005, GUI=False)


def attach_hand(cam, ent):
    """build の後に、腕（カメラ付きモデル）の wrist_camera の部品へ取り付ける。
    Genesis のカメラは自分の −z を見るので、x 軸のまわりに 180° 回す（カメラの −z ＝ 部品の +z）"""
    T = np.diag([1.0, -1.0, -1.0, 1.0])
    T[:3, 3] = WRIST_LENS
    cam.attach(ent.get_link("wrist_camera"), T)
    return cam


def hand_roi(cam):
    """手首のカメラの色の画像のうち、指のあいだの箱が写る範囲（float の配列）。実機の手首カメラ（UVC）には深度がないので、色だけを使う"""
    return roi_of(hand_rgb(cam))


def roi_of(img):
    h, w = img.shape[:2]
    r0, r1, c0, c1 = BOX_ROI
    return img[int(r0 * h):int(r1 * h), int(c0 * w):int(c1 * w)].astype(float)


SHIFT_BOX = (0.40, 0.90, 0.36, 0.74)    # 箱の輪郭を含む範囲（縦・横の割合）。箱が画像の中で何画素ずれたかを探す範囲


def box_shift(before, after, max_dy=24, max_dx=8):
    """持ち上げる前後の手首の画像で、箱が何画素ずれたか (dy, dx, その位置での平均の差)。
    ロボットの色（黄・黒）は手と一緒に動くので比べない。箱が手と一緒に上がれば (0, 0)、滑れば滑った分だけ dy が正（下へ）"""
    h, w = before.shape[:2]
    r0, r1, c0, c1 = (int(SHIFT_BOX[0] * h), int(SHIFT_BOX[1] * h), int(SHIFT_BOX[2] * w), int(SHIFT_BOX[3] * w))
    rgb = before.astype(int)
    robot = ((rgb[..., 0] > 140) & (rgb[..., 1] > 110) & (rgb[..., 2] < 90)) | ((rgb[..., 0] < 45) & (rgb[..., 1] < 45) & (rgb[..., 2] < 45))
    g0, g1 = before.mean(axis=2), after.mean(axis=2)
    tpl, keep = g0[r0:r1, c0:c1], ~robot[r0:r1, c0:c1]
    best = None
    for dy in range(-max_dy, max_dy + 1):
        for dx in range(-max_dx, max_dx + 1):
            if r0 + dy < 0 or r1 + dy > h or c0 + dx < 0 or c1 + dx > w:
                continue
            d = np.abs(g1[r0 + dy:r1 + dy, c0 + dx:c1 + dx] - tpl)[keep].mean()
            if best is None or d < best[2]:
                best = (dy, dx, float(d))
    return best


def same_view(a, b):
    """2 枚の範囲の見た目の近さ：(正規化相互相関 NCC、平均の差 MAD)。箱が手と一緒に動けば、同じ場所に同じように写る。
    実測（handover/spikes/wrist_rgb.py、段ボール・発泡・金属、床）：持てた NCC 0.93〜0.97・MAD 4〜5／持てない NCC 0.11〜0.76・MAD 27〜49"""
    x, y = a.mean(axis=2).ravel(), b.mean(axis=2).ravel()
    x, y = x - x.mean(), y - y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9)), float(np.abs(a - b).mean())


def hand_rgb(cam):
    cam.move_to_attach()
    return np.asarray(cam.render(rgb=True)[0])
