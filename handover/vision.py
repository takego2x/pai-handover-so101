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


# ---- ハンドカメラ（手首の中央に付け、指のあいだを見る） ----
HAND_OFFSET = (0.0, 0.0, -0.035)      # グリッパの部品の原点から指先の方向へ 3.5 cm（指のあいだの真上）。横に付けると床しか写らない


def add_hand_camera(sc, res=(160, 120), fov=70):
    """build の前に作る。手前の限界 near は 5 mm（既定の 10 cm では、手を下ろすと箱も床も描かれなかった）"""
    return sc.add_camera(res=res, fov=fov, near=0.005, GUI=False)


def attach_hand(cam, link):
    """build の後に、グリッパの部品へ取り付ける"""
    T = np.eye(4)
    T[:3, 3] = HAND_OFFSET
    cam.attach(link, T)
    return cam


def hand_depth(cam):
    """指のあいだ（画像の中央 16×16 画素）までの距離の中央値 [m]。箱を挟んでいれば箱の上の面、なければ床"""
    cam.move_to_attach()
    _, depth, _, _ = cam.render(rgb=False, depth=True)
    d = np.asarray(depth)
    h, w = d.shape
    return float(np.median(d[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8]))


def hand_rgb(cam):
    cam.move_to_attach()
    return np.asarray(cam.render(rgb=True)[0])
