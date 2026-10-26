"""潰れる箱：本体（自由に動く）＋ 外側の面の板（ばねの付いたすべり関節）。

  板を押す力 ＝ 硬さ k × 沈み量。ただし 強度 まで。強度を超えると板は沈みきる（＝潰れた）。
  決まり：柔らかい箱ほど弱い（強度 ＝ k × STRENGTH_PER_K）。
  重さ・硬さ・摩擦は箱ごとに決める（ランダムにもできる）。見た目（色）は、硬さの傾向と緩く結び付ける。

ロボットが読めるのは、自分の指の角度と、指の力の設定だけ。板の沈み量（本当の値）は答え合わせにだけ使う。
"""
from pathlib import Path

import numpy as np

SIDE, PLATE, TRAVEL = 0.03, 0.010, 0.008   # 箱の一辺、板の厚さ、沈みきる量。板は沈みきる量より厚くする
STRENGTH_PER_K = 0.003                     # 強度 [N] ＝ k [N/m] × 3 mm
LOOKS = {  # 見た目 → 色（MJCF の rgba）
    "cardboard": (0.72, 0.55, 0.36),
    "foam": (0.93, 0.93, 0.90),
    "metal": (0.55, 0.57, 0.60),
}


def strength_of(k):
    return k * STRENGTH_PER_K


def xml(path, k, mass, mu, look="cardboard", show_plate=False):
    """本体＋板 2 枚（+x の面と +y の面、どちらもばね付きのすべり関節）の MJCF を書き、パスを返す。
    板が 1 枚だと、カメラで分かる向きは 90° ごとまでなので、板のない向きで挟むことがあり、硬さが測れない。
    本物の箱はどの向きから押しても潰れるので、2 つの向きに板を付ける。
    見た目：show_plate=False（既定）は、本体に箱 1 個分の外形（見た目だけ、ぶつからない）を付けて絵を 1 枚で貼り、
    ぶつかり判定用の本体と板は見せない（グループ 3）。板が沈むのを見せたいとき（潰す場面）は show_plate=True"""
    h = SIDE / 2
    core = SIDE - PLATE                            # 本体の幅（板の分だけ小さい。x・y とも）
    m_plate = mass * 0.1
    r, g, b = LOOKS[look]
    grp = "" if show_plate else ' group="3"'
    visual = "" if show_plate else f'\n      <geom type="box" size="{h} {h} {h}" contype="0" conaffinity="0" mass="0" rgba="{r} {g} {b} 1"/>'
    plate = lambda name, axis, pos, size: f"""
      <body name="{name}" pos="{pos}">
        <joint name="{name}_squash" type="slide" axis="{axis}" range="{-TRAVEL} 0.001" damping="2"/>
        <geom type="box" size="{size}" mass="{m_plate}" friction="{mu} 0.005 0.0001" rgba="{r} {g} {b} 1"{grp}/>
      </body>"""
    text = f"""<mujoco model="soft_box">
  <worldbody>
    <body name="core" pos="0 0 {h}">
      <freejoint/>
      <geom type="box" size="{core / 2} {core / 2} {h}" pos="{-PLATE / 2} {-PLATE / 2} 0" mass="{mass - 2 * m_plate}" friction="{mu} 0.005 0.0001" rgba="{r} {g} {b} 1"{grp}/>{visual}{plate("plate_x", "1 0 0", f"{h - PLATE / 2} {-PLATE / 2} 0", f"{PLATE / 2} {core / 2} {h}")}{plate("plate_y", "0 1 0", f"{-PLATE / 2} {h - PLATE / 2} 0", f"{core / 2} {PLATE / 2} {h}")}
    </body>
  </worldbody>
  <actuator>
    <position name="spring_x" joint="plate_x_squash" kp="{k}" forcerange="{-strength_of(k)} {strength_of(k)}"/>
    <position name="spring_y" joint="plate_y_squash" kp="{k}" forcerange="{-strength_of(k)} {strength_of(k)}"/>
  </actuator>
</mujoco>
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    return str(path)


def hold_spring(ent):
    """板のばね（2 枚）を効かせる（作った直後と、場面を初期状態に戻した後に、毎回呼ぶ。戻すと指示が消える）"""
    ent.control_dofs_position(np.zeros((1, 2)), [ent.n_dofs - 2, ent.n_dofs - 1])


def squash(ent):
    """板の沈み量 [m]（2 枚の大きいほう。答え合わせ用）"""
    return -float(ent.get_dofs_position().cpu().numpy()[0, -2:].min())


def crushed(ent):
    return squash(ent) > TRAVEL * 0.9


def random_boxes(n, seed=0, k_range=(2500, 20000), mass_range=(0.03, 0.25), mu_range=(0.6, 1.0)):
    """ランダムな箱 n 個。見た目は硬さの傾向で選ぶ（柔らかい→発泡、硬い→金属、あいだ→段ボール）が、2 割は見た目と硬さを入れ替える"""
    # 重さは 250 g まで、摩擦は 0.6 以上：どの箱も τ 0.4 以下で持ち上がり（300 g・摩擦 0.64 は回によって 0.4 で持てなかった）、宙で強度を測れるようにする（床の上で測ると、箱が滑ったり傾いたりして
    # 強度を 4 割大きく見積もった）。0.3〜0.5 の滑りやすい箱は、課題の見せ場（同じ力でも滑る）として別に扱う
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = float(np.exp(rng.uniform(np.log(k_range[0]), np.log(k_range[1]))))
        look = "foam" if k < 3500 else ("metal" if k > 12000 else "cardboard")
        if rng.uniform() < 0.2:
            look = rng.choice([x for x in LOOKS if x != look])
        out.append({"name": f"box{i}", "k": round(k), "strength": round(strength_of(k), 1),
                    "mass": round(float(rng.uniform(*mass_range)), 3), "mu": round(float(rng.uniform(*mu_range)), 2), "look": str(look)})
    return out
