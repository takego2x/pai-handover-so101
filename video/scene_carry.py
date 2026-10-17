"""動画の場面 7：同じ部品を 3 通りの持ち方で運ぶ（環境 0 = データなし 60 N、1 = 式・半径向き、2 = 学習した方策）。
1 つの環境だけを録るので、環境ごとに 3 回動かす:
  PAI_CLIP=outputs/video/carry_0.mp4 PAI_CLIP_ENV=0 python video/scene_carry.py --part 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson"))
import common as c  # noqa: E402
import policy as pol  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--part", type=int, default=3, help="評価の部品の番号（test seed 9000 の 30 個のうち）")
args = ap.parse_args()
c.quiet()
mats, mu, mu_hat, m = pol.sample(30, 9000, stream=99)
i = args.part
W = json.load(open(c.ROOT / "data" / "policy_W.json"))["W"]
sc = pol.ArcScene(3)
n = 3
kF, a, yaw = pol.formula(n, tangent=False)
F_formula = float(c.carry_force(m[i], mu_hat[i]))
kF[0] = c.F_MAX / F_formula                       # データなし：上限の 60 N で握る（指は半径の向き、加速度 9）
kL, aL, yL = pol.learned(W, mu_hat[[i]], m[[i]])  # 学習した方策
kF[2], a[2], yaw[2] = kL[0], aL[0], yL[0]
ok, F, t = sc.episode([mats[i]] * n, np.full(n, mu[i]), np.full(n, mu_hat[i]), kF, a, yaw)
print(f"part {i}: {mats[i]} {m[i] * 1000:.0f} g  mu {mu[i]:.3f} (data {mu_hat[i]:.3f})")
for k, name in enumerate(["nodata_60N", "formula_radial", "learned"]):
    print(f"  {name:15s} F {F[k]:5.2f} N  a {a[k]:4.1f}  yaw {np.degrees(yaw[k]):5.1f} deg  time {t[k]:.2f} s  ok {bool(ok[k])}")
