"""試し：指のモーターの力の設定（τ）と、指が小箱を押す力、持ち上がるかどうかの関係を、小箱の重さごとに測る。

環境ごとに (τ, 重さ) を変え、その 1（原点）が床の小箱を掴んで 2 cm 持ち上げる。
  docker compose run --rm -v <SO-ARM100>:/m:ro sim python handover/spikes/grip_window.py
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import so101 as s  # noqa: E402

TAUS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.94]
MASSES = [0.03, 0.1, 0.3]
grid = list(itertools.product(TAUS, MASSES))
N = len(grid)
R, SIDE = 0.22, 0.03

sc = s.make_scene()
sc.add_entity(s.gs.morphs.Plane())
ent = s.add_so101(sc)
cube = sc.add_entity(s.gs.morphs.Box(size=(SIDE,) * 3, pos=(R, 0, SIDE / 2)), material=s.gs.materials.Rigid(friction=1.0, rho=1000.0))
sc.build(n_envs=N)
cube.set_mass(torch.tensor([m for _, m in grid], dtype=s.gs.tc_float))
arm = s.Arm(sc, ent, (0, 0), 0.0, N)
for _ in range(30):
    sc.step()
tau = np.array([t for t, _ in grid])
peak = np.zeros(N)


def watch(k):
    global peak
    peak = np.maximum(peak, s.squeeze(arm, cube))


s.move(sc, [(arm, 0.0, R - s.JAW, 0.10, s.OPEN)], 150)
s.move(sc, [(arm, 0.0, R - s.JAW, 0.02, s.OPEN)], 150)
s.close_gently(sc, arm, 0.0, R - s.JAW, 0.02, tau, watch)
hold_sq = s.squeeze(arm, cube)
s.move(sc, [(arm, 0.0, R - s.JAW, 0.04, s.CLOSE)], 80, watch)
s.move(sc, [(arm, 0.0, R - s.JAW, 0.04, s.CLOSE)], 60, watch)
z = cube.get_pos().cpu().numpy()[:, 2]
print(" tau[N·m] mass[g]  squeeze_at_close[N]  peak[N]  lifted")
for i, (t, m) in enumerate(grid):
    print(f"  {t:5.2f}   {m * 1000:5.0f}      {hold_sq[i]:7.2f}           {peak[i]:6.2f}   {'yes' if z[i] > 0.03 else 'NO'}")
