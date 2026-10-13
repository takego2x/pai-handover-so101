"""CI で回すテスト。教材（lesson/）の部品と、物理の結果を 2 つだけ確かめる（数分で終わる）。"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lesson"))
sys.path.insert(0, str(ROOT / "sim"))

import common as c  # noqa: E402
from mu_table import mu_range, safe_mu  # noqa: E402


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-1500:]
    return r.stdout


def test_mu_table_reads_only_verified_values():
    lo, hi, src = mu_range("POM")
    assert (lo, hi) == (0.07, 0.16) and src == ["opalka2022"]  # 未確認の wieleba2009 は入らない
    assert safe_mu(["POM", "UNKNOWN"]) is None  # 値のない候補 → 決めない


def test_carry_force_keeps_the_margin():
    # §1.3：運ぶ力 F で、加速度 9 m/s^2 の持ち上げに 15% の余裕が残る
    for m, mu in [(0.090, 0.07), (0.502, 0.2), (0.173, 0.61)]:
        F = c.carry_force(m, mu)
        if F < c.F_MAX:
            assert abs(2 * mu * F * c.SAFETY - m * (c.G + c.A_CAP)) < 1e-9
            assert abs(c.accel_for(m, mu, F) - c.A_CAP) < 1e-9


def test_trapezoid_reaches_distance():
    pos, total = c.trapezoid(0.4, 2.0)
    assert abs(pos(total) - 0.4) < 1e-9 and total > 0


def test_learned_policy_wraps_yaw():
    import policy
    W = np.zeros((3, 3))
    W[2, 0] = math.radians(100)  # 100° は回り込んで −80°（同じ持ち方）
    policy.YAW_ONE, policy.YAW_WRAP = True, True
    _, _, yaw = policy.learned(W, np.array([0.3]), np.array([0.1]))
    assert abs(math.degrees(yaw[0]) + 80) < 1e-6


def test_hold_matches_theory():
    # L1：保持の条件を、シミュレータと理論の 24 マスで突き合わせる
    assert "match 24/24" in run("lesson/hold.py")


def test_learned_policy_carries_test_parts():
    # L7：同梱の学習済みの方策が、学習に使わない 30 部品を、式の方策より小さい力で運ぶ
    out = ROOT / "outputs" / "ci_policy_eval.json"
    run("lesson/policy.py", "eval", "--test-seed", 9000, "--json", out)
    r = json.load(open(out))
    assert r["learned"]["success"] == 30 and r["formula_tangent"]["success"] == 30
    assert r["learned"]["F_median_N"] < r["formula_tangent"]["F_median_N"]
