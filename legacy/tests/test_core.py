"""S12: CI で回す最小のテスト。物理は数秒で終わる 1 ケースだけ回す。"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))

from mu_table import mu_range, safe_mu  # noqa: E402
from transport import a_max, planned_accel, trapezoid  # noqa: E402
from vlm_material import parse  # noqa: E402


def test_mu_table_reads_only_verified_values():
    lo, hi, src = mu_range("POM")
    assert (lo, hi) == (0.07, 0.16) and src == ["opalka2022"]  # 未確認の wieleba2009 は入らない


def test_safe_mu_takes_lowest_and_refuses_unknown():
    assert safe_mu(["POM", "ABS"]) == 0.07
    assert safe_mu(["POM", "UNKNOWN"]) is None  # 値のない候補 → 決めない


def test_planned_accel_keeps_friction_margin():
    for mu in (0.07, 0.1, 0.2, 0.61):
        a = planned_accel(mu)
        assert a <= a_max(mu) + 1e-9  # 計画は理論の限界を超えない


def test_trapezoid_reaches_distance():
    pos, total = trapezoid(0.4, 2.0, 1.5)
    assert abs(pos(total) - 0.4) < 1e-9 and total > 0


def test_parse_handles_code_fence():
    text = 'Sure:\n```json\n[{"point": [500, 250], "label": "white box", "candidates": ["POM", "glass"]}]\n```'
    items = parse(text)
    assert items == [{"point": [500.0, 250.0], "label": "white box", "candidates": ["POM"]}]  # 表にない材質は捨てる


def test_physics_c_is_safe_on_worst_case_pom(tmp_path):
    # μ が下限ちょうど（seed 1004, μ=0.0701）の POM を、戦略 C が 1 環境で落とさないこと（余裕 4.5% で落ちた回帰）
    out = tmp_path / "r.json"
    subprocess.run([sys.executable, str(ROOT / "sim" / "transport.py"), "POM", "C", "1", "--seed0", "1004",
                    "--json", str(out)], check=True, capture_output=True)
    assert json.load(open(out))["success"] == [True]
