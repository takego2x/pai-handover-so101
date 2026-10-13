"""CI の閾値判定：metrics.json の成功率が下がったら失敗させる。"""
import json
import sys

m = json.load(open(sys.argv[1]))["summary"]
rules = {"B": 1.0, "C": 1.0}  # 安全側の戦略は 1 件も落としてはいけない
bad = [f"{k}: {m[k]['success_rate']} < {v}" for k, v in rules.items() if m[k]["success_rate"] < v]
if m["C"]["mean_lift_move_s"] >= m["B"]["mean_lift_move_s"]:
    bad.append("C is not faster than B")
print(json.dumps(m, indent=1))
sys.exit("FAILED: " + "; ".join(bad) if bad else 0)
