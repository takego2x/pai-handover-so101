"""executor：計画どおりに掴んで運ぶ（Genesis）。シミュレータの中の本当の材質は、正解の画素位置と照合して決める。"""
import json
import subprocess
import sys
import tempfile
import time

import pyarrow as pa
from dora import Node

from common import ROOT

SIMULATED = {"POM", "aluminium", "steel"}  # 評価と同じ 3 材質だけ運ぶ
node = Node()
for event in node:
    if event["type"] != "INPUT":
        continue
    md = event["metadata"]
    plan = json.loads(event["value"].to_pylist()[0])
    truth = json.load(open(ROOT / "outputs" / "tray" / f"tray_{plan['tray']}.json"))
    ty, tx = plan["point"]
    t_ = min(truth, key=lambda t: (t["point_1000"][0] - ty) ** 2 + (t["point_1000"][1] - tx) ** 2)
    res = {"tray": plan["tray"], "part": plan["part"], "material": t_["label"], "candidates": plan["candidates"],
           "mu_hat": plan["mu_hat"], "a_plan": plan["a_plan"]}
    t0 = time.monotonic()
    if t_["label"] not in SIMULATED:
        res["status"] = "skipped (not simulated)"
    else:
        seed = 1000 + plan["tray"] * 5 + plan["part"]
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            r = subprocess.run([sys.executable, str(ROOT / "sim" / "transport.py"), t_["label"], "C", "1",
                                "--seed0", str(seed), "--candidates", ",".join(plan["candidates"] or ["UNKNOWN"]),
                                "--json", f.name], capture_output=True, text=True)
            if r.returncode != 0:
                res["status"] = "error: " + r.stderr[-300:]
            else:
                d = json.load(open(f.name))
                res |= {"status": "held" if d["success"][0] else "DROPPED",
                        "t_lift_move": round(d["t_lift"] + d["t_move"], 3), "margin": d["margin"][0]}
    res["t_exec_start"], res["t_exec_done"] = t0, time.monotonic()
    node.send_output("result", pa.array([json.dumps(res)]), metadata=md)
    print(f"[executor] tray {res['tray']} part {res['part']} {res['material']:9s} {res['status']}", flush=True)
