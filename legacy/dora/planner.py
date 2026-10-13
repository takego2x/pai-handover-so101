"""planner：部品ごとに、候補の μ の下限から持ち上げの加速度を決める（戦略 C）。"""
import json

import pyarrow as pa
from dora import Node

import common  # noqa: F401  (sim/ を import できるようにする)
from mu_table import load, safe_mu
from transport import assumed_mu, planned_accel

rows = load()
mu_worst = assumed_mu("B", None, rows)
node = Node()
for event in node:
    if event["type"] != "INPUT":
        continue
    md = event["metadata"]
    for i, s in enumerate(event["value"].to_pylist()):
        p = json.loads(s)
        mu = safe_mu(p["candidates"], rows=rows) if p["candidates"] else None
        mu = mu if mu is not None else mu_worst  # 候補が空・値のない候補 → 最悪値に倒す
        plan = {**p, "tray": md["tray"], "part": i, "mu_hat": mu, "a_plan": round(planned_accel(mu), 3)}
        node.send_output("plan", pa.array([json.dumps(plan)]), metadata={**md, "part": i})
