"""L8a 第二層ノード（dora）：part_data を受け取るたびに運び、result を送る。"""
import json
import sys
from pathlib import Path

import pyarrow as pa
from dora import Node

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as c  # noqa: E402
import node_core as core  # noqa: E402

c.quiet()
carrier = core.Carrier(policy=__import__("os").environ.get("POLICY", "learned"))  # learned（L7）か formula
node = Node()
closed = set()
for event in node:
    if event["type"] == "INPUT":
        msg = json.loads(event["value"].to_pylist()[0])  # 受け取る：配列から JSON を取り出す
        r = carrier.carry(msg)
        print(f"[layer2] cell {msg['cell']} part {msg['part']} {r['material']:9s} F={r['F_N']:.2f}N "
              f"{'delivered' if r['delivered'] else 'DROPPED'}", flush=True)
        node.send_output("result", pa.array([json.dumps(r)]))
    elif event["type"] == "INPUT_CLOSED":
        closed.add(event["id"])
        if closed >= {"from0", "from1", "from2"}:
            break
