"""L8a 記録ノード（dora）：result を集め、上流が終わったら集計する。"""
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
from dora import Node

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import l8_core as core  # noqa: E402

node = Node()
results = []
expected = int(os.environ.get("EXPECTED", "0"))
for event in node:
    if event["type"] == "INPUT":
        results.append(json.loads(event["value"].to_pylist()[0]))
        if len(results) == expected:
            node.send_output("done", pa.array([len(results)]))  # 第一層に「全部届いた」と知らせる
    elif event["type"] == "INPUT_CLOSED":
        break
print(core.summarize(results), flush=True)
# 届いた数だけで「全部成功」と言わない。届くはずの数（第一層の台数 × 部品の数）と比べる
if expected and len(results) != expected:
    print(f"[monitor] MISSING: received {len(results)} results, expected {expected}", flush=True)
out = Path(__file__).resolve().parent.parent.parent / "outputs" / "lesson"
out.mkdir(parents=True, exist_ok=True)
json.dump(results, open(out / "l8_dora.json", "w"), indent=1)
