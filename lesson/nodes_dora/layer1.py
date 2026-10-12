"""L8a 第一層ノード（dora）：試し掴みで部品のデータを作り、1 個ずつ part_data として送る。"""
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
from dora import Node

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lesson/ を読み込めるようにする
import common as c  # noqa: E402
import node_core as core  # noqa: E402

c.quiet()
CELL, N = int(os.environ["CELL"]), int(os.environ["N_PARTS"])
node = Node()
sent = False
for event in node:
    if event["type"] != "INPUT":
        continue
    if event["id"] == "tick" and not sent:  # 最初の合図で 1 回だけ、試し掴みをして送る
        for msg in core.estimate_parts(N, cell=CELL, log=lambda s: print(s, flush=True)):
            node.send_output("part_data", pa.array([json.dumps(msg)]))  # 送る：配列に JSON を 1 つ
        sent = True
        if os.environ.get("MISTAKE") == "exit_early":
            break  # わざと失敗：送ったらすぐ終了する
    elif event["id"] == "done":
        # 送り終わってもすぐ終了しない。dora では、送ったノードが終了すると、相手の待ち行列に
        # 残っているメッセージが捨てられる（2026-10-12 に実測）。記録ノードの「全部届いた」を待ってから終了する
        break
