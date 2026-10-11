"""vlm：画像から、部品ごとの位置と材質の候補を出す。"""
import json
import os
import tempfile
import time

import imageio.v3 as iio
import numpy as np
import pyarrow as pa
from dora import Node

from common import ROOT
from vlm_material import ask_gemini, parse

BACKEND = os.environ.get("VLM_BACKEND", "replay")
node = Node()
for event in node:
    if event["type"] != "INPUT":
        continue
    md = event["metadata"]
    t0 = time.monotonic()
    if BACKEND == "gemini":
        img = event["value"].to_numpy().reshape(md["h"], md["w"], 3).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            iio.imwrite(f.name, img)
            text = ask_gemini(f.name)
    else:  # replay：S7b で保存した Gemini の生の応答を読む（キー不要）
        text = (ROOT / "outputs" / "vlm" / f"tray_{md['tray']}.gemini.txt").read_text()
    parts = parse(text)
    t1 = time.monotonic()
    node.send_output("parts", pa.array([json.dumps(p) for p in parts]),
                     metadata={**md, "t_vlm_start": t0, "t_vlm_done": t1})
    print(f"[vlm] tray {md['tray']}: {len(parts)} parts in {t1 - t0:.2f}s ({BACKEND})", flush=True)
