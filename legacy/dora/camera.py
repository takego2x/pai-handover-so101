"""camera：トレイの画像を 1 枚ずつ送る（Genesis で描画済みの画像を、カメラの代わりに流す）。"""
import os
import time

import imageio.v3 as iio
import pyarrow as pa
from dora import Node

from common import ROOT

N = int(os.environ.get("N_TRAYS", "3"))
node = Node()
sent = 0
for event in node:
    if event["type"] == "INPUT" and event["id"] == "tick":
        img = iio.imread(ROOT / "outputs" / "tray" / f"tray_{sent}.png")[..., :3]
        h, w, _ = img.shape
        # 画素をそのまま Arrow の配列で送る（受け手は形を metadata から戻す）
        node.send_output("image", pa.array(img.ravel()), metadata={"tray": sent, "h": h, "w": w, "t_cam": time.monotonic()})
        print(f"[camera] tray {sent} sent", flush=True)
        sent += 1
        if sent >= N:
            break
