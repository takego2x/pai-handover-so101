"""試し：SmolVLA（lerobot/smolvla_base）を Mac 本体で動かし、Docker の中のシミュレータに次の動きを返す推論役。

SmolVLA は LeRobot の別の版の PyTorch を要るので、シミュレータの Docker には入れず、Mac 本体の .smolvla/.venv で動かす。
受け取る：{"state": [関節 6、ラジアン], "images": [PNG の base64 × 3（camera1〜3）], "task": "言葉の指示"}
返す：{"action_norm": [[6] × 50]}（学習データの平均・ばらつきで割った値。シミュレータ側で関節に当てはめる）
注意：学習データ（実機の SO-100、度、別の校正）とシミュレータの SO-101（ラジアン）は単位も 0 の位置も違う。
      smolvla_base は手本で微調整してから使う土台で、そのままでは狙った作業はしない前提の試し。

使い方（リポジトリ直下）: .smolvla/.venv/bin/python handover/spikes/smolvla_server.py
"""
import base64
import io
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from PIL import Image
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

POLICY = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
POLICY.eval()
DEV = next(POLICY.parameters()).device
# 前処理（言葉の指示をトークンにする など）。関節の値の正規化は、学習データの組の名前が合わず効かないので、下で自分で割る
PRE, _ = make_pre_post_processors(POLICY.config, pretrained_path="lerobot/smolvla_base",
                                  preprocessor_overrides={"device_processor": {"device": str(DEV)}},
                                  postprocessor_overrides={"device_processor": {"device": "cpu"}})
# 学習データ（コミュニティの SO-100）の関節の値の平均・ばらつき [度]。前処理の設定から読んだ値（so100 の組）
MEAN = np.array([1.6, 119.9, 109.8, 56.7, -27.4, 12.0])
STD = np.array([26.4, 52.4, 49.9, 37.0, 59.4, 19.0])
HOME = None   # 最初に受け取った関節の角度（シミュレータの構え）を、学習データの平均に対応させる


def img(b64):
    a = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((256, 256)), np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(DEV)


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        global HOME
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        q = np.array(req["state"])
        if HOME is None:
            HOME = q.copy()
        state_norm = (q - HOME) * 180 / math.pi / STD          # 構えからのずれを、学習データのばらつきで割る
        batch = {"observation.state": torch.tensor(state_norm, dtype=torch.float32)[None], "task": [req["task"]]}
        for i, b in enumerate(req["images"]):
            batch[f"observation.images.camera{i + 1}"] = img(b).cpu()
        batch = PRE(batch)
        with torch.no_grad():
            a = POLICY.predict_action_chunk(batch)[0].float().cpu().numpy()
        out = json.dumps({"action_norm": a.tolist()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"SmolVLA 推論役 :8765（{DEV}）", flush=True)
    HTTPServer(("0.0.0.0", 8765), H).serve_forever()
