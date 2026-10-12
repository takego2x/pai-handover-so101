"""L8b 記録ノード（ROS 2）：/result を集め、届くはずの数がそろったら /done を出して集計する。"""
import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import l8_core as core  # noqa: E402
from qos import qos  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--expected", type=int, required=True)
ap.add_argument("--timeout", type=float, default=600.0, help="これだけ待ってもそろわなければ、足りないまま終える [s]")
args = ap.parse_args()

rclpy.init()
node = Node("monitor")
results = []
node.create_subscription(String, "result", lambda m: results.append(json.loads(m.data)), qos())
done_pub = node.create_publisher(Empty, "done", qos())
t0 = time.time()
while rclpy.ok() and len(results) < args.expected and time.time() - t0 < args.timeout:
    rclpy.spin_once(node, timeout_sec=0.5)
done_pub.publish(Empty())  # 第一層と第二層に「終わり」を知らせる
print(core.summarize(results), flush=True)
if len(results) != args.expected:  # 届いた数だけで「全部成功」と言わない
    print(f"[monitor] MISSING: received {len(results)} results, expected {args.expected}", flush=True)
out = Path(__file__).resolve().parent.parent.parent / "outputs" / "lesson"
out.mkdir(parents=True, exist_ok=True)
json.dump(results, open(out / "l8_ros2.json", "w"), indent=1)
time.sleep(2.0)  # /done が届くまで少し待つ
node.destroy_node()
rclpy.shutdown()
