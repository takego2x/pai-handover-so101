"""L8b 第一層ノード（ROS 2）：試し掴みで部品のデータを作り、/part_data に 1 個ずつ出す。/done を受け取ったら終わる。"""
import argparse
import json
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lesson/ を読み込めるようにする
import common as c  # noqa: E402
import l8_core as core  # noqa: E402
from qos import qos  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cell", type=int, required=True)
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--mistake", choices=["exit_early", "depth1", "volatile"])
args = ap.parse_args()
c.quiet()

rclpy.init()
node = Node(f"layer1_cell{args.cell}")
pub = node.create_publisher(String, "part_data", qos(volatile=args.mistake == "volatile"))  # 出す：トピック名と QoS を決めて publisher を作る
finished = {"done": False}
node.create_subscription(Empty, "done", lambda m: finished.update(done=True), qos())
for msg in core.estimate_parts(args.n, cell=args.cell, log=node.get_logger().info):
    pub.publish(String(data=json.dumps(msg)))
if args.mistake != "exit_early":
    while rclpy.ok() and not finished["done"]:  # 記録ノードの「全部届いた」を待ってから終わる
        rclpy.spin_once(node, timeout_sec=0.5)
node.destroy_node()
rclpy.shutdown()
