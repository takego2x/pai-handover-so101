"""L8b 第二層ノード（ROS 2）：/part_data を受け取るたびに運び、/result に出す。/done を受け取ったら終わる。"""
import argparse
import json
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as c  # noqa: E402
import node_core as core  # noqa: E402
from qos import qos  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mistake", choices=["exit_early", "depth1", "volatile"])
ap.add_argument("--policy", choices=["learned", "formula"], default="learned")
args = ap.parse_args()
c.quiet()

carrier = core.Carrier(policy=args.policy)  # Genesis の場面を作る（30 秒ほどかかる）
rclpy.init()
node = Node("layer2")
pub = node.create_publisher(String, "result", qos())
finished = {"done": False}


def on_part(msg):  # 受け取る：メッセージが届くたびに呼ばれる関数（コールバック）
    d = json.loads(msg.data)
    r = carrier.carry(d)
    node.get_logger().info(f"cell {d['cell']} part {d['part']} {r['material']:9s} F={r['F_N']:.2f}N "
                           f"{'delivered' if r['delivered'] else 'DROPPED'}")
    pub.publish(String(data=json.dumps(r)))


node.create_subscription(String, "part_data", on_part,
                         qos(1 if args.mistake == "depth1" else 100, volatile=args.mistake == "volatile"))
node.create_subscription(Empty, "done", lambda m: finished.update(done=True), qos())
while rclpy.ok() and not finished["done"]:
    rclpy.spin_once(node, timeout_sec=0.5)
node.destroy_node()
rclpy.shutdown()
