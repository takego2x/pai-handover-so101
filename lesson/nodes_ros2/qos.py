"""L8b: トピックの QoS（届け方の約束）。3 つのノードで同じものを使う。"""
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def qos(depth=100, volatile=False):
    # RELIABLE：届くまで送り直す。KEEP_LAST + depth：受け手の待ち行列の深さ（dora の queue_size に当たる）
    # TRANSIENT_LOCAL：あとから来た受け手にも、送り済みのメッセージを渡す設定。
    # （この教材の時間の並びでは、外しても 15/15 で差が出なかった。効き目は確かめていない）
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=depth,
                      reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE if volatile else DurabilityPolicy.TRANSIENT_LOCAL)
