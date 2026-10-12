#!/bin/bash
# L8b: ROS 2 のノード 5 つ（第一層 3 台・第二層・記録）を、別々のプロセスとして起動する。
# 使い方（作業場所の直下で）:
#   docker compose --profile ros2 run --rm ros2 bash lesson/l8_ros2/run.sh [部品の数] [わざと失敗]
#   わざと失敗: exit_early（第一層が送ったらすぐ終わる）／ depth1（第二層の待ち行列の深さを 1 にする）
#              volatile（あとから来た受け手には、送り済みのメッセージを渡さない）
N=${1:-5}
MIS=${2:+--mistake $2}
TIMEOUT=${TIMEOUT:-600}
cd "$(dirname "$0")"
python monitor.py --expected $((3 * N)) --timeout "$TIMEOUT" &
MON=$!
python layer2.py $MIS &
for k in 0 1 2; do python layer1.py --cell $k --n "$N" $MIS & done
sleep 15
echo "---- ros2 node list"; ros2 node list --no-daemon
echo "---- ros2 topic list"; ros2 topic list --no-daemon
echo "---- ros2 topic info /part_data"; ros2 topic info /part_data --no-daemon
wait $MON
sleep 3
kill $(jobs -p) 2>/dev/null   # 終わっていないノードがあれば止める
wait 2>/dev/null
echo "---- all nodes stopped"
