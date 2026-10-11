#!/usr/bin/env bash
# V15 dora の待ち行列が既定のままだと、実行が遅い間に計画が黙って捨てられる。queue_size を外した配線で部品数を数える。
source "$(dirname "$0")/_lib.sh"
python - <<'PY'
s = open("dora/dataflow.yml").read()
s = s.replace("""      plan:
        source: planner/plan
        queue_size: 100                 # 取りこぼさない。既定のままだと、実行が遅い間に古い計画が黙って捨てられた""", "      plan: planner/plan")
assert "plan: planner/plan" in s
open("dora/dataflow_default_queue.yml", "w").write(s)
PY
(cd dora && dora run dataflow_default_queue.yml 2>&1 | grep -o '\[monitor\].*' | sed 's/"vlm_latency.*//')
rm -f dora/dataflow_default_queue.yml
echo "比較: 既定の dataflow.yml では parts=15"
