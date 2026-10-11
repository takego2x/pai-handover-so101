"""monitor：各段の時刻と結果を記録し、認識と実行が重なっていたかを測る。"""
import json
import time

from dora import Node

from common import ROOT

node = Node()
vlm, results = {}, []
closed = set()
for event in node:
    if event["type"] == "INPUT":
        md = event["metadata"]
        if event["id"] == "parts":
            vlm[md["tray"]] = {"t_cam": md["t_cam"], "t_vlm_start": md["t_vlm_start"], "t_vlm_done": md["t_vlm_done"]}
        else:
            results.append(json.loads(event["value"].to_pylist()[0]))
    elif event["type"] == "INPUT_CLOSED":
        closed.add(event["id"])
        if closed >= {"parts", "result"}:
            break

if not vlm or not results:  # 上流が 1 件も処理できなかった（VLM の失敗など）。落ちずに理由を出す
    print(f"[monitor] no data: trays={len(vlm)} results={len(results)}。vlm ノードのログを確認する", flush=True)
    raise SystemExit(1)
t0 = min(v["t_cam"] for v in vlm.values())
done = [r for r in results if "t_lift_move" in r]
summary = {
    "trays": len(vlm), "parts": len(results), "simulated": len(done),
    "dropped": sum(r["status"] == "DROPPED" for r in done),
    "vlm_latency_s": [round(v["t_vlm_done"] - v["t_vlm_start"], 2) for v in vlm.values()],
    "wall_s": round(max(r["t_exec_done"] for r in results) - t0, 2),
    "exec_busy_s": round(sum(r["t_exec_done"] - r["t_exec_start"] for r in results), 2),
    "vlm_busy_s": round(sum(v["t_vlm_done"] - v["t_vlm_start"] for v in vlm.values()), 2),
}
# 認識の時間のうち、実行と重なっていた割合（重なるほど認識の遅さが隠れている）
exec_iv = [(r["t_exec_start"], r["t_exec_done"]) for r in results]
overlap = sum(max(0.0, min(ve, ee) - max(vs, es)) for v in vlm.values()
              for (vs, ve) in [(v["t_vlm_start"], v["t_vlm_done"])] for (es, ee) in exec_iv)
summary["vlm_overlapped_with_exec_s"] = round(overlap, 2)
out = ROOT / "outputs" / "dora"
out.mkdir(parents=True, exist_ok=True)
json.dump({"summary": summary, "results": results, "vlm": vlm}, open(out / "run.json", "w"), indent=1)
print("[monitor]", json.dumps(summary), flush=True)
