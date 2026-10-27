"""絵コンテ（video/storyboard_slides.json）の映像の切れ目と長さを、録画の記録（outputs/video2/<名前>.json）から入れ直す。

  p5：1 台目が箱を台座に置いた時刻で 3 ページに分ける（ページごとに、いま映っている箱の行を太字にする）
  p6：全部を 1 ページ（運べない箱が出ない組にしたため、2 ページに分けない）
  p7：全部を 1 ページ
  倍速は絵コンテの値のまま。撮り直したら、これを流してから compose_slides.py

  python3 video/board_from_runs.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "video" / "storyboard_slides.json"


def run(name):
    d = json.loads((ROOT / "outputs" / "video2" / f"{name}.json").read_text())
    return d["steps"] / 100, d["marks"]


def cut(board, prefix, clip, bounds):
    """bounds：クリップの時刻の切れ目 [0, t1, ..., 終わり]。prefix_0, prefix_1, ... のページに順に割り当てる"""
    pages = [seg for seg in board if seg.get("page", "").startswith(prefix)]
    if len(pages) != len(bounds) - 1:
        raise SystemExit(f"{prefix}: ページ {len(pages)} 枚と切れ目 {len(bounds) - 1} 区間が合わない")
    for seg, t0, t1 in zip(pages, bounds, bounds[1:]):
        v = seg["videos"][clip]
        v["start"] = round(t0, 2)
        seg["sec"] = round((t1 - t0) / float(v["speed"]), 3)


board = json.loads(BOARD.read_text())
end, marks = run("p5")
cut(board, "p5_", "p5", [0.0] + [m["step"] / 100 for m in marks if m["who"].startswith("1-") and "台座 0 に置いた" in m["what"]][:2] + [end])
end, marks = run("p6")
cut(board, "p6", "p6", [0.0, end])
end, _ = run("p7")
cut(board, "p7", "p7", [0.0, end])
BOARD.write_text(json.dumps(board, ensure_ascii=False, indent=1))
for seg in board:
    if "videos" in seg:
        print(seg["page"], seg["sec"], {k: (v.get("start"), v.get("speed")) for k, v in seg["videos"].items()})
print("合計", round(sum(seg["sec"] for seg in board), 1), "秒")
