#!/usr/bin/env bash
# V07 セグメンテーションの番号は物体の番号と 1 つずれる（背景が 0）。思い込みの版と、対応表で引く版を比べる。
source "$(dirname "$0")/_lib.sh"
patch_copy sim/tray_camera.py $W/tray_naive_seg.py \
  'ys, xs = np.nonzero(np.isin(seg, idxc_of.get(e.idx, [])))' 'ys, xs = np.nonzero(seg == e.idx)'
quiet python $W/tray_naive_seg.py $W/seg_naive.png 0 >/dev/null
quiet python sim/tray_camera.py $W/seg_ok.png 0 >/dev/null
python - <<'PY'
import json
a = json.load(open("outputs/verify/seg_naive.json")); b = json.load(open("outputs/verify/seg_ok.json"))
print(f"{'part':10s} {'world x,y':>16s}  naive px,py   mapped px,py")
for x, y in zip(a, b):
    print(f"{x['label']:10s} {x['x']:7.3f},{x['y']:7.3f}   {x['px']:4d},{x['py']:4d}     {y['px']:4d},{y['py']:4d}")
print("画像（outputs/verify/seg_ok.png）と見比べ、どちらが部品の位置に合うかを確かめる")
PY
