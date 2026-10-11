#!/usr/bin/env bash
# V18 動画のカメラ位置を 4 案で比べる（戦略 A が POM を落とす seed 1004、落ちた直後 4.5 秒の画面）。
source "$(dirname "$0")/_lib.sh"
i=0
for c in "1.5,0.0,0.7|0.5,0.0,0.2" "0.55,-1.0,0.30|0.55,-0.1,0.22" "1.2,-0.9,0.55|0.55,-0.1,0.2" "0.95,-0.55,0.25|0.55,-0.2,0.2"; do
  i=$((i+1)); quiet python sim/transport.py POM A 1 --seed0 1004 --record $W/cam$i.mp4 --cam "${c%%|*}" --lookat "${c##*|}" >/dev/null
  ffmpeg -v error -y -ss 4.5 -i $W/cam$i.mp4 -frames:v 1 $W/cam$i.png
  echo "cam$i  --cam ${c%%|*} --lookat ${c##*|}  -> $W/cam$i.png"
done
