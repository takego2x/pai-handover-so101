#!/usr/bin/env bash
# V19 公開前の点検が、本当に検出できるかを「わざと禁止語を入れたファイル」で確かめる（確かめた後に消す）。
source "$(dirname "$0")/_lib.sh"
f=zz_negative_test.txt
python -c "print('担当: ' + chr(0x5927) + chr(0x5b9d) + '工業')" > $f   # 禁止語はこのファイル自体に書かない（書くと点検が常に NG になる）
bash scripts/preflight.sh | sed -n '/公開前/,$p' || true
rm -f $f
echo "→ 上に「公開してはいけない語がある」と $f の行が出ていれば、点検は働いている"
