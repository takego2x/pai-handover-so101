#!/bin/bash
# スライド（outputs/video2/slides_take.pptx）を、ページごとの 1280x720 の画像（outputs/video2/pages/<名前>.png）にする。
# 名前は slides_take_layout.json の order の順。LibreOffice で PDF にしてから pdftoppm で画像に（Meiryo は LibreOffice の個人用の書体フォルダ）。
#   bash video/render_pages.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/video2
TMP=$(mktemp -d)
soffice --headless --convert-to pdf --outdir "$TMP" "$OUT/slides_take.pptx" >/dev/null
pdftoppm -png -scale-to-x 1280 -scale-to-y 720 "$TMP/slides_take.pdf" "$TMP/p"
names=($(python3 -c "import json;print(' '.join(json.load(open('$OUT/slides_take_layout.json'))['order']))"))
imgs=($(ls "$TMP"/p-*.png | sort))
if [ ${#names[@]} -ne ${#imgs[@]} ]; then
  echo "ページ数が合わない：名前 ${#names[@]}、画像 ${#imgs[@]}" >&2; exit 1
fi
mkdir -p "$OUT/pages"
for i in "${!names[@]}"; do
  cp "${imgs[$i]}" "$OUT/pages/${names[$i]}.png"
done
echo "pages: ${#names[@]} 枚 → $OUT/pages/"
rm -rf "$TMP"
