#!/usr/bin/env bash
# report/report.html を PDF にする（Brave / Chrome の画面なし印刷）。使い方: scripts/make_pdf.sh <omnicampusアカウント名>
set -euo pipefail
NAME="${1:?omnicampus のアカウント名を渡してください}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/outputs/説明資料_PAI最終課題_${NAME}.pdf"
for B in "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; do
  [ -x "$B" ] && BROWSER="$B" && break
done
[ -n "${BROWSER:-}" ] || { echo "Brave / Chrome が見つかりません" >&2; exit 1; }
mkdir -p "$ROOT/outputs"
"$BROWSER" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$OUT" "file://$ROOT/report/report.html" 2>/dev/null
# 出口で実測：ファイルがあり、ページ数が取れること
test -s "$OUT" || { echo "PDF が作られませんでした" >&2; exit 1; }
PAGES=$(python3 -c "import re,sys;d=open(sys.argv[1],'rb').read();print(len(re.findall(rb'/Type\s*/Page[^s]',d)))" "$OUT")
echo "$OUT  ${PAGES} pages"
