# verify/ のスクリプトが共通で使う関数。作業場所の直下で実行する前提（venv を有効にしておく）。
set -euo pipefail
cd "$(dirname "$0")/.."
W=outputs/verify; mkdir -p "$W"
# 元のスクリプトを書き換えずに、条件を変えた複製を outputs/verify/ に作る
patch_copy() {  # patch_copy <元> <複製> <置換前> <置換後>
  python - "$1" "$2" "$3" "$4" <<'PY'
import sys
src, dst, old, new = sys.argv[1:5]
s = open(src).read()
if old not in s:
    sys.exit(f"patch target not found in {src}: {old[:60]}")
open(dst, "w").write(s.replace(old, new))
PY
}
quiet() { "$@" 2>/dev/null | grep -v -E "WARNING|frontend_ir|warnings.warn|FutureWarning" || true; }
