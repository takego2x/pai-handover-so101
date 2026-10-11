#!/usr/bin/env bash
# ハンズオンの部品取り出し：ステップに必要なファイルを、この答え合わせ用リポジトリから作業場所へコピーする。
# 使い方（作業場所 pai-final の直下で）: bash ~/Documents/AgentProject/pai-final-mu/scripts/kit.sh S8
#   すでにあるファイルは上書きしない（自分で書き換えた分を守る）。上書きしたいときは --force。
#   --diff を付けると、コピーせずに作業場所との差分だけを見せる（答え合わせ）。
set -euo pipefail
REF="$(cd "$(dirname "$0")/.." && pwd)"
STEP="${1:?ステップ番号（例: S8）を渡してください}"
MODE="${2:-}"

case "$STEP" in
  S1)  FILES=".gitignore LICENSE requirements.txt scripts/preflight.sh" ;;
  S2)  FILES="sim/s2_drop_box.py" ;;
  S3)  FILES="sim/s3_franka_ik.py" ;;
  S4)  FILES="sim/s4_grasp_one.py sim/s4b_mu_sweep.py" ;;
  S5)  FILES="Dockerfile compose.yaml .dockerignore" ;;
  S6)  FILES="data/friction_literature.csv data/SOURCES.md sim/mu_table.py" ;;
  S7a) FILES="sim/tray_camera.py" ;;
  S7b) FILES="sim/vlm_material.py" ;;
  S8)  FILES="sim/transport.py" ;;
  S9)  FILES="sim/evaluate.py sim/sensitivity.py" ;;
  S9b) FILES="dora/dataflow.yml dora/common.py dora/camera.py dora/vlm_node.py dora/planner.py dora/executor.py dora/monitor.py" ;;
  S10) FILES="scripts/make_video.py video/storyboard_lv0.json report/report.html scripts/make_pdf.sh scripts/figures.py README.md" ;;
  S11) FILES="scripts/fetch_microduck_assets.sh sim/beak_mu_sweep.py" ;;
  S12) FILES="tests/test_core.py scripts/check_metrics.py .github/workflows/ci.yml" ;;
  S13) FILES="spikes/line_two_arms.py sim/line.py" ;;
  *) echo "知らないステップ: $STEP（S1〜S13, S7a, S7b, S9b）" >&2; exit 1 ;;
esac

[ "$(pwd)" = "$REF" ] && { echo "答え合わせ用リポジトリの中では実行しない（作業場所 pai-final で実行する）" >&2; exit 1; }

for f in $FILES; do
  if [ "$MODE" = "--diff" ]; then
    if [ -f "$f" ]; then diff -u "$f" "$REF/$f" && echo "same: $f"; else echo "missing: $f"; fi
    continue
  fi
  if [ -e "$f" ] && [ "$MODE" != "--force" ]; then
    echo "skip (already exists): $f"
    continue
  fi
  mkdir -p "$(dirname "$f")"
  cp "$REF/$f" "$f"
  echo "copied: $f"
done
