#!/usr/bin/env bash
# V21 公開すると氏名が出てしまう 2 か所を確かめる：dora の実行ログの絶対パス、commit の作者メール。
source "$(dirname "$0")/_lib.sh"
echo "== dora/out のログに、ホームの絶対パスが入っているか"
grep -rl "$HOME" dora/out 2>/dev/null | head -3 || true
git check-ignore -q dora/out && echo "dora/out は git の対象外（OK）" || echo "dora/out が git の対象になっている（NG）"
echo "== これまでの commit の作者"
git log --format='%an <%ae>' | sort -u
