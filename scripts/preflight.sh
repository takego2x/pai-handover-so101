#!/usr/bin/env bash
# 作業を始める前の事前チェック。足りないものを一度に全部出す（途中で止まらない）。
# 使い方: bash scripts/preflight.sh
cd "$(dirname "$0")/.." || exit 1
ok=0; ng=0
pass() { printf '  OK   %s\n' "$1"; ok=$((ok+1)); }
fail() { printf '  NG   %s\n       → %s\n' "$1" "$2"; ng=$((ng+1)); }

echo "== 道具"
command -v uv >/dev/null && pass "uv $(uv --version | cut -d' ' -f2)" || fail "uv がない" "brew install uv"
command -v git >/dev/null && pass "git" || fail "git がない" "xcode-select --install"
command -v ffmpeg >/dev/null && pass "ffmpeg" || fail "ffmpeg がない" "brew install ffmpeg"
uv python find 3.12 >/dev/null 2>&1 && pass "Python 3.12 が使える" || fail "Python 3.12 がない" "uv python install 3.12"

echo "== Python 環境（.venv）"
if [ -x .venv/bin/python ]; then
  v=$(.venv/bin/python -c 'import sys;print("%d.%d"%sys.version_info[:2])')
  [ "$v" = "3.12" ] && pass ".venv は Python $v" || fail ".venv が Python $v" "rm -rf .venv && uv venv -p 3.12 .venv"
  .venv/bin/python -c 'import genesis' 2>/dev/null && pass "genesis を import できる" || fail "genesis が入っていない" "uv pip install genesis-world==1.4.3 torch==2.14.1"
else
  fail ".venv がない" "uv venv -p 3.12 .venv（S1）"
fi

echo "== Docker"
if docker info >/dev/null 2>&1; then
  pass "Docker が動いている（$(docker info --format '{{.Architecture}}, {{.NCPU}} CPU'))"
else
  fail "Docker が動いていない" "open -a Docker（起動に 10〜30 秒）"
fi

echo "== Gemini（S7b から必要）"
if [ -f .env ] && grep -q '^GEMINI_API_KEY=..*' .env; then
  pass ".env に GEMINI_API_KEY がある（値は表示しない）"
  git check-ignore -q .env && pass ".env は git の対象外" || fail ".env が git の対象になっている" ".gitignore に .env を足す"
else
  fail ".env に GEMINI_API_KEY がない" "AI Studio でキーに制限を付けて作り、.env に GEMINI_API_KEY=... と書く"
fi

echo "== 公開前の点検（S10 から）"
if git rev-parse --git-dir >/dev/null 2>&1; then
  # commit の作者メールは GitHub で誰でも見られる。氏名の入ったメールにしない
  email=$(git config user.email)
  case "$email" in
    *@users.noreply.github.com) pass "commit の作者メールは匿名用（noreply）" ;;
    *) fail "commit の作者メールが ${email}" "git config user.email \"<ID>+<login>@users.noreply.github.com\"（手順書 S1）" ;;
  esac
  # 未 commit のファイルも含める（.gitignore 済みの .venv / third_party / outputs は除く）
  n=$(git ls-files --cached --others --exclude-standard | grep -c .)
  hits=$(git grep --untracked -n -i -E "${PRIVATE_WORDS:-@gmail}" -- . ':!scripts/preflight.sh' 2>/dev/null | head -5)
  if [ "$n" -eq 0 ]; then
    fail "検査するファイルが 0 件" "リポジトリの場所が正しいか確認する"
  elif [ -z "$hits" ]; then
    pass "氏名・所属・社名は見つからない（${n} ファイルを検査）"
  else
    fail "公開してはいけない語がある" "$hits"
  fi
fi

echo
echo "OK ${ok} / NG ${ng}"
[ "$ng" -eq 0 ]
