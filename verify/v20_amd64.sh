#!/usr/bin/env bash
# V20 GitHub の実行環境（amd64）でも通るかを、手元でエミュレーションして確かめる（ビルドに約 4 分、テストに約 1 分半）。
source "$(dirname "$0")/_lib.sh"
docker buildx build --platform linux/amd64 -t pai-final:amd64 --load . > $W/amd64_build.log 2>&1
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work pai-final:amd64 python -m pytest -q tests 2>&1 | tail -1
