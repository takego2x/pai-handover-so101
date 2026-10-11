#!/usr/bin/env bash
# Fetch the Microduck robot description at a pinned commit into third_party/.
# The .stl meshes are CC BY-NC-SA 4.0 (Pollen Robotics) and are NOT redistributed in this repo.
# https://github.com/pollen-robotics/microduck_rl  (see LICENSE-MODELS.md there)
set -euo pipefail
COMMIT=ceca017
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/microduck_rl"
if [ ! -d "$DEST/.git" ]; then
  git clone -q --filter=blob:none --sparse https://github.com/pollen-robotics/microduck_rl "$DEST"
fi
git -C "$DEST" sparse-checkout set src/mjlab_microduck/robot/microduck
git -C "$DEST" checkout -q "$COMMIT"
test -f "$DEST/src/mjlab_microduck/robot/microduck/assets/jaw_soft.stl" || { echo "assets missing" >&2; exit 1; }
echo "$DEST/src/mjlab_microduck/robot/microduck/assets"
