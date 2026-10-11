#!/usr/bin/env bash
# V13 VLM の候補を全画像で合計（和集合）して使うと、C-VLM が B と同じ遅さになる。画像ごとに使う版と比べる。
source "$(dirname "$0")/_lib.sh"
test -f outputs/vlm_candidates.json || { echo "先に S7b（sim/vlm_material.py）を実行する"; exit 1; }
python - <<'PY'
import json
d = json.load(open("outputs/vlm_candidates.json"))
union = {k: sorted({c for cs in v for c in cs}) for k, v in d["per_image"].items()}
json.dump({"per_image": {k: [u] for k, u in union.items()}}, open("outputs/verify/vlm_union.json", "w"), indent=1)
print("union candidates:", union)
PY
for m in aluminium steel; do
  printf "union   %-9s " $m; quiet python sim/transport.py $m C 5 --candidates "$(python -c "import json;print(','.join(json.load(open('$W/vlm_union.json'))['per_image']['$m'][0]))")" | grep -o "mu_assumed=.*t_lift+move=[0-9.]*s"
done
echo "per-image の結果は S9（evaluate.py --vlm）の C-VLM の行と比べる"
