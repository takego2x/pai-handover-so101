#!/usr/bin/env bash
# V06 金属の質感指定（Aluminium()/Iron()）だと、アルミも鋼も白い樹脂に見える。色を寄せた版と並べる。
source "$(dirname "$0")/_lib.sh"
patch_copy sim/tray_camera.py $W/tray_metal_presets.py \
  'gs.surfaces.Metal(color=(0.80, 0.81, 0.83), roughness=0.3, metallic=1.0)' 'gs.surfaces.Aluminium(roughness=0.35)'
python - <<'PY'
s = open("outputs/verify/tray_metal_presets.py").read()
s = s.replace("gs.surfaces.Metal(color=(0.42, 0.44, 0.47), roughness=0.2, metallic=1.0)", "gs.surfaces.Iron(roughness=0.25)")
open("outputs/verify/tray_metal_presets.py", "w").write(s)
PY
quiet python $W/tray_metal_presets.py $W/metal_presets.png 0
quiet python sim/tray_camera.py $W/metal_colored.png 0
echo "見比べる: open $W/metal_presets.png $W/metal_colored.png"
