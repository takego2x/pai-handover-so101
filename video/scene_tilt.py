"""動画の場面：力の余裕が小さい（6%）と、部品が滑って回り、指先に角でぶら下がる（L1 の ✏️、μ 0.30・3 N）。
lesson/hold.py の FORCES だけを 3・6・12・24 N に変えて動かす（hold.py そのものは変えない）。
  PAI_CLIP=outputs/video/clip_tilt.mp4 PAI_CLIP_ENV=12 python video/scene_tilt.py
"""
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "lesson" / "hold.py"
src = src_path.read_text().replace("FORCES = [2.0, 5.0, 10.0, 20.0]", "FORCES = [3.0, 6.0, 12.0, 24.0]")
assert "FORCES = [3.0, 6.0, 12.0, 24.0]" in src, "hold.py の FORCES の行が見つからない"
sys.path.insert(0, str(src_path.parent))
sys.argv = [str(src_path)] + sys.argv[1:]
exec(compile(src, str(src_path), "exec"))
