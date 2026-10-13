"""S7b: トレイの画像から、部品ごとの位置と「材質の候補」を VLM に答えさせ、正解と突き合わせる。

使い方:
  python sim/vlm_material.py gemini outputs/tray/tray_*.png     # Gemini Robotics-ER 2（要 GEMINI_API_KEY）
  python sim/vlm_material.py qwen   outputs/tray/tray_*.png     # ローカル Qwen（Mac のみ。QWEN_PYTHON に mlx-vlm 入りの python）
  python sim/vlm_material.py replay outputs/tray/tray_*.png     # 保存済みの応答を読み直す（キー不要・CI 用）

出力:
  outputs/vlm/<画像名>.<backend>.txt   VLM の生の応答（記録として残す。replay はこれを読む）
  outputs/vlm/score_<backend>.json     正答率・遅延
  outputs/vlm_candidates.json          材質ごとの候補（画像ごと）。教材は data/vlm_candidates_gemini.json に写したものを使う
"""
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "vlm"
MATERIALS = ["POM", "ABS", "aluminium", "steel", "NBR"]  # data/friction_literature.csv の材質名にそろえる
GEMINI_MODEL = "gemini-robotics-er-2-preview"
TIMEOUT_S = int(os.environ.get("GEMINI_TIMEOUT_S", "120"))  # 1 回の呼び出しの上限時間
QWEN_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

PROMPT = f"""This is a top-down image of a blue tray with industrial parts on it.
For EVERY part on the tray, return one JSON object.
- "point": [y, x] of the part's center, normalized to 0-1000.
- "label": a short visual description (color, shape, finish).
- "candidates": the materials it could be, chosen ONLY from {MATERIALS}.
  If you are not sure, list every material it could plausibly be. Never guess a single material when unsure.
Return only a JSON array, no prose."""


def load_dotenv():
    """リポジトリ直下の .env から KEY=VALUE を読む。.env を優先する（シェルの設定に別のキーが残っていても、
    この課題用のキーを使うため。~/.zshrc に別プロジェクトのキーがあり、それが使われて 429 になった）。"""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"')


class CallTimedOut(Exception):
    pass


def _on_alarm(signum, frame):
    raise CallTimedOut("call timed out (whole call)")


def ask_gemini(image):
    from google import genai  # pip install google-genai

    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY がありません（.env に書く）")

    from google.genai import types

    # 応答がないと永久に待ってしまう（2026-10-11 の夜、1 枚目で 8 時間止まった）ので、上限時間を付ける
    # SDK 自身の自動再試行は切る（429 などで待ち時間を伸ばしながら待ち続け、上限時間が効かなかった）
    client = genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000,
                                                         retry_options=types.HttpRetryOptions(attempts=1)))
    last = None
    for attempt in (1, 2):  # 時間切れのときだけ 1 回やり直す
        # 呼び出し全体（アップロード・処理の完了待ちを含む）を OS のタイマーで打ち切る。
        # 通信 1 回の上限だけでは、SDK の完了待ち（sleep しながら待つ）が止まらなかった
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(TIMEOUT_S)
        try:
            f = client.files.upload(file=str(image))
            resp = client.interactions.create(
                model=GEMINI_MODEL,
                input=[{"type": "image", "uri": f.uri, "mime_type": f.mime_type},
                       {"type": "text", "text": PROMPT}],
                generation_config={"thinking_level": "medium"},  # 公式は遅延と性能の釣り合いに medium を推奨
            )
            return resp.output_text
        except Exception as e:  # noqa: PERF203
            if "timed out" not in str(e).lower() and "timeout" not in type(e).__name__.lower():
                raise
            last = e
            print(f"  {getattr(image, 'name', image)}: {TIMEOUT_S}s で応答なし（{attempt} 回目）", file=sys.stderr)
        finally:
            signal.alarm(0)  # 成功しても失敗しても、タイマーは必ず止める
    raise last


def ask_qwen(image):
    py = os.environ.get("QWEN_PYTHON")
    if not py:
        sys.exit("QWEN_PYTHON に mlx-vlm を入れた python のパスを入れてください（Mac のみ）")
    r = subprocess.run([py, "-I", "-m", "mlx_vlm.generate", "--model", QWEN_MODEL, "--image", str(image),
                        "--max-tokens", "600", "--temperature", "0", "--prompt", PROMPT],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(r.stderr[-1500:])
    return r.stdout


def parse(text):
    """応答から JSON 配列を取り出す。```json の囲み・前後の文章があっても読む。読めなければ例外。"""
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
    if not m:
        raise ValueError(f"no JSON array in response: {text[:200]!r}")
    items = json.loads(m.group(0))
    out = []
    for it in items:
        pt = it.get("point") or it.get("point_2d")
        cands = [c for c in it.get("candidates", []) if c in MATERIALS]
        if not (isinstance(pt, list) and len(pt) == 2):
            continue
        out.append({"point": [float(pt[0]), float(pt[1])], "label": it.get("label", ""), "candidates": cands})
    return out


def score(pred, truth):
    """正解の部品ごとに、いちばん近い予測点を対応づけ、候補に正解が入っているかを見る。"""
    rows = []
    for t in truth:
        ty, tx = t["point_1000"]
        best = min(pred, key=lambda p: (p["point"][0] - ty) ** 2 + (p["point"][1] - tx) ** 2, default=None)
        dist = None if best is None else ((best["point"][0] - ty) ** 2 + (best["point"][1] - tx) ** 2) ** 0.5
        cands = best["candidates"] if best and dist < 100 else []  # 0〜1000 の座標で 100 以上離れたら対応なし
        rows.append({"truth": t["label"], "candidates": cands, "dist_1000": None if dist is None else round(dist, 1),
                     "hit": t["label"] in cands, "exact": cands == [t["label"]]})
    return rows


def main():
    backend, images = sys.argv[1], [Path(p) for p in sys.argv[2:]]
    ask = {"gemini": ask_gemini, "qwen": ask_qwen}.get(backend)
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, latencies = [], []
    for img in images:
        raw_path = OUT / f"{img.stem}.{'gemini' if backend == 'replay' else backend}.txt"
        if backend == "replay":
            raw_path = next(iter(sorted(OUT.glob(f"{img.stem}.*.txt"))), None) or \
                next(iter(sorted((ROOT / "data" / "vlm_responses_gemini").glob(f"{img.stem}.*.txt"))), None)
            if raw_path is None:
                sys.exit(f"保存済みの応答がない: {img.name}（S7b を gemini で実行するか、data/vlm_responses_gemini/ を確認）")
            text = raw_path.read_text()
        else:
            t0 = time.perf_counter()  # 経過時間は monotonic な時計で測る（第3回）
            try:
                text = ask(img)
            except Exception as e:  # API の失敗は、次にすることが分かる形で止める
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                hint = {403: "キーに制限が付いていない（AI Studio でキーに制限を付ける）",
                        429: "プロジェクトの利用上限（https://ai.studio/spend）。上限を上げるか、手順書 S7b の"
                             "「Gemini が使えないとき」で保存済みの応答を使う"}.get(code, "")
                if not hint and "timed out" in str(e).lower():
                    hint = (f"{TIMEOUT_S} 秒待っても応答がない。時間をおいて再実行するか、"
                            "手順書 S7b の「Gemini が使えないとき」で保存済みの応答を使う")
                sys.exit(f"Gemini の呼び出しに失敗: {code} {str(e)[:160]}\n→ {hint}")
            latencies.append(time.perf_counter() - t0)
            raw_path.write_text(text)
        truth = json.load(open(img.with_suffix(".json")))
        rows = score(parse(text), truth)
        for r in rows:
            r["image"] = img.name
        all_rows += rows
        print(f"{img.name}: hit {sum(r['hit'] for r in rows)}/{len(rows)}  exact {sum(r['exact'] for r in rows)}/{len(rows)}"
              f"  {[(r['truth'], r['candidates']) for r in rows]}")

    n = len(all_rows)
    summary = {"backend": backend, "n_parts": n,
               "hit_rate": round(sum(r["hit"] for r in all_rows) / n, 3),
               "exact_rate": round(sum(r["exact"] for r in all_rows) / n, 3),
               "mean_n_candidates": round(statistics.mean(len(r["candidates"]) for r in all_rows), 2)}
    if latencies:
        lat = sorted(latencies)
        summary |= {"latency_median_s": round(statistics.median(lat), 2),
                    "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2)}
    json.dump({"summary": summary, "rows": all_rows}, open(OUT / f"score_{backend}.json", "w"), indent=1)

    # 材質ごとの候補を画像ごとに残す（部品は 1 回撮った画像の候補で判断するため）。
    # 候補が空＝見落としは「UNKNOWN」とし、μ の表で引けない → 戦略 C は最悪値に倒れる
    per_image = {}
    for r in all_rows:
        per_image.setdefault(r["truth"], []).append(r["candidates"] or ["UNKNOWN"])
    json.dump({"backend": backend, "per_image": per_image}, open(ROOT / "outputs" / "vlm_candidates.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
