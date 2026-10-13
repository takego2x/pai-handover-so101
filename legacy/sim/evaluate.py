"""S9: 材質 × 戦略を固定 seed で回し、metrics.json と散布図を出す。
使い方: python sim/evaluate.py [n_seeds] [--vlm outputs/vlm_candidates.json]
  --vlm を渡すと、戦略 C に VLM の候補を使う「C-VLM」も評価する。
"""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATERIALS = ["POM", "aluminium", "steel"]


def run(material, strategy, n, out, candidates=None):
    cmd = [sys.executable, str(ROOT / "sim" / "transport.py"), material, strategy, str(n), "--json", str(out)]
    if candidates:
        cmd += ["--candidates", ",".join(candidates)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():  # 戻り値を捨てない：失敗はその場で止める
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return json.load(open(out))


def run_per_image(material, n, out, per_image):
    """C-VLM：seed を画像の数で分け、それぞれの seed には、その画像で VLM が挙げた候補を使う。"""
    k = len(per_image)
    merged = None
    for i, cands in enumerate(per_image):
        n_i = n // k + (1 if i < n % k else 0)
        if n_i == 0:
            continue
        seed0 = 1000 + sum(n // k + (1 if j < n % k else 0) for j in range(i))
        cmd_out = out / f"{material}-C-VLM-img{i}.json"
        cmd = [sys.executable, str(ROOT / "sim" / "transport.py"), material, "C", str(n_i), "--seed0", str(seed0),
               "--json", str(cmd_out), "--candidates", ",".join(cands)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not cmd_out.exists():
            sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stderr[-2000:]}")
        part = json.load(open(cmd_out))
        part["t_each"] = [part["t_lift"] + part["t_move"]] * n_i
        part["candidates_each"] = [cands] * n_i
        if merged is None:
            merged = part
        else:
            for key in ("seeds", "true_mu", "a_max_true", "success", "t_each", "candidates_each", "margin"):
                merged[key] += part[key]
    merged["success_rate"] = sum(merged["success"]) / len(merged["success"])
    # 平均時間は seed ごとの時間の平均（画像ごとに候補が違えば、計画も時間も違う）
    t = sum(merged["t_each"]) / len(merged["t_each"])
    merged["t_lift"], merged["t_move"] = t, 0.0
    json.dump(merged, open(out / f"{material}-C-VLM.json", "w"), indent=1)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_seeds", nargs="?", type=int, default=20)
    ap.add_argument("--vlm", help="sim/vlm_material.py が書く outputs/vlm_candidates.json（画像ごとの候補）")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plans = [(s, None) for s in ["A", "B", "C"]]
    vlm = json.load(open(args.vlm)) if args.vlm else None
    if vlm:
        plans.append(("C-VLM", vlm))

    runs, summary = [], {}
    for name, cands in plans:
        per = []
        for m in MATERIALS:
            if name == "C-VLM":
                r = run_per_image(m, args.n_seeds, out, cands["per_image"][m])
            else:
                r = run(m, name, args.n_seeds, out / f"{m}-{name}.json")
            r["strategy_name"] = name
            print(f"{name:6s} {m:9s} success {sum(r['success'])}/{len(r['success'])}  t={r['t_lift'] + r['t_move']:.2f}s")
            per.append(r)
            runs.append(r)
        n_ok = sum(sum(r["success"]) for r in per)
        n_all = sum(len(r["success"]) for r in per)
        summary[name] = {
            "success_rate": round(n_ok / n_all, 4), "n": n_all,
            "mean_lift_move_s": round(sum(r["t_lift"] + r["t_move"] for r in per) / len(per), 3),
            "min_margin": round(min(min(r["margin"]) for r in per), 4),  # 最悪の部品での摩擦の余裕
        }

    metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip() or None,
        "platform": platform.platform(), "python": platform.python_version(),
        "n_seeds": args.n_seeds, "seed0": 1000, "materials": MATERIALS,
        "summary": summary, "runs": runs,
    }
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, s in summary.items():
        ax.scatter(s["mean_lift_move_s"], s["success_rate"] * 100, s=80, color="black")
        ax.annotate(name, (s["mean_lift_move_s"], s["success_rate"] * 100), textcoords="offset points", xytext=(8, -4))
    ax.set_xlabel("mean lift + move time [s]  (shorter is better)")
    ax.set_ylabel("success rate [%]")
    ax.set_ylim(min(80, min(s["success_rate"] * 100 for s in summary.values()) - 5), 102)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=150)
    print("wrote", out / "metrics.json", out / "tradeoff.png")


if __name__ == "__main__":
    main()
