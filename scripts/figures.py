"""レポート用の図を作る。Docker でも同じ図になるよう、文字は英語にしている（コンテナに日本語フォントがないため）。
使い方: python scripts/figures.py
出力: outputs/fig/mu_ranges.png, architecture.png, time_breakdown.png
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "fig"
OUT.mkdir(parents=True, exist_ok=True)


def mu_ranges():
    """文献値の幅（原文で確かめた値のみ）。材質 × 相手材ごとに 1 本の帯。"""
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "friction_literature.csv")) if r["verified"] == "yes" and r["mu_s_min"]]
    groups = {}
    for r in rows:
        key = f"{r['material'].split(' (')[0]} vs {r['counterpart'].split(' (')[0]}"
        lo, hi = float(r["mu_s_min"]), float(r["mu_s_max"])
        g = groups.setdefault(key, [lo, hi, set()])
        g[0], g[1] = min(g[0], lo), max(g[1], hi)
        g[2].add(r["source_id"])
    items = sorted(groups.items(), key=lambda kv: kv[1][0])
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(items) + 1))
    for i, (k, (lo, hi, src)) in enumerate(items):
        if hi - lo < 1e-9:
            ax.plot([lo], [i], "s", ms=6, color="0.25")  # 値が 1 つだけ（幅が分からない）
        else:
            ax.plot([lo, hi], [i, i], lw=7, color="0.25", solid_capstyle="butt")
        ax.text(hi + 0.03, i, f"{lo:.2f}-{hi:.2f}  [{', '.join(sorted(src))}]", va="center", fontsize=7, color="0.35")
    ax.set_yticks(range(len(items)), [k for k, _ in items], fontsize=8)
    ax.set_xlabel("static friction coefficient (literature, verified values only)")
    ax.set_xlim(0, 2.1)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "mu_ranges.png", dpi=160)


def architecture():
    """dora の配線（dora/dataflow.yml）を、判断の中身つきで描く。"""
    nodes = [
        ("camera", "tray image\n(Genesis render)"),
        ("vlm", "Gemini Robotics-ER 2\n-> point + material candidates"),
        ("planner", "mu table (literature)\nsafe_mu = min lower bound\n-> lift acceleration"),
        ("executor", "Franka in Genesis\npick, lift, move"),
        ("monitor", "latency, success,\nfriction margin"),
    ]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    xs = [0.1 + i * 2.0 for i in range(len(nodes))]
    for x, (name, desc) in zip(xs, nodes):
        ax.add_patch(FancyBboxPatch((x, 0.6), 1.7, 1.4, boxstyle="round,pad=0.02", fc="white", ec="black", lw=1.2))
        ax.text(x + 0.85, 1.75, name, ha="center", fontsize=10, weight="bold")
        ax.text(x + 0.85, 1.05, desc, ha="center", va="center", fontsize=7)
    labels = ["image (Arrow)", "parts", "plan (queue 100)", "result"]
    for x, lab in zip(xs[:-1], labels):
        ax.annotate("", xy=(x + 2.0, 1.3), xytext=(x + 1.72, 1.3), arrowprops=dict(arrowstyle="->", lw=1.2))
        ax.text(x + 1.86, 1.42, lab, ha="center", fontsize=6.5, color="0.3")
    ax.text(5, 0.25, "dora-rs 1.0.1 dataflow: nodes run in parallel, so VLM latency overlaps with execution",
            ha="center", fontsize=8, color="0.3")
    fig.tight_layout()
    fig.savefig(OUT / "architecture.png", dpi=160)


def time_breakdown():
    """戦略 × 材質ごとの、持ち上げ＋移動の時間（outputs/eval の結果から）。"""
    ev = ROOT / "outputs" / "eval" / "metrics.json"
    if not ev.exists():
        print("skip time_breakdown: run sim/evaluate.py first")
        return
    m = json.load(open(ev))
    strategies = [s for s in ("A", "B", "C-VLM", "C") if s in m["summary"]]
    mats = m["materials"]
    t = {(r["strategy_name"], r["material"]): r["t_lift"] + r["t_move"] for r in m["runs"]}
    sr = {(r["strategy_name"], r["material"]): r["success_rate"] for r in m["runs"]}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.8 / len(mats)
    shades = ["0.2", "0.5", "0.8"]
    for j, mat in enumerate(mats):
        xs = [i + (j - (len(mats) - 1) / 2) * w for i in range(len(strategies))]
        vals = [t[(s, mat)] for s in strategies]
        ax.bar(xs, vals, w, label=mat, color=shades[j % 3], edgecolor="black", lw=0.5)
        for x, s, v in zip(xs, strategies, vals):
            if sr[(s, mat)] < 1:
                ax.text(x, v + 0.05, f"{sr[(s, mat)] * 100:.0f}%", ha="center", fontsize=7, color="black")
    ax.set_xticks(range(len(strategies)), strategies)
    ax.set_ylabel("lift + move time [s]")
    ax.legend(fontsize=8)
    ax.set_title("time per part (labels: success rate when below 100%)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "time_breakdown.png", dpi=160)


if __name__ == "__main__":
    mu_ranges()
    architecture()
    time_breakdown()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
