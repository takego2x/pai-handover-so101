"""一律の戦略（A）の代表値を振り、成功率と時間がどう動くかを見る。C / C-VLM と同じ図に置く。
使い方: python sim/sensitivity.py [n_seeds] [--eval outputs/eval/metrics.json]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATERIALS = ["POM", "aluminium", "steel"]
MU_NOMS = [0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.61]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_seeds", nargs="?", type=int, default=20)
    ap.add_argument("--eval", default=str(ROOT / "outputs" / "eval" / "metrics.json"))
    ap.add_argument("--plot-only", action="store_true", help="計算せず、保存済みの sensitivity.json から図だけ作る")
    args = ap.parse_args()
    out = ROOT / "outputs" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)

    rows = json.load(open(out / "sensitivity.json")) if args.plot_only else []
    for mu in ([] if args.plot_only else MU_NOMS):
        ok = n = 0
        t, margins = [], []
        for m in MATERIALS:
            f = out / f"A-{mu}-{m}.json"
            r = subprocess.run([sys.executable, str(ROOT / "sim" / "transport.py"), m, "A", str(args.n_seeds),
                                "--mu-assumed", str(mu), "--json", str(f)], capture_output=True, text=True)
            if r.returncode != 0 or not f.exists():
                sys.exit(f"FAILED mu={mu} {m}\n{r.stderr[-1500:]}")
            d = json.load(open(f))
            ok += sum(d["success"])
            n += len(d["success"])
            t.append(d["t_lift"] + d["t_move"])
            margins += d["margin"]
        rows.append({"mu_nom": mu, "success_rate": ok / n, "mean_lift_move_s": sum(t) / len(t),
                     "min_margin": min(margins)})
        print(f"A(mu={mu:.2f})  success {ok}/{n}  t={sum(t) / len(t):.2f}s  min_margin={min(margins) * 100:+.1f}%")
    if not args.plot_only:
        json.dump(rows, open(out / "sensitivity.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.axhspan(-100, 0, color="0.9")  # 余裕が負＝落とす
    ax.axhline(15, ls="--", color="0.5", lw=0.8)
    ax.text(0.65, 16.5, "design margin 15%", fontsize=8, color="0.4")
    ax.plot([r["mean_lift_move_s"] for r in rows], [r["min_margin"] * 100 for r in rows], "o-", color="0.55",
            label="uniform mu (strategy A, varied)")
    seen = {}
    for r in rows:  # 同じ点に重なる代表値は 1 つのラベルにまとめる
        key = (round(r["mean_lift_move_s"], 2), round(r["min_margin"], 3))
        seen.setdefault(key, []).append(r)
    for (_, _), grp in seen.items():
        r0 = grp[0]
        mus = f"{grp[0]['mu_nom']:.2f}" if len(grp) == 1 else f"{grp[0]['mu_nom']:.2f}-{grp[-1]['mu_nom']:.2f}"
        ax.annotate(f"mu={mus}  ({r0['success_rate'] * 100:.0f}%)", (r0["mean_lift_move_s"], r0["min_margin"] * 100),
                    textcoords="offset points", xytext=(6, 6), fontsize=7, color="0.4")
    ev = Path(args.eval)
    if ev.exists():
        summ = json.load(open(ev))["summary"]
        for name in ("B", "C", "C-VLM"):
            if name in summ:
                s = summ[name]
                if "min_margin" not in s:
                    continue
                ax.scatter(s["mean_lift_move_s"], s["min_margin"] * 100, s=90, color="black", zorder=3)
                ax.annotate(f"{name} ({s['success_rate'] * 100:.0f}%)", (s["mean_lift_move_s"], s["min_margin"] * 100),
                            textcoords="offset points", xytext=(-10, 10))
    ax.set_xlabel("mean lift + move time [s]  (shorter is better)")
    ax.set_ylabel("worst-case friction margin [%]  (below 0 = drop)")
    ax.set_ylim(-40, 40)
    ax.set_xlim(0.6, 2.9)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sensitivity.png", dpi=150)
    print("wrote", out / "sensitivity.png")


if __name__ == "__main__":
    main()
