"""L7: レポートに載せる 3 つの図と表を作る。

  図 1  握る力：データあり / データなし（L6 のライン、材質ランダム）
  図 2  データのずれの許容範囲：L4 で μ を 0.6〜1.6 倍にずらしたときの成功率
  図 3  試行の回数：L3 を 3〜10 回で回したときの μ の誤差と、第一層の 1 部品の時間
使い方: python lesson/l7_report.py [--plot-only]
  --plot-only : 計算せず、outputs/lesson/ にある結果から図と表だけ作る
計算にかかる時間（Docker、M 系の Mac）: 約 6 分
"""
import argparse
import json
import subprocess
import sys

import numpy as np

import common as c

OUT = c.ROOT / "outputs" / "lesson"
LESSON = c.ROOT / "lesson"
SCALES = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
TRIALS = [3, 5, 7, 10]


def run(args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=LESSON)
    if r.returncode != 0:  # 失敗をその場で止める（戻り値を捨てない）
        sys.exit(f"FAILED: {' '.join(map(str, args))}\n{r.stderr[-1500:]}")
    return r.stdout


ap = argparse.ArgumentParser()
ap.add_argument("--plot-only", action="store_true")
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)

if not args.plot_only:
    print("line (data / nodata) ...", flush=True)
    run(["l5_line.py", 20, "--json", OUT / "line_data.json"])
    run(["l5_line.py", 20, "--nodata", "--json", OUT / "line_nodata.json"])
    print("tolerance ...", flush=True)
    run(["l3_trials.py", 30, "--seed0", 2000, "--out", OUT / "parts30.json"])
    for s in SCALES:
        run(["l4_carry.py", "--parts", OUT / "parts30.json", "--mu-scale", s, "--json", OUT / f"carry_mu{s}.json"])
    print("trials ...", flush=True)
    for n in TRIALS:
        run(["l3_trials.py", 30, "--seed0", 2000, "--trials", n, "--out", OUT / f"parts30_t{n}.json"])

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

lines = ["# 結果のまとめ（lesson/l7_report.py が自動で書く）", ""]

# 図 1 握る力
d, nd = json.load(open(OUT / "line_data.json")), json.load(open(OUT / "line_nodata.json"))
mats = np.array(d["materials"]).ravel()
Fd, Fn = np.array(d["F_layer2"]).ravel(), np.array(nd["F_layer2"]).ravel()
order = [m for m in c.POOL if (mats == m).any()]
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.boxplot([Fd[mats == m] for m in order], positions=range(len(order)), widths=0.5)
ax.axhline(c.F_MAX, ls="--", color="0.4")
ax.text(len(order) - 1, c.F_MAX + 1.5, "no data: 60 N for every part", ha="right", fontsize=8, color="0.3")
ax.set_xticks(range(len(order)), order)
ax.set_ylabel("grip force per finger [N]")
ax.set_yscale("log")
ax.set_title("Grip force with measured data (layer 2)")
fig.tight_layout()
fig.savefig(OUT / "fig1_force.png", dpi=150)
sd, sn = np.array(d["success"]).ravel(), np.array(nd["success"]).ravel()
lines += ["## 1. 握る力（第一層 3 台 → 第二層 1 台、材質ランダム）", "",
          "| 材質 | 部品の数 | データあり：握る力（中央値） | データなし | 成功（あり / なし） |", "|---|---|---|---|---|"]
for m in order:
    s = mats == m
    lines.append(f"| {m} | {int(s.sum())} | {np.median(Fd[s]):.1f} N | {c.F_MAX:.0f} N | {int(sd[s].sum())}/{int(s.sum())} / {int(sn[s].sum())}/{int(s.sum())} |")
lines += ["", f"第一層の 1 回 {d['cycle_layer1_s']} 秒（うち試行 {d['trials_s']} 秒）、第二層の 1 部品 {d['cycle_layer2_s']} 秒 "
          f"→ 比 {d['cycle_layer1_s'] / d['cycle_layer2_s']:.2f} → 第一層 {int(np.ceil(d['cycle_layer1_s'] / d['cycle_layer2_s']))} 台", ""]

# 図 2 許容範囲
rate = []
for s in SCALES:
    r = json.load(open(OUT / f"carry_mu{s}.json"))
    rate.append(np.mean(r["success"]) * 100)
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(SCALES, rate, "o-", color="black")
ax.axvline(1.0, ls=":", color="0.5")
ax.set_xlabel("mu given to layer 2  /  mu measured by layer 1")
ax.set_ylabel("success rate [%]")
ax.set_ylim(-5, 105)
ax.grid(alpha=0.3)
ax.set_title("How accurate must the data be?")
fig.tight_layout()
fig.savefig(OUT / "fig2_tolerance.png", dpi=150)
lines += ["## 2. データのずれの許容範囲（30 部品）", "", "| μ を何倍にずらしたか | " + " | ".join(f"{s}" for s in SCALES) + " |",
          "|---|" + "---|" * len(SCALES), "| 成功率 | " + " | ".join(f"{x:.0f}%" for x in rate) + " |", ""]

# 図 3 試行の回数
err_med, err_max, tpart = [], [], []
for n in TRIALS:
    r = json.load(open(OUT / f"parts30_t{n}.json"))
    e = np.array([(dd["mu"] / tt["mu"] - 1) * 100 for dd, tt in zip(r["data"], r["truth"]) if dd["ok"]])
    err_med.append(np.median(np.abs(e)))
    err_max.append(np.abs(e).max())
    tpart.append(r["time_per_part_s"])
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(TRIALS, err_max, "o-", color="black", label="worst |mu error|")
ax.plot(TRIALS, err_med, "s--", color="0.5", label="median |mu error|")
ax.set_xlabel("number of trials")
ax.set_ylabel("mu error [%]")
ax2 = ax.twinx()
ax2.plot(TRIALS, tpart, "^:", color="0.3")
ax2.set_ylabel("time per part, layer 1 [s]  (triangles)")
ax.legend(loc="upper center", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig3_trials.png", dpi=150)
lines += ["## 3. 試行の回数（30 部品）", "", "| 試行の回数 | μ の誤差（中央値） | μ の誤差（最大） | 1 部品の時間 |", "|---|---|---|---|"]
for n, a, b, t in zip(TRIALS, err_med, err_max, tpart):
    lines.append(f"| {n} | {a:.1f}% | {b:.1f}% | {t:.1f} 秒 |")
(OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print("wrote", OUT / "fig1_force.png", OUT / "fig2_tolerance.png", OUT / "fig3_trials.png", OUT / "summary.md")
