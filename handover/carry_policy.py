"""運び方の方策（学習する部分）。手書きの式「握る力 ＝ 力加減 × 3、余裕が足りなければゆっくり」の代わり。

学習：handover/spikes/learn_carry.py の試行（箱ごとに、握る力の倍率 kF × 運ぶ step 数 を並列で試し、潰れ・落下を記録）から、
      「この箱を、この力・この速さで運んだら成功するか」を予測するロジスティック回帰を作る。
      入力（どれもロボットが測れる値）：log 力加減、log 重さ、log(握る力の挟む力 ÷ 推定強度)、log kF、log 速さ（1 / 運ぶ step 数）
方策：候補（kF × 運ぶ step 数）のうち、予測の成功率が P_OK 以上で、挟む力が推定強度の SAFE 倍以下のものから、いちばん速く、
      同じ速さなら弱い力のものを選ぶ。どれも届かなければ None（運ばずに人の確認へ）。

使い方:
  python handover/carry_policy.py train outputs/learn/carry_*.jsonl     # → handover/data/carry_policy.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

SQ_PER_TAU, SAFE, P_OK = 14.5, 0.6, 0.95
KF = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
NC = [60, 100, 150, 250]          # 学習で試した、弧 90° を運ぶ step 数
W_PATH = Path(__file__).resolve().parent / "data" / "carry_policy.json"


def features(found, m, strength, kF, n_carry):
    f = np.asarray(found, float)
    sq = f * np.asarray(kF, float) * SQ_PER_TAU / np.asarray(strength, float)
    return np.stack(np.broadcast_arrays(np.ones_like(f), np.log(f), np.log(np.asarray(m, float)), np.log(sq),
                                        np.log(np.asarray(kF, float)), -np.log(np.asarray(n_carry, float) / 100)), axis=-1)


def train(rows, l2=1e-3, iters=4000, lr=0.5):
    X = features([r["found"] for r in rows], [r["mass"] for r in rows], [r["strength"] for r in rows],
                 [r["kF"] for r in rows], [r["n_carry"] for r in rows])
    y = np.array([not r["crushed"] and not r["dropped"] for r in rows], float)
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
    Xs = np.concatenate([X[:, :1], (X[:, 1:] - mu) / sd], axis=1)
    w = np.zeros(X.shape[1])
    for _ in range(iters):                       # 勾配降下（データは数百行なので十分）
        p = 1 / (1 + np.exp(-Xs @ w))
        w -= lr * (Xs.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]])
    p = 1 / (1 + np.exp(-Xs @ w))
    return {"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "n": len(y), "ok_rate": float(y.mean()),
            "train_acc": float(((p > 0.5) == (y > 0.5)).mean())}


def prob(W, found, m, strength, kF, n_carry):
    X = features(found, m, strength, kF, n_carry)
    Xs = np.concatenate([X[..., :1], (X[..., 1:] - np.array(W["mu"])) / np.array(W["sd"])], axis=-1)
    return 1 / (1 + np.exp(-Xs @ np.array(W["w"])))


def choose(W, found, m, strength, tau_max=2.94):
    """(握る力 τ, 運ぶ step 数の倍率 slow, 予測の成功率)。slow は弧 90° を 150 step で運ぶのを 1 とした倍率。届かなければ None"""
    best = None
    for n in NC:
        for kF in KF:
            tau = found * kF
            if tau > tau_max or tau * SQ_PER_TAU > SAFE * strength:
                continue
            p = float(prob(W, found, m, strength, kF, n))
            if p >= P_OK:
                best = (tau, n / 150, p)
                break
        if best:
            return best
    return None


if __name__ == "__main__" and sys.argv[1:2] == ["train"]:
    rows = [json.loads(line) for f in sys.argv[2:] for line in open(f) if line.strip()]
    W = train(rows)
    W_PATH.write_text(json.dumps(W, indent=1))
    print(f"{len(rows)} 行（成功 {W['ok_rate']:.0%}）で学習、学習データでの正解率 {W['train_acc']:.0%} → {W_PATH}")
