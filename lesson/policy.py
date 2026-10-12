"""L7: 運ぶ方策を学習する（AI 2）。L8 の第二層も、ここの ArcScene と方策を使う。

場面：腕（根元が原点）が、床の部品（4 cm 角）を、根元の周りに弧を描いて運ぶ（-60° → 110°、170°）。
方策：入力 = 重さ m と μ。学習と評価（このファイル）では、正解の μ に L3 で測った誤差の幅（小さめに 3〜15%）を
      まねた乱数を掛けて作った値を入れる（重さは正解）。本当に試し掴みで測った値を入れるのは L8 の第二層
      出力 = 握る力の倍率 kF（§1.3 の式の力 × kF、上限 60 N）、加速度 a、指の向き yaw（根元から見た方向からのずれ）
  式の方策・半径向き : kF = 1、a = 9、yaw = 0（L6 で部品が振り子のように回って落ちた持ち方）
  式の方策・接線向き : kF = 1、a = 9、yaw = 90°（L6 で人が見つけた対策）
  学習した方策       : [log kF, a, yaw] = W @ [1, log μ, log m]（係数 9 個）。CEM で学ぶ
採点：運べたら 1 − 0.3·kF − 0.1·(時間 / 2 秒)、落としたら −3（--mistake few_parts のときは −1）。

使い方（作業場所の直下で）:
  python lesson/policy.py train                     # 学習（約 13 分）。outputs/lesson/policy_W.json に書く
  python lesson/policy.py eval --w data/policy_W.json --test-seed 9000
                                                    # 学習に使わない 30 部品で、3 つの方策を比べる（まっさらな場面）
わざと失敗:
  train --mistake few_parts   : 1 世代で試す部品を 16 → 8 個にし、落としたときの減点を軽く（−1）する
  train --mistake same_scene  : 学習した場面のまま、続けて評価する（まっさらな場面にしない）
"""
import argparse
import json
import math
import time

import numpy as np
import torch

import common as c

TH0, TH1, R = math.radians(-60.0), math.radians(110.0), 0.55
Z_GRIP, Z_UP = c.SIDE / 2 + 0.115, 0.35


def polar(r, th):
    return np.array([r * math.cos(th), r * math.sin(th)])


class ArcScene:
    """弧を描いて運ぶ場面。n 個の環境を並列に回す。episode() は何度でも呼べる（部品を置き直す）。"""

    def __init__(self, n):
        self.n = n
        self.scene = c.make_scene()
        self.scene.add_entity(c.gs.morphs.Plane())
        self.part = c.add_part(self.scene, polar(R, TH0))
        self.arm = c.add_franka(self.scene)
        self.scene.build(n_envs=n)
        c.setup_franka(self.arm, n)

    def _arc(self, t, z):  # t: (n,) 0〜1。半径一定で -60° → 110°
        ang = TH0 + (TH1 - TH0) * t
        return np.column_stack([R * np.cos(ang), R * np.sin(ang), np.broadcast_to(z, ang.shape)])

    def _cmd(self, pos, yaw_off, f=None):
        # IK の出発点は「根元を目標の方向に向けた構えの姿勢」（L6 のわざと失敗 2 の対策）
        az = np.arctan2(pos[:, 1], pos[:, 0])
        seed = np.tile(c.HOME, (self.n, 1))
        seed[:, 0] = az
        seed[:, 6] = c.HOME[6] - yaw_off
        c.command(self.arm, pos, self.n, force=f, yaw=az + yaw_off,
                  seed=torch.tensor(seed, dtype=c.gs.tc_float, device=c.gs.device))

    def episode(self, mats, true_mu, mu_hat, kF, a, yaw_off):
        """全環境で 1 回運ぶ。成功（運べたか）、使った力 [N]、時間 [s] を返す。"""
        n, scene = self.n, self.scene
        m = c.set_parts(self.part, mats, true_mu)  # シミュレータの中の部品（本当の値）
        self.part.set_pos(np.tile([*polar(R, TH0), c.SIDE / 2], (n, 1)))
        self.part.set_quat(np.tile([math.cos(TH0 / 2), 0, 0, math.sin(TH0 / 2)], (n, 1)))
        self.part.zero_all_dofs_velocity()
        self.arm.set_dofs_position(np.tile(c.HOME, (n, 1)))
        self.arm.zero_all_dofs_velocity()
        F = np.minimum(c.F_MAX, c.carry_force(m, mu_hat) * kF)  # 計画にはデータ（mu_hat と測った重さ = m）だけ
        up0, down = self._arc(np.zeros(n), 0.25), self._arc(np.zeros(n), Z_GRIP)
        for _ in range(150):
            self._cmd(up0, yaw_off)
            scene.step()
        for k in range(100):
            self._cmd(up0 + (down - up0) * (k + 1) / 100, yaw_off)
            scene.step()
        for _ in range(50):
            self._cmd(down, yaw_off)
            scene.step()
        f0 = np.minimum(F, 5.0)
        for k in range(50):  # そっと閉じてから目標の力へ
            self._cmd(down, yaw_off, f0 + (F - f0) * (k + 1) / 50)
            scene.step()
        for _ in range(20):
            self._cmd(down, yaw_off, F)
            scene.step()
        lifts = [c.trapezoid(Z_UP - Z_GRIP, ai) for ai in a]
        L = R * (TH1 - TH0)
        moves = [c.trapezoid(L, ai) for ai in a]
        for k in range(int(math.ceil(max(t for _, t in lifts) / c.DT)) + 20):
            z = Z_GRIP + np.array([p((k + 1) * c.DT) for p, _ in lifts])
            self._cmd(self._arc(np.zeros(n), z), yaw_off, F)
            scene.step()
        for k in range(int(math.ceil(max(t for _, t in moves) / c.DT)) + 50):
            s = np.array([p((k + 1) * c.DT) / L for p, _ in moves])
            self._cmd(self._arc(s, Z_UP), yaw_off, F)
            scene.step()
        p = self.part.get_pos().cpu().numpy()
        ok = (p[:, 2] > 0.15) & (np.linalg.norm(p[:, :2] - polar(R, TH1), axis=1) < 0.05)
        t = np.array([tl for _, tl in lifts]) + np.array([tm for _, tm in moves])
        return ok, F, t


def features(mu_hat, m):
    return np.column_stack([np.ones_like(mu_hat), np.log(mu_hat), np.log(m)])


YAW_RANGE = math.pi / 2  # 指の向きの出力を切る範囲（±90°）。--yaw-range で変える


def learned(W, mu_hat, m, yaw_range=None):
    """学習した方策：データ → 握る力の倍率、加速度、指の向き（±yaw_range で切る）。"""
    yr = YAW_RANGE if yaw_range is None else yaw_range
    out = features(mu_hat, m) @ np.asarray(W).T
    return (np.clip(np.exp(out[:, 0]), 0.5, 3.0), np.clip(out[:, 1], 0.5, c.A_CAP), np.clip(out[:, 2], -yr, yr))


def formula(n, tangent):
    """式の方策（§1.3）。tangent=True なら指を接線の向きに閉じる。"""
    return np.ones(n), np.full(n, c.A_CAP), np.full(n, math.pi / 2 if tangent else 0.0)


def sample(n, seed0, stream):
    """部品と、方策に入れる値。μ は正解に 0.85〜0.97 倍の乱数を掛けて作る（L3 で測った誤差の幅をまねた値で、
    測った値そのものではない）。重さは正解。"""
    mats, mu = c.sample_parts(n, seed0, stream=stream)
    rng = np.random.default_rng(seed0 + 7 * stream)
    mu_hat = mu * rng.uniform(0.85, 0.97, n)
    return mats, mu, mu_hat, np.array([c.mass_of(x) for x in mats])


def score(ok, kF, t, fail):
    return np.where(ok, 1 - 0.3 * kF - 0.1 * t / 2.0, -fail)


def evaluate(sc, W, T, test_seed):
    """学習に使わなかった T 個の部品で、式・半径／式・接線／学習 の 3 つを同時に比べる。"""
    mats, mu, mu_hat, m = sample(T, test_seed, stream=99)
    idx, grp = np.arange(sc.n) % T, np.arange(sc.n) // T  # 環境 i → 部品 i % T、方策 i // T
    kF, a, yaw = formula(sc.n, tangent=False)
    kF[grp == 1], a[grp == 1], yaw[grp == 1] = formula(int((grp == 1).sum()), tangent=True)
    sel = grp == 2
    kF[sel], a[sel], yaw[sel] = learned(W, mu_hat[idx[sel]], m[idx[sel]])
    ok, F, t = sc.episode([mats[j] for j in idx], mu[idx], mu_hat[idx], kF, a, yaw)
    res = {}
    print(f"{'policy':16s} {'success':>8s} {'F median':>9s} {'kF':>5s} {'a':>5s} {'yaw':>6s} {'time':>6s}  failed parts")
    for g, name in enumerate(["formula_radial", "formula_tangent", "learned"]):
        s = grp == g
        res[name] = {"success": int(ok[s].sum()), "n": T, "F_median_N": round(float(np.median(F[s])), 2),
                     "kF_mean": round(float(kF[s].mean()), 2), "a_mean": round(float(a[s].mean()), 2),
                     "yaw_deg_mean": round(float(np.degrees(yaw[s]).mean()), 1), "t_mean_s": round(float(t[s].mean()), 2),
                     "failed": [int(i) for i in np.where(~ok[s])[0]]}
        r = res[name]
        print(f"{name:16s} {r['success']:>4d}/{T:<3d} {r['F_median_N']:>7.2f}N {r['kF_mean']:>5.2f} {r['a_mean']:>5.2f} "
              f"{r['yaw_deg_mean']:>5.0f}° {r['t_mean_s']:>5.2f}s  {r['failed']}")
    return res


def train(args):
    few = args.mistake == "few_parts"
    P, pop, fail = (8 if few else 16), 16, (1.0 if few else 3.0)
    n = pop * P if args.mistake != "same_scene" else max(pop * P, 3 * 30)
    sc = ArcScene(n)
    rng = np.random.default_rng(0)
    mean = np.zeros((3, 3))
    mean[1, 0] = c.A_CAP  # 出発点は「式の方策・半径向き」（kF = 1、a = 9、yaw = 0）
    std = np.array([[0.3, 0.2, 0.2], [2.0, 1.0, 1.0], [0.8, 0.3, 0.3]])
    t0 = time.time()
    print(f"{'gen':>3s} {'best score':>10s} {'mean score':>10s} {'kF':>5s} {'a':>5s} {'yaw':>6s} {'wall':>6s}")
    for g in range(args.gens):
        mats, mu, mu_hat, m = sample(P, 5000 + 100 * g, stream=g)
        Ws = [mean] + [mean + std * rng.standard_normal((3, 3)) for _ in range(pop - 1)]  # 方策を 16 通り作る
        idx, grp = np.arange(n) % P, np.arange(n) // P
        kF, a, yaw = formula(n, tangent=False)
        for i, W in enumerate(Ws):
            s = grp == i
            kF[s], a[s], yaw[s] = learned(W, mu_hat[idx[s]], m[idx[s]])
        ok, F, t = sc.episode([mats[j] for j in idx], mu[idx], mu_hat[idx], kF, a, yaw)
        sco = score(ok, kF, t, fail)[: pop * P].reshape(pop, P).mean(axis=1)  # 方策ごとの平均点
        elite = np.argsort(sco)[-4:]  # 上位 4 つに寄せる（CEM）
        mean = np.mean([Ws[i] for i in elite], axis=0)
        std = np.maximum(np.std([Ws[i] for i in elite], axis=0), 0.02)
        b = grp == elite[-1]
        print(f"{g:3d} {sco[elite[-1]]:10.3f} {sco.mean():10.3f} {kF[b].mean():5.2f} {a[b].mean():5.2f} "
              f"{np.degrees(yaw[b]).mean():5.0f}° {time.time() - t0:5.0f}s", flush=True)
    json.dump({"W": mean.tolist(), "gens": args.gens, "parts_per_gen": P, "fail_penalty": fail,
               "yaw_range_deg": math.degrees(YAW_RANGE)}, open(args.out, "w"), indent=1)
    print("wrote", args.out)
    if args.mistake == "same_scene":
        print("---- evaluate in the SAME scene (mistake)")
        evaluate(sc, mean, 30, args.test_seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--w", default=str(c.ROOT / "data" / "policy_W.json"), help="eval で使う学習済みの方策")
    ap.add_argument("--test-seed", type=int, default=9000)
    ap.add_argument("--mistake", choices=["few_parts", "same_scene"])
    ap.add_argument("--out", default=str(c.ROOT / "outputs" / "lesson" / "policy_W.json"))
    ap.add_argument("--json", help="eval の結果を書く")
    ap.add_argument("--yaw-range", type=float, default=90.0,
                    help="指の向きの出力を ±何度で切るか。90 だと、答え（接線 ±90°）が範囲の端にある")
    args = ap.parse_args()
    c.quiet()
    YAW_RANGE = math.radians(args.yaw_range)
    c.ROOT.joinpath("outputs", "lesson").mkdir(parents=True, exist_ok=True)
    if args.cmd == "train":
        train(args)
    else:
        sc = ArcScene(90)  # まっさらな場面（学習の履歴を持ち込まない）
        wj = json.load(open(args.w))
        YAW_RANGE = math.radians(wj.get("yaw_range_deg", 90.0))  # 学習したときと同じ範囲で切る
        res = evaluate(sc, wj["W"], 30, args.test_seed)
        if args.json:
            json.dump(res, open(args.json, "w"), indent=1)
