"""S6: μ の文献値の表を読み、材質の候補から「安全側の μ（下限）」を返す部品。"""
import csv
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "data" / "friction_literature.csv"


def load(path=CSV):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["verified"] != "yes" or not r["mu_s_min"]:
            continue  # 原文で確かめていない値・公開値のない組み合わせは使わない
        rows.append({**r, "mu_s_min": float(r["mu_s_min"]), "mu_s_max": float(r["mu_s_max"])})
    return rows


def mu_range(material, counterpart="steel", rows=None):
    """材質 1 つについて、文献値の (下限, 上限, 出典の一覧) を返す。見つからなければ None。"""
    rows = rows if rows is not None else load()
    hits = [r for r in rows
            if r["material"].lower().startswith(material.lower())
            and counterpart.lower() in r["counterpart"].lower()]
    if not hits:
        return None
    return (min(r["mu_s_min"] for r in hits), max(r["mu_s_max"] for r in hits),
            sorted({r["source_id"] for r in hits}))


def safe_mu(candidates, counterpart="steel", rows=None):
    """VLM が挙げた材質の候補すべての中で、いちばん小さい下限を返す（見分けられないときの安全側）。"""
    ranges = [mu_range(c, counterpart, rows) for c in candidates]
    known = [r for r in ranges if r is not None]
    if len(known) < len(ranges):
        return None  # 値のない候補が混じっていたら決めない（呼び出し側で最も遅い戦略に倒す）
    return min(r[0] for r in known)


if __name__ == "__main__":
    for m in ["POM", "ABS", "aluminium", "steel", "NBR"]:
        print(m, mu_range(m))
    print("safe_mu(['aluminium', 'steel']) =", safe_mu(["aluminium", "steel"]))
    print("safe_mu(['POM', 'ABS'])        =", safe_mu(["POM", "ABS"]))
    print("safe_mu(['POM', 'rubber'])     =", safe_mu(["POM", "rubber"]))
