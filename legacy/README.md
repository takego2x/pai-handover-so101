# legacy/ — 10/11 の旧構成（戦略 A/B/C）の部品

2026-10-12〜13 の組み替えで、作品の構成が変わった（試し掴みで測り、学習した方策で運ぶ。教材 `lesson/`）。
ここには、旧構成のための部品を退かしてある。**このままでは動かない**（`sim/transport.py` を組み替えで変えた後の版を退かしたため。
旧構成の dora の配線と CI の評価は、組み替えで壊れていた）。

| 退かしたもの | 旧構成での役割 |
|---|---|
| `dora/` | 旧構成の dora の配線（カメラ → VLM → 計画 → 実行 → 記録）。今の dora 版は `lesson/nodes_dora/` |
| `sim/transport.py` `evaluate.py` `sensitivity.py` `line.py` | 戦略 A/B/C の運び方・評価・感度・2 台のライン |
| `verify/` の 5 本 | 旧構成の検証の再現（v15 は旧構成の dora の待ち行列） |
| `scripts/figures.py` `check_metrics.py` | 旧構成の図と、CI の判定 |
| `tests/test_core.py` | 旧構成のテスト。今のテストは `tests/test_lesson.py` |
| `spikes/line3x1.py` `line_handover_probe.py` | 組み替えの途中の試作（中身は `lesson/line.py` へ） |
| `video/storyboard_lv0.json` `report/report.html` `README_lv0.md` | 旧構成の動画の台本・レポート・説明 |

旧構成を動かして見たいときは、組み替え前の main ブランチ（cfd81cb）を別の場所に取り出す：
```bash
git worktree add ../pai-final-mu-lv0 main
```
