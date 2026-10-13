# 見て・触って測り、学習した方策で運ぶ ― 材質の分からない部品を、小さい力で傷めずに

松尾研 Physical AI 応用編1 最終課題。動画：（YouTube の URL）

## 概要
材質の分からない部品が次々に来るラインを、シミュレータ（Genesis）の中に作った。
- **第一層（アーム 3 台）**：部品を **見て**（AI 1：VLM が材質の候補を挙げ、試す範囲を狭める）、**触って**（握る力を変えて何度か持ち上げを試し、落としてもよい）、重さ・摩擦係数 μ・保持に要る力を測る
- **第二層（アーム 1 台）**：受け取った重さと μ を、**学習した方策**（AI 2：シミュレータの中で CEM により学習）が「握る力・加速度・指の向き」に変え、弧を描いて一発で運ぶ
- 第一層と第二層は、ノードに分けてメッセージでつなぐ（dora-rs 版と ROS 2 Jazzy 版の両方）

| 結果 | 値 |
|---|---|
| 握る力（データあり／データなし） | 中央値 3.5N ／ 全部品 60N（60 部品） |
| 学習した方策（学習に使わない 90 部品） | 90/90 運ぶ（式の方策 90/90）。握る力は式より約 25% 小さい。代わりに時間が 12% 長い |
| 学習した方策が見つけた持ち方 | 指を接線の向き（85°）に閉じる。人が原因を切り分けて見つけた対策と同じ。範囲の端の助けなし |
| 見て絞る（VLM）の効果 | 試し掴み 7.0 → 6.5 回（材質が分かっても μ のばらつきは大きい） |

## 再現手順（Docker、Apple シリコンの Mac で確認）
```bash
docker compose build
docker compose run --rm sim python lesson/hold.py              # 保持の条件（理論と 24/24）
docker compose run --rm sim python lesson/trials.py            # 試し掴みで重さ・μ
docker compose run --rm sim python lesson/policy.py eval       # 学習した方策と式の方策の比較
docker compose run --rm sim python lesson/report.py           # 図と表（約 7 分）
docker compose run --rm sim bash -c "cd lesson/nodes_dora && dora run dataflow.yml"           # dora 版
docker compose --profile ros2 build ros2 && docker compose --profile ros2 run --rm ros2 bash lesson/nodes_ros2/run.sh 5   # ROS 2 版
docker compose run --rm sim python -m pytest -q tests          # テスト（CI と同じ）
```
L0〜L9 の手順と、作者の失敗の記録・わざと失敗の再現は、教材にまとめてある。

## 構成
| 場所 | 中身 |
|---|---|
| `lesson/` | 教材の本体（L1〜L9）。`policy.py` は運ぶ方策の学習、`nodes_dora/` と `nodes_ros2/` はノードのつなぎ方 |
| `data/` | μ の文献値の表と出典、Gemini の材質の候補（保存済み）、学習済みの方策の係数 |
| `sim/` | 手順書の S1〜S7・S11 の部品（箱を落とす、IK、μ の掃引、トレイの画像、Gemini の呼び出し、くちばし） |
| `spikes/` `verify/` | 試作と検証の再現 |
| `legacy/` | 10/11 の旧構成（戦略 A/B/C）の部品。動かない（`legacy/README.md`） |

## 限界
シミュレータで示せるのは握る力の N 数まで（力と傷の関係は示していない）。μ は押す力で変わらない。学習と評価で方策に入れる μ は、正解に測定の誤差の幅をまねた乱数を掛けた値。VLM の候補は、部品の正解の材質で過去の応答を割り当てている。

## データと出典
μ の文献値は `data/friction_literature.csv`、出典は `data/SOURCES.md`。原文で値を確かめたもの（`verified=yes`）だけを使う。

## ライセンス
- このリポジトリのコード：Apache License 2.0（`LICENSE`）
- Microduck のモデル：[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)。モデル記述（`.xml`）は Apache-2.0、3D 形状ファイル（`.stl`）は CC BY-NC-SA 4.0（© Pollen Robotics）。形状ファイルはこのリポジトリに含めず、`scripts/fetch_microduck_assets.sh` で固定版から取得する。くちばしの試験台では、頭部の部品のうち下あごを切り出して関節を足している（改変）
- Franka Emika Panda のモデルは Genesis に同梱のものを使用
