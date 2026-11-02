# 見た目で分からない箱を、触って確かめてから受け渡す 2 台の SO-101

松尾研 Physical AI 応用編1 最終課題（物理シミュレータ Genesis の中で動かした作品）。

- 動画（YouTube）：https://youtu.be/zjjU4SU3UOU
- 説明資料：`report/説明資料_PAI最終課題.md`（PDF は `scripts/make_pdf.sh` で作る）

## 概要
重さ・摩擦・強度が箱ごとにランダムで、見た目からは分からない 3 cm 角の小箱を、2 種類の SO-101 で受け渡す。

- **1 台目（確かめる）**：ゆるい力から少しずつ強めて試し掴みし、力加減・重さ・強度を自分の関節の角度とトルクから測る。情報を付けて台座に置く
- **2 台目（運ぶ）**：受け取った情報で、試さずに 1 回で持ち、大箱へ運ぶ
- **カメラ**：真上の深度カメラの画像処理で箱の位置と向きを測る。同じ画像を Gemini Robotics-ER 2 に見せ、「壊れにくい」と見た箱だけ試し掴みを強めから始める

力覚センサーは使わない。シミュレータの中の箱の本当の位置・重さ・強度は、答え合わせにだけ使う。

## 結果
同じ 6 箱（乱数の種 seed 278）を、3 つの構成で 1 回ずつ処理した。

| 1 台目 | カメラ | 時間 | 試した回数 | 総稼働時間（台数 × 時間） |
|---|---|---|---|---|
| 3 台 | なし | 111.6 秒 | 13 回 | 446 台・秒 |
| 2 台 | なし | 107.0 秒 | 11 回 | 321 台・秒 |
| 2 台 | あり | 105.1 秒 | 8 回 | 315 台・秒 |

- 3 つの構成とも、6 箱すべてを潰さず・落とさずに大箱へ入れた
- 1 台目を 3 台から 2 台に減らすと、総稼働時間は 446 → 321 台・秒。カメラ（Gemini）を加えると 315 台・秒（合わせて 29% 減）

## 動かし方（Docker、Apple シリコンの Mac で確認）

SO-101 のモデル（Apache-2.0）は含めていないので、同じ版を取ってくる。

```
git clone https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100
git -C third_party/SO-ARM100 checkout a758567
docker compose build sim
```

3 つの構成（1 回 2〜3 分）。最後に箱ごとの表とかかった時間が出る。

```
docker compose run --rm sim python handover/soft_line.py --first 3 --per 2 --seed 278
docker compose run --rm sim python handover/soft_line.py --first 2 --total 6 --seed 278
docker compose run --rm sim python handover/soft_line.py --first 2 --total 6 --seed 278 --camera
```

`--camera` は保存済みの Gemini の答え（`handover/data/top_first2_n6_seed278.gemini.txt`）を使うので、API キーは要らない。保存の無い箱の組では、リポジトリ直下の `.env` に `GEMINI_API_KEY=...` が要る。録画は `--record outputs/video2/p7.mp4 --cam over`（長い録画は `-e NO_HAND=1`）。

## ファイル
| 場所 | 中身 |
|---|---|
| `handover/soft_line.py` | 本体。1 台目・2 台目の手順、試し掴み、重さと強度の測定、台座の排他 |
| `handover/so101.py` | SO-101 を手先の位置で動かす部品（逆運動学、S 字の動き） |
| `handover/softbox.py` | 潰れる箱（ばね付きの板。押す力が強度を超えると沈みきる） |
| `handover/vision.py` | 真上のカメラで箱の位置と向きを出す、手首カメラの画像の比較 |
| `handover/soft_demo.py` | 課題を見せる 3 場面（潰す・落とす・滑る） |
| `sim/vlm_material.py` | Gemini の呼び出し |
| `handover/data/` | 真上のカメラの画像と、保存した Gemini の答え |
| `handover/spikes/` | 試作（手首カメラの位置、校正、Gemini への問い方など） |
| `video/` | 提出動画のスライドと組み立て |
| `report/` | 説明資料の原稿と図 |

**試したが作品には採用しなかったもの**（説明資料 §7.2）
- 運び方の学習：`handover/carry_policy.py`、`handover/spikes/learn_carry.py`、`soft_line.py --policy`。本番の場面では 2 台目が運ぶ途中で落とした
- SmolVLA（微調整なし）：`handover/spikes/smolvla_server.py`（Mac 本体で動かす推論役）、`handover/spikes/smolvla_sim.py`。腕は箱に近づかなかった

## 限界
- 1 組の箱を 1 回ずつ流した結果。別の箱の組では、運ぶ途中で落とす箱や、人の確認に回す箱が出る
- Gemini の答えは同じ画像でも毎回変わり、重さの予想は大きく外れる。重さと強度は触って測る
- シミュレータの中だけで確かめた。実機の SO-101 では試していない

## 使ったもの・ライセンス
| 区分 | 名前（版） | ライセンス |
|---|---|---|
| 物理シミュレータ | Genesis 1.4.3 | Apache-2.0 |
| ロボットモデル | SO-101（TheRobotStudio/SO-ARM100、a758567） | Apache-2.0 |
| VLM | Gemini Robotics-ER 2（google-genai 2.29） | API |
| VLA（試しのみ） | SmolVLA `lerobot/smolvla_base`（LeRobot 0.6.1） | Apache-2.0 |
| 数値計算・画像処理 | PyTorch／NumPy／SciPy | BSD-3 |
| 動画・資料 | FFmpeg／python-pptx／LibreOffice | LGPL／MIT／MPL-2.0 |

このリポジトリのコード：Apache License 2.0（`LICENSE`）。コード・スライド・説明資料の作成に、AI のコーディング支援（Claude Code）を使った。
