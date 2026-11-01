# 見た目で分からない箱を、触って確かめてから受け渡す 2 台の SO-101

松尾研 Physical AI 応用編1 最終課題。

- 動画（YouTube）：（URL）

## 概要
物理シミュレータ Genesis の中で、ロボットアーム SO-101 を役割の違う 2 種類に分けた。

- **1 台目（触って確かめる）**：ゆるい力から少しずつ強めて箱を試し掴みし、持ち上がる力（力加減）・重さ・強度を、自分の関節の角度とトルクから測る。測った情報を付けて、箱を台座に置く
- **2 台目（情報で運ぶ）**：受け取った情報で、試さずに 1 回で箱を持ち、大箱へ運ぶ
- **カメラ**：真上の固定カメラの画像を Gemini Robotics-ER 2 に見せ、「壊れにくい」と見た箱だけ、試し掴みの刻みを大きくする

箱は 3 cm 角で、重さ（30〜250 g）・摩擦（0.6〜1.0）・強度（7.5〜60 N）が箱ごとにランダムで、見た目からは分からない。力覚センサーは使わず、判断に使うのはカメラ 2 つ（真上・手首）の画像と、ロボット自身の関節の値だけ。シミュレータの箱の座標は、答え合わせにだけ使う。

## 結果
同じ 6 箱（seed 278）を 3 つの構成で処理した。

| 1 台目 | カメラ | 時間 | 試した回数 | 総稼働時間（台数 × 時間） |
|---|---|---|---|---|
| 3 台 | なし | 111.6 秒 | 13 回 | 446 台・秒 |
| 2 台 | なし | 107.0 秒 | 11 回 | 321 台・秒 |
| 2 台 | あり | 105.1 秒 | 8 回 | 315 台・秒 |

- 3 つの構成とも、6 箱すべてを潰さず・落とさずに大箱へ入れた
- 1 台目を 3 台から 2 台に減らしてカメラを加えると、総稼働時間は 29% 少ない
- 1 組の箱を 1 回ずつ流した結果。3 台の構成には、台座の取り合いを待つ時間が含まれる

## 再現手順（Docker、Apple シリコンの Mac で確認）

SO-101 のモデル（Apache-2.0）はこのリポジトリに含めていない。同じ版を取ってくる。

```
git clone https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100
```

```
git -C third_party/SO-ARM100 checkout a758567
```

シミュレータの環境（Genesis 1.4.3、CPU）を作る。

```
docker compose build sim
```

1 台目 3 台・カメラなし（1 台あたり 2 箱、計 6 箱）。1 回 2〜3 分。

```
docker compose run --rm sim python handover/soft_line.py --first 3 --per 2 --seed 278
```

1 台目 2 台・カメラなし（同じ 6 箱）。

```
docker compose run --rm sim python handover/soft_line.py --first 2 --total 6 --seed 278
```

1 台目 2 台・カメラあり。Gemini の答えは `handover/data/top_first2_n6_seed278.gemini.txt` に保存してあり、API キーなしでこれを使う。保存の無い箱の組で `--camera` を付けると、リポジトリ直下の `.env` に `GEMINI_API_KEY=...` が要る。

```
docker compose run --rm sim python handover/soft_line.py --first 2 --total 6 --seed 278 --camera
```

最後に箱ごとの表（本当の重さと測った重さ、本当の強度と推定、力加減、試した回数、結果）と、かかった時間（`steps ... (105.1 s)`）が出る。

録画するときは `--record outputs/video2/p7.mp4 --cam over` を足す。長い録画では `-e NO_HAND=1` を付けて手首カメラの映像を保存しない（描画が止まることがあった）。試し掴みの中身を見るときは `-e SOFT_DEBUG=1` を付ける。

## 構成
| 場所 | 中身 |
|---|---|
| `handover/soft_line.py` | 本体。1 台目・2 台目の手順、試し掴み、重さと強度の測定、台座の排他（札） |
| `handover/so101.py` | SO-101 を手先の位置で動かす部品（逆運動学、S 字の動き） |
| `handover/softbox.py` | 潰れる箱（ばね付きの板。押す力が強度を超えると沈みきる） |
| `handover/vision.py` | 真上の固定カメラで箱の位置と向きを出す、手首カメラの画像の比較 |
| `handover/soft_demo.py` | 課題を見せる 3 場面（潰す・落とす・滑る） |
| `handover/data/` | 真上のカメラの画像と、Gemini の答え（保存済み） |
| `handover/spikes/` | 試作（手首カメラの位置、校正、Gemini への問い方など） |
| `video/` | 提出動画のスライドと組み立て（`make_slides_take.py` → `render_pages.sh` → `board_from_runs.py` → `compose_slides.py`） |
| `report/` | 説明資料の原稿 |

次は前のテーマ（部品の摩擦係数 μ を測って運ぶ作品）の物で、この作品では使っていない：`lesson/`、`colab/`、`sim/`（Gemini の呼び出し `sim/vlm_material.py` だけは使う）、`spikes/`、`verify/`、`data/`、`legacy/`、`tests/`、`Dockerfile.ros2`、`handover/line.py`・`line3.py`・`run.py`。ROS 2 と dora-rs の部品はここにあり、この作品では使っていない。

動画の組み立てに使った表紙の映像と写真（`video/assets/`、`outputs/video2/real_returns.png`）は含めていない。

## 限界
- 1 組の箱を 1 回ずつ流した結果。別の箱の組では、運ぶ途中で落とす箱や、人の確認に回す箱が出る
- 箱を 12 個に増やすと、1 台目が詰まり、1 台目を減らす利点は消えた
- Gemini の答えは同じ画像でも毎回変わり、重さの予想は大きく外れる（金属 160〜240 g を 15〜60 g）。重さは触って測る
- シミュレータの中だけで確かめた。実機の SO-101 では試していない

## 使ったもの
| 区分 | 名前（版） | ライセンス |
|---|---|---|
| 物理シミュレータ | Genesis 1.4.3 | Apache-2.0 |
| ロボットモデル | SO-101（TheRobotStudio/SO-ARM100、a758567） | Apache-2.0 |
| VLM | Gemini Robotics-ER 2（google-genai 2.29） | API |
| 数値計算・画像処理 | PyTorch／NumPy／SciPy | BSD-3 |
| 動画・資料 | FFmpeg／python-pptx／LibreOffice | LGPL／MIT／MPL-2.0 |

## ライセンス
このリポジトリのコード：Apache License 2.0（`LICENSE`）。

前のテーマで使った Microduck のモデル（[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)）は、モデル記述（`.xml`）が Apache-2.0、3D 形状ファイル（`.stl`）が CC BY-NC-SA 4.0（© Pollen Robotics）。形状ファイルはこのリポジトリに含めず、`scripts/fetch_microduck_assets.sh` で取得する。

コード・スライド・説明資料の作成に、AI のコーディング支援（Claude Code）を使った。
