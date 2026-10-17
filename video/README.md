# 提出用の動画の作り方

割り振り表 `video/storyboard.json` の順に、場面をつないで 1 本にする。構成の考え方は Obsidian の `PAI応用編1_最終課題_動画構成.md`。

## 1. 撮る（Docker の中。答え合わせ用リポジトリの直下で）

ロボットの場面は、`common.py` の録画用カメラ（環境変数 `PAI_CLIP`）で撮る。コードは変えない。
`PAI_CLIP_ENV` は録る環境の番号、`PAI_CLIP_CAM` は「カメラの位置; 見る点」、`PAI_CLIP_FOV` は画角。

試し掴み（鋼の部品、環境 2）。`parts3.json` に 1 回ごとの力と成否が入る。
```bash
docker compose run --rm -e PAI_CLIP=/work/outputs/video/clip_trials.mp4 -e PAI_CLIP_ENV=2 -e "PAI_CLIP_CAM=0.62,-0.42,0.17;0.55,0.0,0.07" -e PAI_CLIP_FOV=40 sim python lesson/trials.py 3 --out outputs/video/parts3.json
```
同じ部品を 3 通りで運ぶ（環境 0 = データなし 60 N、1 = 式・半径向き、2 = 学習した方策）。環境ごとに 1 回ずつ。
```bash
docker compose run --rm -e PAI_CLIP=/work/outputs/video/carry_0.mp4 -e PAI_CLIP_ENV=0 -e "PAI_CLIP_CAM=1.3,-0.4,1.2;0.05,0.05,0.22" -e PAI_CLIP_FOV=55 sim python video/scene_carry.py --part 3
```
（`carry_1.mp4` と `PAI_CLIP_ENV=1`、`carry_2.mp4` と `PAI_CLIP_ENV=2` でも同じように）

3 台 → 1 台のライン。
```bash
docker compose run --rm -e PAI_CLIP=/work/outputs/video/clip_line.mp4 -e PAI_CLIP_ENV=0 -e "PAI_CLIP_CAM=2.0,-1.7,2.0;0.35,-0.05,0.1" -e PAI_CLIP_FOV=50 sim python lesson/line.py --cells 3 1
```
24 マス（L1）。
```bash
docker compose run --rm sim python lesson/hold.py --record outputs/video/L1_grid.mp4
```
学習の記録（約 28 分。係数は `data/policy_W.json` と同じになる）。
```bash
docker compose run --rm sim python lesson/policy.py train --out outputs/lesson/policy_W_hist.json
```

## 2. 素材を作る（Mac。日本語のフォントと ffmpeg を使う）
```bash
uv run --no-project --python 3.12 --with pillow==11.3.0 --with matplotlib==3.10.7 --with numpy==2.3.4 python video/make_assets.py
```
`outputs/video/` に、図（`diagram.png`）、Gemini の候補（`tray_gemini.png`）、結果（`results.png`）、失敗と考察（`failures.png`）、
試し掴みの映像とグラフ（`scene_trials.mp4`）、運ぶ 3 通り（`scene_carry.mp4`）、学習曲線（`scene_learning.mp4`）ができる。

## 3. つなぐ
```bash
uv run --no-project --python 3.12 --with pillow==11.3.0 python scripts/make_video.py video/storyboard.json outputs/video/PAI_final_draft.mp4
```
最後に長さを測って `OK (1-3 min)` と出る。倍速の場面（`"speed": 2`）は、字幕に「2 倍速」と書く（提出の決まり）。

## 差し替え
- rerun や wandb の画面を使いたいときは、ブラウザの画面を Mac の画面収録（⌘⇧5）で mp4 にし、`storyboard.json` の `clip` を差し替える
- 字幕は `caption`、文字だけの画面は `card` と `sub`
- 氏名・所属・メールを映さない（提出の決まり）
