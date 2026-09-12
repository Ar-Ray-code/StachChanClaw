# StachChanClaw

かわいさ・手軽さ・身体性を兼ね備えたAIエージェントの構築

## 概要

WebcamChan の 320x240 UVC カメラを使って人物を検出・追従する Python ツールキットです。

### 検出パイプライン

複数の検出器を組み合わせて人物を検出します。

1. **MediaPipe Pose** - 骨格検出による人物検出・追従
2. **OpenCV 顔検出** - 顔が見えるときの中心点補正
3. **OpenCV Upper-body 検出** - 上半身検出による補完

### 探索アルゴリズム

粗密探索で効率的に人物を発見します。

| 段階 | ステップ | 動作 |
|------|----------|------|
| 粗探索 | 30° | パン軸を `90° → 0° → 90° → 180° → 90°` で走査 |
| 中探索 | 15° | 候補発見位置を中心に詳細走査 |
| 密探索 | 5° | 最終的な位置決め |

チルト軸は 110° 固定で探索し、候補が見つからなければ原点 (90°/90°) に戻ります。

### 追従制御

以下の 3 条件を同時に満たすようサーボを制御します。

- 枠の X 中心 → 画面中央
- 枠の Y 重心 → 画面中央
- 枠の上端 → 画面上端（top margin）

## 前提条件

- WebcamChan ファームウェア書き込み済み（`/dev/video0` 等で認識）
- `v4l2-ctl` コマンド
- Linux + `uv` パッケージマネージャ

動作確認:

```bash
ffplay -f v4l2 -input_format mjpeg -video_size 320x240 -framerate 30 /dev/video0
v4l2-ctl -d /dev/video0 --all
```

## 実行

初回起動時に MediaPipe Pose モデル (`pose_landmarker_full.task`) を `models/` へ自動ダウンロードします。

### face_search.py

```bash
# GUI 表示あり
uv run python face_search.py --camera-device /dev/video0 --display

# ヘッドレス実行
uv run python face_search.py --camera-device /dev/video0

# デバッグ録画
uv run python face_search.py --camera-device /dev/video0 --display --record captures/run.mp4
```

探索中の評価画像は `captures/search/sweep_XXXX/` に保存されます。各画像には人物・顔・pose・upper-body の bounding box とターゲットマーカーが描画されます。

### presence_observer.py

OpenClaw agent 向けに 1 回の探索結果をレポート出力します。

```bash
uv run python presence_observer.py \
  --camera-device /dev/video0 \
  --output-dir ~/.openclaw/workspace/camera_presence
```

#### 出力ファイル

| ファイル | 内容 |
|----------|------|
| `latest_report.json` | 人の有無・時間帯・姿勢ヒント・物体関連付け |
| `latest_report.md` | agent 向け短縮要約 |
| `latest_frame.jpg` | 最終スナップショット（生画像） |
| `latest_annotated.jpg` | bounding box 重畳画像 |
| `runs/<timestamp>/` | 実行ごとの完全保存 |
| `runs/<timestamp>/sweeps/detection_report.json` | 探索中の全画像検出結果 |
| `history.jsonl` | 時系列観測履歴 |

#### detection_report.json

探索中に撮影した全画像の検出結果を記録します。bounding box 描画済み画像とは別に、構造化された検出データを参照できます。

```json
{
  "generated_at": "2026-04-18T17:05:55",
  "sweep_index": 1,
  "total_images": 35,
  "images_saved": true,
  "best_objective": 0.567699,
  "best_target_kind": "human_box",
  "images": [
    {
      "image_path": "/path/to/sweep_0001/01_coarse_pan090p4_tilt110p0_pose_box.jpg",
      "pose_index": 1,
      "stage": "coarse",
      "pan_deg": 90.353,
      "tilt_deg": 110.0,
      "detection": {
        "humans_count": 1,
        "faces_count": 0,
        "poses_count": 1,
        "target_kind": "pose_box",
        "objective": 0.476205
      },
      "detections": { "humans": [...], "faces": [...], "poses": [...] }
    }
  ]
}
```

#### --delete-cache オプション

探索中のスクリーンショット画像を保存せず、検出結果のみ `detection_report.json` に記録します。`detection_report.json` の `images_saved` は `false` になり、`image_path` にはパス情報のみ記録されます（実ファイルは作成されません）。

#### 姿勢推定との連携

`latest_report.json` の `scene.people[].associated_objects` には、人物枠と重なった物体（椅子・ソファなど）が含まれます。これにより「椅子が近い → 座っている可能性」といった判断を agent 側で行えます

> 検出モデルがサポートしない場合、空配列になります。

## 主要パラメータ

### 追従制御

| パラメータ | 既定値 | 説明 |
|------------|--------|------|
| `--deadzone` | `24,18` | 停止判定の許容範囲 (px) |
| `--top-margin` | `4` | 枠上端の目標位置 (px) |
| `--top-deadzone` | `8` | 枠上端の許容幅 (px) |
| `--pan-gain` | `72` | パン軸の追従ゲイン |
| `--tilt-gain` | `60` | チルト軸の追従ゲイン |
| `--lost-timeout` | `0.9` | 見失い後の探索開始待ち (秒) |

### 探索設定

| パラメータ | 既定値 | 説明 |
|------------|--------|------|
| `--search-pan-route` | `90,0,90,180,90` | パン軸の探索ルート (度) |
| `--search-tilt-degree` | `110` | 探索時のチルト角度 (度) |
| `--search-step-degrees` | `30,15,5` | 段階的な探索刻み (度) |
| `--search-interval` | `0.60` | 撮影前の待ち時間 (秒) |
| `--search-flush-reads` | `6` | 撮影前に捨てるフレーム数 |

### 検出設定

| パラメータ | 既定値 | 説明 |
|------------|--------|------|
| `--min-votes-without-face` | `2` | 顔なし時の必要検出票数 |
| `--pose-min-detection-confidence` | `0.45` | pose 検出の信頼度閾値 |
| `--pose-min-visibility` | `0.45` | landmark の visibility 閾値 |

### 出力制御

| パラメータ | 説明 |
|------------|------|
| `--search-captures-dir` | 探索評価画像の保存先 |
| `--no-save-search-captures` | 探索評価画像を保存しない |
| `--delete-cache` | スクリーンショットを保存せず JSON のみ出力 |
| `--dry-run` | モーターを動かさずカメラ処理のみ |
