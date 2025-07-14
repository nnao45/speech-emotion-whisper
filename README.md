# Speech Emotion Recognition with Multi-Modal Analysis

MP4ファイルから音声を抽出し、感情分析・文字起こし・話者分析・音声メトリクス分析を行うコマンドラインAIアプリケーション。

## 主要機能

- 🎬 MP4動画ファイルからの音声抽出
- 🎭 感情分析（7種類の感情：怒り、嫌悪、恐怖、喜び、平静、悲しみ、驚き）
- 📝 OpenAI Whisperによる音声文字起こし
- 👥 話者分析（pyannote.audioまたはフォールバック方式）
- 🎵 音声メトリクス分析（ピッチ、音量、明度、清涼度）
- ⚡ GPU加速サポート
- 🌸 日本語ローカライズされたUI
- 🔄 セグメント自動分割（無音・時間・話者ベース）
- 🎯 豊富な表示オプション

## インストール

### 必要条件

- Python 3.12以上
- CUDA対応GPU（推奨、高速処理のため）

### ソースからインストール

```bash
git clone https://github.com/your-username/speech-emotion-whisper.git
cd speech-emotion-whisper
pip install -e .
```

## 使用方法

### 基本的な分析

```bash
# 感情分析のみ
python main.py video.mp4 --emotion-analysis

# 文字起こしのみ
python main.py video.mp4 --segment-mode --transcription

# 話者分析のみ（HuggingFaceトークンが必要）
python main.py video.mp4 --segment-mode --speaker-diarization --hf-token YOUR_TOKEN

# すべての分析を実行
python main.py video.mp4 --segment-mode --emotion-analysis --transcription --speaker-diarization --audio-metrics --hf-token YOUR_TOKEN
```

### セグメント分析

```bash
# セグメントごとの感情分析
python main.py video.mp4 --segment-mode --emotion-analysis

# 話者が変わるたびにセグメント分割
python main.py video.mp4 --segment-mode --speaker-segments --speaker-diarization --transcription --hf-token YOUR_TOKEN

# 時間ベースの強制分割（15秒間隔）
python main.py video.mp4 --segment-mode --force-time-split --max-segment-duration 15.0 --emotion-analysis
```

### 高度なオプション

```bash
# 詳細な分析結果表示
python main.py video.mp4 --segment-mode --emotion-analysis --transcription --speaker-diarization --audio-metrics --detailed --metrics-detailed --speaker-detailed --hf-token YOUR_TOKEN

# 日本語音声に特化したモデル使用
python main.py video.mp4 --segment-mode --emotion-analysis --model japanese --transcription --language ja

# 音声ファイルの保存
python main.py video.mp4 --emotion-analysis --output-audio extracted_audio.wav

# CPU専用モード
python main.py video.mp4 --emotion-analysis --no-gpu
```

## コマンドラインオプション

### 分析タイプ

| オプション | 説明 |
|----------|------|
| `--emotion-analysis` | 感情分析を有効化 |
| `--transcription` | OpenAI Whisperによる文字起こし |
| `--speaker-diarization` | 話者分析（要HuggingFaceトークン） |
| `--audio-metrics` | 音声メトリクス分析 |

### セグメント設定

| オプション | デフォルト | 説明 |
|----------|----------|------|
| `--segment-mode` | - | セグメント分析モード |
| `--speaker-segments` | - | 話者変更時にセグメント分割 |
| `--max-segment-duration` | 15.0 | 最大セグメント長（秒） |
| `--min-segment-duration` | 3.0 | 最小セグメント長（秒） |
| `--force-time-split` | - | 時間ベース分割を強制 |

### モデル設定

| オプション | デフォルト | 説明 |
|----------|----------|------|
| `--model` | whisper | 感情分析モデル（whisper/sensevoice/japanese） |
| `--whisper-model` | base | Whisperモデルサイズ（tiny/base/small/medium/large） |
| `--language` | auto | 文字起こし言語（ja/en/auto） |

### 表示オプション

| オプション | 説明 |
|----------|------|
| `--detailed` | 詳細な感情スコア表示 |
| `--metrics-detailed` | 詳細な音声メトリクス表示 |
| `--speaker-detailed` | 詳細な話者分析表示 |

## 出力例

### 複合分析結果

```
🎵 音声・文字起こし・話者タイムライン
┏━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ 時間   ┃ 話者           ┃ テキスト                             ┃ ピッチ         ┃ 音量           ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ 0.0s   │ 話者A          │ こんにちは、今日は...                │ 150Hz          │ 0.123          │
│ 3.5s   │ 話者B          │ はい、お疲れ様...                    │ 180Hz          │ 0.089          │
│ 7.2s   │ 話者A          │ それでは始めま...                    │ 155Hz          │ 0.134          │
└────────┴────────────────┴──────────────────────────────────────┴────────────────┴────────────────┘

📈 統計サマリー
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 項目           ┃ 値                                                                                       ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 総セグメント数 │ 23                                                                                       │
│ 総時間         │ 89.4秒                                                                                   │
│ 平均セグメント長 │ 3.9秒                                                                                    │
│ 平均ピッチ     │ 162 Hz                                                                                   │
│ 検出言語       │ JA                                                                                       │
│ 転写成功率     │ 95.7%                                                                                    │
│ 検出話者数     │ 2                                                                                        │
│ 最多発話者     │ 話者A (13 セグメント)                                                                    │
└────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘
```

## HuggingFace トークンの設定

話者分析機能を使用するには、HuggingFaceトークンが必要です：

1. [HuggingFace Settings](https://hf.co/settings/tokens) でトークンを作成
2. 以下のモデルの利用規約に同意：
   - [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
3. トークンを指定して実行：

```bash
export HF_TOKEN=your_token_here
python main.py video.mp4 --speaker-diarization --hf-token $HF_TOKEN
```

## サポートされるモデル

### 感情分析モデル

- **whisper**: `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3`（英語・多言語対応）
- **sensevoice**: SenseVoice-Small（多言語対応）
- **japanese**: Japanese wav2vec2（日本語特化）

### 文字起こしモデル

- OpenAI Whisper（tiny/base/small/medium/large）
- 90言語以上をサポート

## アーキテクチャ

### コアクラス

- **AudioExtractor**: MP4から音声抽出（MoviePy使用）
- **AudioSegmenter**: 音声セグメント分割（無音・時間・話者ベース）
- **EmotionAnalyzer**: 感情分析（複数モデル対応）
- **TranscriptionAnalyzer**: 文字起こし（OpenAI Whisper）
- **SpeakerDiarization**: 話者分析（pyannote.audio + フォールバック）
- **AudioMetricsAnalyzer**: 音声メトリクス分析
- **ResultDisplay**: 結果表示（Rich UI）

## 依存関係

主要な依存関係は`pyproject.toml`で管理されています：

- `transformers>=4.44.0`: HuggingFace モデル
- `torch>=2.4.0`: PyTorch
- `librosa>=0.10.0`: 音声処理
- `moviepy>=1.0.3`: 動画処理
- `openai-whisper>=20231117`: 文字起こし
- `pyannote.audio>=3.1.0`: 話者分析
- `rich>=13.0.0`: UI表示

## トラブルシューティング

### よくある問題

1. **HuggingFaceゲートモデルエラー**
   - トークンが正しく設定されているか確認
   - モデルの利用規約に同意済みか確認

2. **セグメントが1つしか生成されない**
   - `--force-time-split`オプションを試す
   - `--max-segment-duration`を小さくする

3. **感情分析結果が不正確（日本語音声）**
   - `--model japanese`を使用
   - 日本語特化モデルを選択

4. **GPU メモリエラー**
   - `--no-gpu`でCPUモードを使用
   - 小さなWhisperモデル（tiny/base）を選択

## ライセンス

このプロジェクトは教育・研究目的です。使用されているモデルやデータセットの個別ライセンスをご確認ください。