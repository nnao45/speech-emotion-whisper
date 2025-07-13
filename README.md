# Speech Emotion Recognition with OpenAI Whisper Large V3

A command-line AI application that extracts audio from MP4 files and performs emotion analysis using OpenAI Whisper Large V3.

## Features

- <¬ Extract audio from MP4 video files
- <­ Analyze 7 different emotions: angry, disgust, fearful, happy, neutral, sad, surprised
- =€ GPU acceleration support
- =Ê Detailed emotion confidence scores
- =¥ Simple command-line interface

## Installation

### Prerequisites

- Python 3.12 or higher
- CUDA-compatible GPU (recommended for faster processing)

### Install from source

```bash
git clone https://github.com/your-username/speech-emotion-whisper.git
cd speech-emotion-whisper
pip install -e .
```

## Usage

### Basic emotion analysis

```bash
python main.py video.mp4
```

### Save extracted audio

```bash
python main.py video.mp4 --output-audio extracted_audio.wav
```

### Show detailed emotion scores

```bash
python main.py video.mp4 --detailed
```

### Disable GPU (use CPU only)

```bash
python main.py video.mp4 --no-gpu
```

### Using as installed package

```bash
speech-emotion video.mp4
```

## Example Output

```
Processing: sample_video.mp4
Using device: cuda
Extracting audio from MP4...
Analyzing emotions...

==================================================
EMOTION ANALYSIS RESULTS
==================================================
Dominant Emotion: HAPPY
Confidence: 78.45%

Detailed Scores:
------------------------------
Happy        78.45%
Neutral      12.30%
Surprised     4.21%
Sad           2.10%
Angry         1.54%
Fearful       0.89%
Disgust       0.51%
```

## Model Information

This application uses the [firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3](https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3) model:

- **Base Model**: OpenAI Whisper Large V3
- **Parameters**: 637M
- **Accuracy**: 91.99%
- **Training Data**: RAVDESS, SAVEE, TESS, URDU datasets
- **Supported Emotions**: 7 classes

## Requirements

- Python e3.12
- PyTorch e2.4.0
- Transformers e4.44.0
- LibROSA e0.10.0
- MoviePy e1.0.3

## License

This project is for educational and research purposes. Please check the individual licenses of the underlying models and datasets.