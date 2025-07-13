# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A command-line AI application that extracts audio from MP4 files and performs emotion analysis using OpenAI Whisper Large V3 model (firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3).

## Development Setup

- Python requirement: >=3.12 (specified in pyproject.toml)
- Package management: Uses pyproject.toml for dependency management
- GPU support: CUDA recommended for faster inference

### Installing Dependencies

```bash
pip install -e .
```

## Running the Application

### Basic Usage
```bash
python main.py video.mp4
```

### Advanced Options
```bash
# Save extracted audio
python main.py video.mp4 --output-audio audio.wav

# Show detailed emotion scores
python main.py video.mp4 --detailed

# Disable GPU usage
python main.py video.mp4 --no-gpu

# Using as installed package
speech-emotion video.mp4
```

## Architecture

### Core Classes

1. **AudioExtractor**: Handles MP4 to audio conversion using MoviePy
   - `extract_audio_from_mp4()`: Converts MP4 to WAV format

2. **EmotionAnalyzer**: Performs emotion analysis using HuggingFace transformers
   - Uses `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3` model
   - Supports 7 emotions: angry, disgust, fearful, happy, neutral, sad, surprised
   - GPU acceleration when available
   - `analyze_audio()`: Returns emotion confidence scores
   - `get_dominant_emotion()`: Returns highest-confidence emotion

### Dependencies

- `transformers>=4.44.0`: HuggingFace model loading
- `torch>=2.4.0`: PyTorch for ML inference
- `librosa>=0.10.0`: Audio processing
- `moviepy>=1.0.3`: Video to audio extraction
- `numpy`, `scipy`, `soundfile`: Audio data handling

## Model Information

- Base Model: OpenAI Whisper Large V3 (637M parameters)
- Accuracy: 91.99%
- Trained on RAVDESS, SAVEE, TESS, URDU datasets
- Sample rate: 16kHz
- Input: Audio waveform
- Output: 7-class emotion probabilities