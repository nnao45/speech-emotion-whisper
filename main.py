import argparse
import os
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress warnings
warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing` to a config initialization is deprecated")
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized from the model checkpoint")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated")

import librosa
import numpy as np
import torch
from moviepy import VideoFileClip
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from pydub import AudioSegment
from pydub.silence import split_on_silence
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID
from scipy.stats import skew, kurtosis


class AudioExtractor:
    """Extract audio from MP4 files."""
    
    @staticmethod
    def extract_audio_from_mp4(video_path: str, output_path: str = None) -> str:
        """
        Extract audio from MP4 file and save as WAV.
        
        Args:
            video_path: Path to input MP4 file
            output_path: Path to save extracted audio (optional)
            
        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")
            
        try:
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(output_path, logger=None)
            video.close()
            audio.close()
            return output_path
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio from {video_path}: {e}")


class AudioSegmenter:
    """Segment audio into conversation parts."""
    
    def __init__(self, min_silence_len: int = 1000, silence_thresh: int = -40, min_segment_len: int = 3000, 
                 max_segment_len: int = 30000, force_time_split: bool = False):
        """
        Initialize audio segmenter.
        
        Args:
            min_silence_len: Minimum length of silence to be used for splitting (ms)
            silence_thresh: Silence threshold in dBFS
            min_segment_len: Minimum segment length in ms
            max_segment_len: Maximum segment length in ms (fallback time-based split)
            force_time_split: Force time-based splitting instead of silence-based
        """
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.min_segment_len = min_segment_len
        self.max_segment_len = max_segment_len
        self.force_time_split = force_time_split
    
    def segment_audio(self, audio_path: str) -> List[Tuple[str, float, float]]:
        """
        Split audio file into segments based on silence or time.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of tuples (segment_file_path, start_time, end_time)
        """
        try:
            # Load audio with pydub
            audio = AudioSegment.from_wav(audio_path)
            
            if self.force_time_split:
                return self._time_based_split(audio)
            
            # Try silence-based splitting first
            chunks = split_on_silence(
                audio,
                min_silence_len=self.min_silence_len,
                silence_thresh=self.silence_thresh,
                keep_silence=500  # Keep 500ms of silence at start and end
            )
            
            # If silence-based splitting results in too few segments or segments too long, fallback to time-based
            if len(chunks) <= 1 or any(len(chunk) > self.max_segment_len for chunk in chunks):
                print(f"Silence-based segmentation ineffective ({len(chunks)} segments found). Using time-based fallback.")
                return self._time_based_split(audio)
            
            segments = []
            current_pos = 0
            
            for i, chunk in enumerate(chunks):
                # Skip segments that are too short
                if len(chunk) < self.min_segment_len:
                    current_pos += len(chunk)
                    continue
                
                # Create temporary file for segment
                segment_path = tempfile.mktemp(suffix=f"_segment_{i}.wav")
                chunk.export(segment_path, format="wav")
                
                # Calculate timing
                start_time = current_pos / 1000.0  # Convert to seconds
                end_time = (current_pos + len(chunk)) / 1000.0
                
                segments.append((segment_path, start_time, end_time))
                current_pos += len(chunk)
            
            return segments
            
        except Exception as e:
            raise RuntimeError(f"Failed to segment audio {audio_path}: {e}")
    
    def _time_based_split(self, audio: AudioSegment) -> List[Tuple[str, float, float]]:
        """
        Split audio into fixed-time segments.
        
        Args:
            audio: AudioSegment object
            
        Returns:
            List of tuples (segment_file_path, start_time, end_time)
        """
        segments = []
        audio_length_ms = len(audio)
        current_pos = 0
        segment_idx = 0
        
        while current_pos < audio_length_ms:
            # Calculate end position for this segment
            end_pos = min(current_pos + self.max_segment_len, audio_length_ms)
            
            # Extract segment
            chunk = audio[current_pos:end_pos]
            
            # Skip if segment is too short (unless it's the last segment)
            if len(chunk) < self.min_segment_len and end_pos < audio_length_ms:
                current_pos = end_pos
                continue
            
            # Create temporary file for segment
            segment_path = tempfile.mktemp(suffix=f"_time_segment_{segment_idx}.wav")
            chunk.export(segment_path, format="wav")
            
            # Calculate timing
            start_time = current_pos / 1000.0
            end_time = end_pos / 1000.0
            
            segments.append((segment_path, start_time, end_time))
            
            current_pos = end_pos
            segment_idx += 1
        
        return segments
    
    def cleanup_segments(self, segments: List[Tuple[str, float, float]]):
        """Clean up temporary segment files."""
        for segment_path, _, _ in segments:
            if os.path.exists(segment_path):
                os.unlink(segment_path)


class AudioMetricsAnalyzer:
    """Analyze various audio metrics like pitch, tone, clarity, volume, etc."""
    
    def __init__(self):
        """Initialize audio metrics analyzer."""
        pass
    
    def analyze_audio_metrics(self, audio_path: str) -> Dict[str, float]:
        """
        Analyze comprehensive audio metrics.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio metrics
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            metrics = {}
            
            # Pitch analysis
            pitch_metrics = self._analyze_pitch(y, sr)
            metrics.update(pitch_metrics)
            
            # Volume/Energy analysis
            volume_metrics = self._analyze_volume(y)
            metrics.update(volume_metrics)
            
            # Spectral analysis (tone, brightness, clarity)
            spectral_metrics = self._analyze_spectral_features(y, sr)
            metrics.update(spectral_metrics)
            
            # Temporal analysis
            temporal_metrics = self._analyze_temporal_features(y, sr)
            metrics.update(temporal_metrics)
            
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze audio metrics {audio_path}: {e}")
    
    def _analyze_pitch(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze pitch-related metrics."""
        # Extract fundamental frequency using librosa
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        
        # Filter out unvoiced frames
        voiced_f0 = f0[voiced_flag]
        
        if len(voiced_f0) == 0:
            return {
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_min': 0.0,
                'pitch_max': 0.0,
                'pitch_range': 0.0,
                'voiced_ratio': 0.0
            }
        
        return {
            'pitch_mean': float(np.mean(voiced_f0)),
            'pitch_std': float(np.std(voiced_f0)),
            'pitch_min': float(np.min(voiced_f0)),
            'pitch_max': float(np.max(voiced_f0)),
            'pitch_range': float(np.max(voiced_f0) - np.min(voiced_f0)),
            'voiced_ratio': float(len(voiced_f0) / len(f0))
        }
    
    def _analyze_volume(self, y: np.ndarray) -> Dict[str, float]:
        """Analyze volume/energy metrics."""
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Zero crossing rate (indicates noisiness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        return {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'rms_max': float(np.max(rms)),
            'dynamic_range': float(np.max(rms) - np.min(rms)),
            'zero_crossing_rate': float(np.mean(zcr))
        }
    
    def _analyze_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze spectral features (tone, brightness, clarity)."""
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral rolloff (measure of brightness)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Spectral bandwidth (tonal vs. noisy)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        # Spectral contrast (clarity)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # MFCCs for tonal characteristics
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_contrast_mean': float(np.mean(spectral_contrast)),
            'brightness_score': float(np.mean(spectral_centroids) / (sr/2) * 100),  # Normalized 0-100
            'clarity_score': float(np.mean(spectral_contrast) * 10),  # Scaled for readability
            'mfcc_1_mean': float(np.mean(mfccs[0])),  # First MFCC (overall spectrum shape)
            'mfcc_2_mean': float(np.mean(mfccs[1]))   # Second MFCC (spectral slope)
        }
    
    def _analyze_temporal_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze temporal features."""
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Onset detection (speech rate proxy)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Speaking rate (onsets per second)
        duration = len(y) / sr
        speaking_rate = len(onset_times) / duration if duration > 0 else 0
        
        return {
            'tempo': float(tempo.item() if hasattr(tempo, 'item') else tempo),
            'speaking_rate': float(speaking_rate),
            'duration': float(duration)
        }


class EmotionAnalyzer:
    """Analyze emotions from audio using various emotion recognition models."""
    
    def __init__(self, model_type: str = "whisper", use_gpu: bool = True):
        """
        Initialize emotion analyzer.
        
        Args:
            model_type: Type of model to use ('whisper', 'sensevoice', 'japanese')
            use_gpu: Whether to use GPU if available
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        print(f"Using device: {self.device}")
        
        # Model configurations
        self.model_configs = {
            "whisper": {
                "model_id": "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
                "emotions": ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
                "description": "Whisper Large V3 (English-focused)"
            },
            "sensevoice": {
                "model_id": "FunAudioLLM/SenseVoiceSmall",
                "emotions": ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
                "description": "SenseVoice Small (Multilingual: JP/EN/CN/KR)"
            },
            "japanese": {
                "model_id": "Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition",
                "emotions": ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
                "description": "Wav2Vec2 (Japanese-specific)"
            }
        }
        
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}. Choose from: {list(self.model_configs.keys())}")
        
        config = self.model_configs[model_type]
        print(f"Loading model: {config['description']}")
        print(f"Model ID: {config['model_id']}")
        
        self.model_id = config["model_id"]
        self.emotions = config["emotions"]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_id).to(self.device)
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        except Exception as e:
            print(f"Error loading model {self.model_id}: {e}")
            print("Falling back to Whisper model...")
            self.model_type = "whisper"
            fallback_config = self.model_configs["whisper"]
            self.model_id = fallback_config["model_id"]
            self.emotions = fallback_config["emotions"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_id).to(self.device)
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
    
    def analyze_audio(self, audio_path: str) -> Dict[str, float]:
        """
        Analyze emotions in audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with emotion scores
        """
        try:
            # Load audio using librosa
            audio, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Extract features
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=sample_rate, 
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            scores = probabilities.cpu().numpy()[0]
            
            # Create emotion-score mapping
            emotion_scores = dict(zip(self.emotions, scores))
            
            return emotion_scores
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze audio {audio_path}: {e}")
    
    def get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Get the dominant emotion and its confidence score.
        
        Args:
            emotion_scores: Dictionary with emotion scores
            
        Returns:
            Tuple of (emotion_name, confidence_score)
        """
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion]
        return dominant_emotion, confidence


class ResultDisplay:
    """Display emotion analysis results with rich formatting."""
    
    def __init__(self):
        self.console = Console()
    
    def show_processing_start(self, file_path: str, mode: str):
        """Show processing start message."""
        panel = Panel(
            f"[bold blue]🎬 Processing:[/bold blue] {file_path}\n"
            f"[bold yellow]📊 Mode:[/bold yellow] {mode}",
            title="[bold green]Speech Emotion Recognition[/bold green]",
            border_style="blue"
        )
        self.console.print(panel)
    
    def show_single_result(self, emotion_scores: Dict[str, float], detailed: bool = False):
        """Display results for single audio analysis."""
        dominant_emotion, confidence = max(emotion_scores.items(), key=lambda x: x[1])
        
        # Main result panel
        result_text = f"[bold red]{dominant_emotion.upper()}[/bold red]\n"
        result_text += f"[yellow]Confidence: {confidence:.2%}[/yellow]"
        
        panel = Panel(
            result_text,
            title="[bold green]🎯 Dominant Emotion[/bold green]",
            border_style="green"
        )
        self.console.print(panel)
        
        if detailed:
            self._show_detailed_scores(emotion_scores)
    
    def show_segment_results(self, results: List[Tuple[float, float, Dict[str, float]]], 
                           metrics_results: List[Tuple[float, float, Dict[str, float]]] = None):
        """Display results for segmented audio analysis."""
        if metrics_results:
            # Show combined emotion and audio metrics table
            table = Table(title="🎭 Emotion & Audio Metrics Timeline")
            table.add_column("Time", style="cyan", no_wrap=True)
            table.add_column("Emotion", style="red", no_wrap=True)
            table.add_column("Conf", style="green", no_wrap=True)
            table.add_column("Pitch", style="yellow", no_wrap=True)
            table.add_column("Bright", style="blue", no_wrap=True)
            table.add_column("Clarity", style="magenta", no_wrap=True)
            table.add_column("Volume", style="white", no_wrap=True)
            
            for i, ((start_time, end_time, emotion_scores), (_, _, audio_metrics)) in enumerate(zip(results, metrics_results)):
                dominant_emotion, confidence = max(emotion_scores.items(), key=lambda x: x[1])
                
                table.add_row(
                    f"{start_time:.1f}s",
                    dominant_emotion.upper()[:4],
                    f"{confidence:.0%}",
                    f"{audio_metrics.get('pitch_mean', 0):.0f}Hz",
                    f"{audio_metrics.get('brightness_score', 0):.0f}",
                    f"{audio_metrics.get('clarity_score', 0):.1f}",
                    f"{audio_metrics.get('rms_mean', 0):.3f}"
                )
        else:
            # Show emotion-only table
            table = Table(title="🎭 Emotion Analysis Timeline")
            table.add_column("Time", style="cyan", no_wrap=True)
            table.add_column("Duration", style="magenta")
            table.add_column("Emotion", style="red", no_wrap=True)
            table.add_column("Confidence", style="green")
            
            for start_time, end_time, emotion_scores in results:
                dominant_emotion, confidence = max(emotion_scores.items(), key=lambda x: x[1])
                duration = end_time - start_time
                
                table.add_row(
                    f"{start_time:.1f}s",
                    f"{duration:.1f}s",
                    dominant_emotion.upper(),
                    f"{confidence:.1%}"
                )
        
        self.console.print(table)
        
        # Show summary statistics
        self._show_summary_stats(results, metrics_results)
    
    def _show_detailed_scores(self, emotion_scores: Dict[str, float]):
        """Show detailed emotion scores table."""
        table = Table(title="📊 Detailed Emotion Scores")
        table.add_column("Emotion", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Bar", style="blue")
        
        for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(score * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            table.add_row(
                emotion.capitalize(),
                f"{score:.2%}",
                bar
            )
        
        self.console.print(table)
    
    def _show_summary_stats(self, results: List[Tuple[float, float, Dict[str, float]]], 
                          metrics_results: List[Tuple[float, float, Dict[str, float]]] = None):
        """Show summary statistics for segment analysis."""
        if not results:
            return
        
        # Calculate emotion distribution
        emotion_counts = {}
        total_duration = 0
        
        for start_time, end_time, emotion_scores in results:
            dominant_emotion, _ = max(emotion_scores.items(), key=lambda x: x[1])
            emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
            total_duration += end_time - start_time
        
        # Create summary table
        table = Table(title="📈 Summary Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Segments", str(len(results)))
        table.add_row("Total Duration", f"{total_duration:.1f}s")
        table.add_row("Avg Segment Length", f"{total_duration/len(results):.1f}s")
        
        # Most common emotion
        most_common = max(emotion_counts.items(), key=lambda x: x[1])
        table.add_row("Most Common Emotion", f"{most_common[0].upper()} ({most_common[1]} segments)")
        
        # Add audio metrics summary if available
        if metrics_results:
            all_pitch = [metrics.get('pitch_mean', 0) for _, _, metrics in metrics_results if metrics.get('pitch_mean', 0) > 0]
            all_brightness = [metrics.get('brightness_score', 0) for _, _, metrics in metrics_results]
            all_clarity = [metrics.get('clarity_score', 0) for _, _, metrics in metrics_results]
            all_volume = [metrics.get('rms_mean', 0) for _, _, metrics in metrics_results]
            
            if all_pitch:
                table.add_row("Avg Pitch", f"{np.mean(all_pitch):.0f} Hz")
            if all_brightness:
                table.add_row("Avg Brightness", f"{np.mean(all_brightness):.1f}")
            if all_clarity:
                table.add_row("Avg Clarity", f"{np.mean(all_clarity):.1f}")
            if all_volume:
                table.add_row("Avg Volume", f"{np.mean(all_volume):.3f}")
        
        self.console.print(table)
    
    def show_audio_metrics_details(self, metrics_results: List[Tuple[float, float, Dict[str, float]]]):
        """Show detailed audio metrics analysis."""
        if not metrics_results:
            return
        
        # Pitch analysis table
        pitch_table = Table(title="🎵 Pitch Analysis")
        pitch_table.add_column("Time", style="cyan")
        pitch_table.add_column("Mean Pitch", style="yellow")
        pitch_table.add_column("Pitch Range", style="magenta")
        pitch_table.add_column("Voiced Ratio", style="green")
        
        for start_time, end_time, metrics in metrics_results:
            pitch_table.add_row(
                f"{start_time:.1f}s",
                f"{metrics.get('pitch_mean', 0):.0f} Hz",
                f"{metrics.get('pitch_range', 0):.0f} Hz",
                f"{metrics.get('voiced_ratio', 0):.1%}"
            )
        
        self.console.print(pitch_table)
        
        # Spectral analysis table
        spectral_table = Table(title="🌈 Spectral Analysis")
        spectral_table.add_column("Time", style="cyan")
        spectral_table.add_column("Brightness", style="blue")
        spectral_table.add_column("Clarity", style="magenta")
        spectral_table.add_column("Bandwidth", style="yellow")
        
        for start_time, end_time, metrics in metrics_results:
            spectral_table.add_row(
                f"{start_time:.1f}s",
                f"{metrics.get('brightness_score', 0):.1f}",
                f"{metrics.get('clarity_score', 0):.1f}",
                f"{metrics.get('spectral_bandwidth_mean', 0):.0f} Hz"
            )
        
        self.console.print(spectral_table)


def main():
    parser = argparse.ArgumentParser(
        description="Speech emotion recognition from MP4 files using OpenAI Whisper Large V3"
    )
    parser.add_argument(
        "input_file", 
        help="Path to input MP4 file"
    )
    parser.add_argument(
        "--output-audio", 
        help="Path to save extracted audio (optional)"
    )
    parser.add_argument(
        "--no-gpu", 
        action="store_true", 
        help="Disable GPU usage"
    )
    parser.add_argument(
        "--detailed", 
        action="store_true", 
        help="Show detailed emotion scores"
    )
    parser.add_argument(
        "--segment-mode", 
        action="store_true", 
        help="Analyze emotion for each conversation segment"
    )
    parser.add_argument(
        "--min-segment-duration", 
        type=float, 
        default=3.0, 
        help="Minimum segment duration in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--silence-threshold", 
        type=int, 
        default=-40, 
        help="Silence threshold in dBFS (default: -40)"
    )
    parser.add_argument(
        "--min-silence-duration", 
        type=float, 
        default=1.0, 
        help="Minimum silence duration for splitting in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--max-segment-duration", 
        type=float, 
        default=15.0, 
        help="Maximum segment duration for time-based fallback in seconds (default: 15.0)"
    )
    parser.add_argument(
        "--force-time-split", 
        action="store_true", 
        help="Force time-based segmentation instead of silence-based"
    )
    parser.add_argument(
        "--model", 
        choices=["whisper", "sensevoice", "japanese"], 
        default="whisper", 
        help="Emotion recognition model to use (default: whisper)"
    )
    parser.add_argument(
        "--audio-metrics", 
        action="store_true", 
        help="Analyze audio metrics (pitch, tone, clarity, volume) alongside emotions"
    )
    parser.add_argument(
        "--metrics-detailed", 
        action="store_true", 
        help="Show detailed audio metrics tables (requires --audio-metrics)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return 1
    
    try:
        # Initialize display
        display = ResultDisplay()
        mode = "Segment Analysis" if args.segment_mode else "Full Audio Analysis"
        display.show_processing_start(args.input_file, mode)
        
        # Extract audio from MP4
        extractor = AudioExtractor()
        audio_path = extractor.extract_audio_from_mp4(args.input_file, args.output_audio)
        
        if args.output_audio:
            display.console.print(f"[green]✅ Audio saved to:[/green] {audio_path}")
        
        # Initialize analyzers
        analyzer = EmotionAnalyzer(model_type=args.model, use_gpu=not args.no_gpu)
        metrics_analyzer = AudioMetricsAnalyzer() if args.audio_metrics else None
        
        if args.segment_mode:
            # Segment-based analysis
            segmenter = AudioSegmenter(
                min_silence_len=int(args.min_silence_duration * 1000),  # Convert to ms
                silence_thresh=args.silence_threshold,
                min_segment_len=int(args.min_segment_duration * 1000),   # Convert to ms
                max_segment_len=int(args.max_segment_duration * 1000),   # Convert to ms
                force_time_split=args.force_time_split
            )
            
            segments = segmenter.segment_audio(audio_path)
            
            if not segments:
                display.console.print("[yellow]⚠️ No segments found. Audio might be too quiet or short.[/yellow]")
                return 1
            
            # Analyze each segment with progress bar
            results = []
            metrics_results = []
            with Progress() as progress:
                task_name = "[green]Analyzing emotions & metrics..." if metrics_analyzer else "[green]Analyzing emotions..."
                task = progress.add_task(task_name, total=len(segments))
                
                for segment_path, start_time, end_time in segments:
                    try:
                        emotion_scores = analyzer.analyze_audio(segment_path)
                        results.append((start_time, end_time, emotion_scores))
                        
                        if metrics_analyzer:
                            audio_metrics = metrics_analyzer.analyze_audio_metrics(segment_path)
                            metrics_results.append((start_time, end_time, audio_metrics))
                        
                        progress.advance(task)
                    except Exception as e:
                        display.console.print(f"[red]❌ Error analyzing segment {start_time:.1f}s-{end_time:.1f}s: {e}[/red]")
                        continue
            
            # Display segment results
            if results:
                display.show_segment_results(results, metrics_results if metrics_analyzer else None)
                
                # Show detailed metrics if requested
                if args.metrics_detailed and metrics_results:
                    display.show_audio_metrics_details(metrics_results)
            else:
                display.console.print("[red]❌ No segments could be analyzed.[/red]")
                return 1
            
            # Clean up segment files
            segmenter.cleanup_segments(segments)
            
        else:
            # Full audio analysis
            with Progress() as progress:
                task = progress.add_task("[green]Analyzing emotions...", total=1)
                emotion_scores = analyzer.analyze_audio(audio_path)
                progress.advance(task)
            
            # Display single result
            display.show_single_result(emotion_scores, args.detailed)
        
        # Clean up temporary audio file if created
        if not args.output_audio and os.path.exists(audio_path):
            os.unlink(audio_path)
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
