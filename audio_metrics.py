from typing import Dict
import numpy as np
import librosa


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