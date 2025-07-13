import warnings
from typing import Dict, Tuple
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


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