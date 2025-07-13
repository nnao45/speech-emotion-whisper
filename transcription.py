import warnings
from typing import Dict, Optional
import whisper
import torch


class TranscriptionAnalyzer:
    """Transcribe audio using OpenAI Whisper models."""
    
    def __init__(self, model_size: str = "base", use_gpu: bool = True, language: str = "auto"):
        """
        Initialize transcription analyzer.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            use_gpu: Whether to use GPU if available
            language: Target language for transcription ('auto' for auto-detection, 'ja' for Japanese, etc.)
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model_size = model_size
        self.language = language if language != "auto" else None
        
        print(f"Loading Whisper {model_size} model on {self.device}...")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = whisper.load_model(model_size, device=self.device)
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Falling back to tiny model...")
            self.model_size = "tiny"
            self.model = whisper.load_model("tiny", device=self.device)
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, str]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                task="transcribe",
                fp16=False if self.device == "cpu" else True,
                verbose=False
            )
            
            # Extract text and clean it
            full_text = result.get("text", "").strip()
            
            # Get first 10 characters for preview
            preview_text = self._clean_text_for_display(full_text)
            
            return {
                "full_text": full_text,
                "preview_text": preview_text,
                "language": result.get("language", "unknown"),
                "confidence": self._calculate_avg_confidence(result)
            }
            
        except Exception as e:
            print(f"Warning: Transcription failed for {audio_path}: {e}")
            return {
                "full_text": "",
                "preview_text": "[転写失敗]",
                "language": "unknown",
                "confidence": 0.0
            }
    
    def _clean_text_for_display(self, text: str, max_length: int = 10) -> str:
        """
        Clean and truncate text for display in table.
        
        Args:
            text: Full transcription text
            max_length: Maximum characters to display
            
        Returns:
            Cleaned preview text
        """
        if not text:
            return "[無音]"
        
        # Remove extra whitespace and newlines
        cleaned = " ".join(text.split())
        
        # Truncate to max_length characters
        if len(cleaned) > max_length:
            # Try to break at word boundary if possible
            truncated = cleaned[:max_length]
            if max_length < len(cleaned) and cleaned[max_length] != " ":
                # Find last space within the limit
                last_space = truncated.rfind(" ")
                if last_space > max_length // 2:  # Only if we don't lose too much
                    truncated = truncated[:last_space]
            return truncated + "..."
        
        return cleaned
    
    def _calculate_avg_confidence(self, result: Dict) -> float:
        """
        Calculate average confidence from Whisper segments.
        
        Args:
            result: Whisper transcription result
            
        Returns:
            Average confidence score (0.0 to 1.0)
        """
        segments = result.get("segments", [])
        if not segments:
            return 0.0
        
        # Calculate average confidence if available in segments
        confidences = []
        for segment in segments:
            if "avg_logprob" in segment:
                # Convert log probability to confidence (approximate)
                confidence = min(1.0, max(0.0, (segment["avg_logprob"] + 1.0)))
                confidences.append(confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_size": self.model_size,
            "device": self.device,
            "language": self.language or "auto-detect",
            "parameters": self._get_model_params()
        }
    
    def _get_model_params(self) -> str:
        """Get approximate parameter count for the model."""
        param_counts = {
            "tiny": "39M",
            "base": "74M", 
            "small": "244M",
            "medium": "769M",
            "large": "1550M"
        }
        return param_counts.get(self.model_size, "unknown")