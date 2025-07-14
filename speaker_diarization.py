import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import librosa
try:
    import torch
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Using fallback speaker diarization.")


class SpeakerDiarization:
    """Perform speaker diarization using pyannote.audio or fallback methods."""
    
    def __init__(self, use_gpu: bool = True, hf_token: Optional[str] = None):
        """
        Initialize speaker diarization.
        
        Args:
            use_gpu: Whether to use GPU if available
            hf_token: HuggingFace token for accessing gated models (optional)
        """
        self.pipeline = None
        self.use_fallback = False
        
        if not PYANNOTE_AVAILABLE:
            print("🔄 Using fallback speaker diarization (audio-based clustering)")
            self.use_fallback = True
            return
        
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token
        
        print(f"🔄 Loading speaker diarization pipeline on {self.device}...")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Try different model options
                models_to_try = [
                    "pyannote/speaker-diarization-3.1",
                    "pyannote/speaker-diarization"
                ]
                
                for model_name in models_to_try:
                    try:
                        self.pipeline = Pipeline.from_pretrained(
                            model_name,
                            use_auth_token=hf_token
                        )
                        
                        if torch.cuda.is_available() and use_gpu:
                            self.pipeline = self.pipeline.to(torch.device(self.device))
                        
                        print(f"✅ Successfully loaded {model_name}")
                        return
                    except Exception as model_error:
                        print(f"⚠️ Failed to load {model_name}: {model_error}")
                        continue
                
                # All models failed
                raise Exception("All pyannote models failed to load")
                    
        except Exception as e:
            print(f"❌ Error loading speaker diarization pipeline: {e}")
            if "gated" in str(e).lower() or "private" in str(e).lower():
                self._show_hf_token_instructions()
            print("🔄 Falling back to audio-based clustering method...")
            self.pipeline = None
            self.use_fallback = True
    
    def _show_hf_token_instructions(self):
        """Show instructions for getting HuggingFace token."""
        print("\n" + "="*60)
        print("🔑 HuggingFace Token Required")
        print("="*60)
        print("The pyannote models require authentication. To use them:")
        print()
        print("1. Visit: https://hf.co/settings/tokens")
        print("2. Create a new token (read access is sufficient)")
        print("3. Accept the terms for these models:")
        print("   • https://hf.co/pyannote/segmentation-3.0")
        print("   • https://hf.co/pyannote/speaker-diarization-3.1")
        print()
        print("4. Run with your token:")
        print("   python main.py video.mp4 --speaker-diarization --hf-token YOUR_TOKEN")
        print()
        print("5. Or set environment variable:")
        print("   export HF_TOKEN=YOUR_TOKEN")
        print("   python main.py video.mp4 --speaker-diarization --hf-token $HF_TOKEN")
        print("="*60)
        print()
    
    def diarize_audio(self, audio_path: str) -> Dict[str, any]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with speaker diarization results
        """
        try:
            if self.pipeline is None or self.use_fallback:
                # Use audio-based clustering fallback
                return self._fallback_audio_clustering(audio_path)
            
            # Run pyannote speaker diarization
            diarization = self.pipeline(audio_path)
            
            # Extract speaker segments
            speaker_segments = []
            speaker_labels = set()
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "duration": turn.end - turn.start
                })
                speaker_labels.add(speaker)
            
            # Sort by start time
            speaker_segments.sort(key=lambda x: x["start"])
            
            return {
                "segments": speaker_segments,
                "num_speakers": len(speaker_labels),
                "speakers": sorted(list(speaker_labels)),
                "total_speech_duration": sum(seg["duration"] for seg in speaker_segments)
            }
            
        except Exception as e:
            print(f"Warning: Speaker diarization failed for {audio_path}: {e}")
            return self._fallback_audio_clustering(audio_path)
    
    def _fallback_audio_clustering(self, audio_path: str) -> Dict[str, any]:
        """
        Advanced fallback method using audio feature clustering.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Multi-speaker diarization result based on audio features
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)
            duration = len(y) / sr
            
            # Extract MFCC features for speaker identification
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
            
            # Combine features
            features = np.vstack([mfccs, spectral_centroids])
            features = features.T  # Time x Features
            
            # Simple clustering based on feature variance
            segments = self._cluster_audio_features(features, sr, duration)
            
            return segments
            
        except Exception as e:
            print(f"Audio clustering failed: {e}, using single speaker fallback")
            return self._fallback_single_speaker(audio_path)
    
    def _cluster_audio_features(self, features: np.ndarray, sr: int, duration: float) -> Dict[str, any]:
        """
        Cluster audio features to identify speakers.
        
        Args:
            features: Audio features array
            sr: Sample rate
            duration: Audio duration
            
        Returns:
            Speaker segments based on clustering
        """
        # Simple clustering based on feature changes
        frame_duration = 512 / sr  # Duration per feature frame
        
        # Calculate feature variance over sliding windows
        window_size = int(2.0 / frame_duration)  # 2-second windows
        num_frames = len(features)
        
        # Detect speaker changes based on feature variance
        speaker_changes = [0]  # Start with first speaker
        current_speaker = 0
        
        for i in range(window_size, num_frames - window_size, window_size // 2):
            # Compare current window with previous
            window1 = features[i-window_size:i]
            window2 = features[i:i+window_size]
            
            # Calculate feature distance
            if len(window1) > 0 and len(window2) > 0:
                dist = np.linalg.norm(np.mean(window1, axis=0) - np.mean(window2, axis=0))
                
                # If distance is large, assume speaker change
                if dist > np.std(features) * 1.5:  # Threshold based on feature variance
                    current_speaker += 1
                    speaker_changes.append(i * frame_duration)
        
        # Create speaker segments
        speaker_changes.append(duration)  # End of audio
        
        segments = []
        speakers = set()
        
        for i in range(len(speaker_changes) - 1):
            start_time = speaker_changes[i]
            end_time = speaker_changes[i + 1]
            speaker_id = f"SPEAKER_{i:02d}"
            
            segments.append({
                "start": start_time,
                "end": end_time,
                "speaker": speaker_id,
                "duration": end_time - start_time
            })
            speakers.add(speaker_id)
        
        # Merge very short segments (< 1 second) with neighbors
        segments = self._merge_short_segments(segments)
        
        return {
            "segments": segments,
            "num_speakers": len(set(seg["speaker"] for seg in segments)),
            "speakers": sorted(list(set(seg["speaker"] for seg in segments))),
            "total_speech_duration": sum(seg["duration"] for seg in segments)
        }
    
    def _merge_short_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge segments that are too short with adjacent segments."""
        if len(segments) <= 1:
            return segments
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # If segment is too short (< 1 second), try to merge
            if current["duration"] < 1.0 and i > 0:
                # Merge with previous segment
                previous = merged[-1]
                previous["end"] = current["end"]
                previous["duration"] = previous["end"] - previous["start"]
            else:
                merged.append(current)
            
            i += 1
        
        return merged
    
    def _fallback_single_speaker(self, audio_path: str) -> Dict[str, any]:
        """
        Simple fallback: assign all audio to a single speaker.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Single speaker diarization result
        """
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
        except:
            duration = 60.0  # Default estimate
        
        return {
            "segments": [{
                "start": 0.0,
                "end": duration,
                "speaker": "SPEAKER_00",
                "duration": duration
            }],
            "num_speakers": 1,
            "speakers": ["SPEAKER_00"],
            "total_speech_duration": duration
        }
    
    def get_speaker_for_segment(self, segment_start: float, segment_end: float, 
                              diarization_result: Dict[str, any]) -> str:
        """
        Get the dominant speaker for a given time segment.
        
        Args:
            segment_start: Start time of segment in seconds
            segment_end: End time of segment in seconds
            diarization_result: Result from diarize_audio()
            
        Returns:
            Speaker label for the segment
        """
        if not diarization_result or not diarization_result.get("segments"):
            return "SPEAKER_01"
        
        # Find overlapping speaker segments
        speaker_overlaps = {}
        segment_duration = segment_end - segment_start
        
        for speaker_seg in diarization_result["segments"]:
            # Calculate overlap
            overlap_start = max(segment_start, speaker_seg["start"])
            overlap_end = min(segment_end, speaker_seg["end"])
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                speaker = speaker_seg["speaker"]
                
                if speaker not in speaker_overlaps:
                    speaker_overlaps[speaker] = 0
                speaker_overlaps[speaker] += overlap_duration
        
        if not speaker_overlaps:
            return "UNKNOWN"
        
        # Return speaker with most overlap
        dominant_speaker = max(speaker_overlaps.items(), key=lambda x: x[1])[0]
        return dominant_speaker
    
    def get_speaker_stats(self, diarization_result: Dict[str, any]) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each speaker.
        
        Args:
            diarization_result: Result from diarize_audio()
            
        Returns:
            Dictionary with speaker statistics
        """
        if not diarization_result or not diarization_result.get("segments"):
            return {}
        
        speaker_stats = {}
        
        for speaker in diarization_result["speakers"]:
            # Calculate total speaking time for this speaker
            speaker_segments = [seg for seg in diarization_result["segments"] 
                              if seg["speaker"] == speaker]
            
            total_time = sum(seg["duration"] for seg in speaker_segments)
            num_segments = len(speaker_segments)
            avg_segment_length = total_time / num_segments if num_segments > 0 else 0
            
            speaker_stats[speaker] = {
                "total_time": total_time,
                "num_segments": num_segments,
                "avg_segment_length": avg_segment_length,
                "percentage": (total_time / diarization_result["total_speech_duration"] * 100) 
                            if diarization_result["total_speech_duration"] > 0 else 0
            }
        
        return speaker_stats
    
    def format_speaker_label(self, speaker: str) -> str:
        """
        Format speaker label for display.
        
        Args:
            speaker: Raw speaker label
            
        Returns:
            Formatted speaker label
        """
        if speaker.startswith("SPEAKER_"):
            # Convert SPEAKER_00 to 話者A, SPEAKER_01 to 話者B, etc.
            try:
                speaker_num = int(speaker.split("_")[1])
                speaker_char = chr(ord('A') + speaker_num)
                return f"話者{speaker_char}"
            except:
                return speaker
        
        return speaker