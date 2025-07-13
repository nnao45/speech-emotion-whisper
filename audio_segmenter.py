import os
import tempfile
from typing import List, Tuple
from pydub import AudioSegment
from pydub.silence import split_on_silence


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