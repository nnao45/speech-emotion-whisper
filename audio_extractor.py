import tempfile
from moviepy import VideoFileClip


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