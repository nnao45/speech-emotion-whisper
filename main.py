import argparse
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing` to a config initialization is deprecated")
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized from the model checkpoint")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated")

from rich.progress import Progress

# Import our custom modules
from audio_extractor import AudioExtractor
from audio_segmenter import AudioSegmenter
from audio_metrics import AudioMetricsAnalyzer
from emotion_analyzer import EmotionAnalyzer
from result_display import ResultDisplay
from transcription import TranscriptionAnalyzer




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
    parser.add_argument(
        "--transcription", 
        action="store_true", 
        help="Add speech-to-text transcription using OpenAI Whisper"
    )
    parser.add_argument(
        "--whisper-model", 
        choices=["tiny", "base", "small", "medium", "large"], 
        default="base", 
        help="Whisper model size for transcription (default: base)"
    )
    parser.add_argument(
        "--language", 
        default="auto", 
        help="Language for transcription ('auto' for auto-detection, 'ja' for Japanese, etc.)"
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
        transcription_analyzer = TranscriptionAnalyzer(
            model_size=args.whisper_model, 
            use_gpu=not args.no_gpu, 
            language=args.language
        ) if args.transcription else None
        
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
            transcription_results = []
            with Progress() as progress:
                task_parts = ["emotions"]
                if metrics_analyzer:
                    task_parts.append("metrics")
                if transcription_analyzer:
                    task_parts.append("transcription")
                
                task_name = f"[green]Analyzing {' & '.join(task_parts)}..."
                task = progress.add_task(task_name, total=len(segments))
                
                for segment_path, start_time, end_time in segments:
                    try:
                        # Emotion analysis
                        emotion_scores = analyzer.analyze_audio(segment_path)
                        results.append((start_time, end_time, emotion_scores))
                        
                        # Audio metrics analysis
                        if metrics_analyzer:
                            audio_metrics = metrics_analyzer.analyze_audio_metrics(segment_path)
                            metrics_results.append((start_time, end_time, audio_metrics))
                        
                        # Transcription analysis
                        if transcription_analyzer:
                            transcription = transcription_analyzer.transcribe_audio(segment_path)
                            transcription_results.append((start_time, end_time, transcription))
                        
                        progress.advance(task)
                    except Exception as e:
                        display.console.print(f"[red]❌ Error analyzing segment {start_time:.1f}s-{end_time:.1f}s: {e}[/red]")
                        continue
            
            # Display segment results
            if results:
                display.show_segment_results(
                    results, 
                    metrics_results if metrics_analyzer else None,
                    transcription_results if transcription_analyzer else None
                )
                
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
