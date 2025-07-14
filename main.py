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
from speaker_diarization import SpeakerDiarization




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
    parser.add_argument(
        "--speaker-diarization", 
        action="store_true", 
        help="Enable speaker diarization (identify different speakers)"
    )
    parser.add_argument(
        "--hf-token", 
        default=None, 
        help="HuggingFace token for accessing gated models (required for speaker diarization)"
    )
    parser.add_argument(
        "--speaker-detailed", 
        action="store_true", 
        help="Show detailed speaker analysis (requires --speaker-diarization)"
    )
    parser.add_argument(
        "--speaker-segments", 
        action="store_true", 
        help="Split segments at speaker changes (requires --speaker-diarization and --segment-mode)"
    )
    parser.add_argument(
        "--emotion-analysis", 
        action="store_true", 
        help="Enable emotion analysis (by default, only transcription and speaker analysis are performed)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return 1
    
    # Validate speaker-segments option
    if args.speaker_segments and not args.speaker_diarization:
        print("Error: --speaker-segments requires --speaker-diarization")
        return 1
    
    if args.speaker_segments and not args.segment_mode:
        print("Error: --speaker-segments requires --segment-mode")
        return 1
    
    # Check if at least one analysis type is enabled
    if not any([args.emotion_analysis, args.transcription, args.speaker_diarization, args.audio_metrics]):
        print("Error: At least one analysis type must be enabled (--emotion-analysis, --transcription, --speaker-diarization, or --audio-metrics)")
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
        analyzer = EmotionAnalyzer(model_type=args.model, use_gpu=not args.no_gpu) if args.emotion_analysis else None
        metrics_analyzer = AudioMetricsAnalyzer() if args.audio_metrics else None
        transcription_analyzer = TranscriptionAnalyzer(
            model_size=args.whisper_model, 
            use_gpu=not args.no_gpu, 
            language=args.language
        ) if args.transcription else None
        speaker_analyzer = SpeakerDiarization(
            use_gpu=not args.no_gpu,
            hf_token=args.hf_token
        ) if args.speaker_diarization else None
        
        if args.segment_mode:
            # Segment-based analysis
            segmenter = AudioSegmenter(
                min_silence_len=int(args.min_silence_duration * 1000),  # Convert to ms
                silence_thresh=args.silence_threshold,
                min_segment_len=int(args.min_segment_duration * 1000),   # Convert to ms
                max_segment_len=int(args.max_segment_duration * 1000),   # Convert to ms
                force_time_split=args.force_time_split
            )
            
            # Perform speaker diarization first if speaker-based segmentation is requested
            diarization_result = None
            if args.speaker_segments and speaker_analyzer:
                try:
                    with Progress() as progress:
                        task = progress.add_task("[green]Performing speaker diarization for segmentation...", total=1)
                        diarization_result = speaker_analyzer.diarize_audio(audio_path)
                        progress.advance(task)
                    display.console.print(f"[green]✅ Speaker diarization complete:[/green] {diarization_result.get('num_speakers', 0)} speakers detected")
                except Exception as e:
                    display.console.print(f"[yellow]⚠️ Speaker diarization for segmentation failed: {e}[/yellow]")
                    display.console.print("[yellow]Falling back to silence/time-based segmentation[/yellow]")
                    diarization_result = None
            
            segments = segmenter.segment_audio(audio_path, diarization_result)
            
            if not segments:
                display.console.print("[yellow]⚠️ No segments found. Audio might be too quiet or short.[/yellow]")
                return 1
            
            # Perform speaker diarization if enabled and not already done for segmentation
            if speaker_analyzer and not args.speaker_segments:
                try:
                    with Progress() as progress:
                        task = progress.add_task("[green]Performing speaker diarization...", total=1)
                        diarization_result = speaker_analyzer.diarize_audio(audio_path)
                        progress.advance(task)
                except Exception as e:
                    display.console.print(f"[yellow]⚠️ Speaker diarization failed: {e}[/yellow]")
                    speaker_analyzer = None
                    diarization_result = None
            
            # Analyze each segment with progress bar
            results = []
            metrics_results = []
            transcription_results = []
            speaker_results = []
            with Progress() as progress:
                task_parts = []
                if analyzer:
                    task_parts.append("emotions")
                if metrics_analyzer:
                    task_parts.append("metrics")
                if transcription_analyzer:
                    task_parts.append("transcription")
                if speaker_analyzer:
                    task_parts.append("speakers")
                
                task_name = f"[green]Analyzing {' & '.join(task_parts)}..."
                task = progress.add_task(task_name, total=len(segments))
                
                for segment_path, start_time, end_time in segments:
                    try:
                        # Emotion analysis (if enabled)
                        if analyzer:
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
                        
                        # Speaker analysis
                        if speaker_analyzer and diarization_result:
                            speaker = speaker_analyzer.get_speaker_for_segment(
                                start_time, end_time, diarization_result
                            )
                            formatted_speaker = speaker_analyzer.format_speaker_label(speaker)
                            speaker_results.append((start_time, end_time, formatted_speaker))
                        
                        progress.advance(task)
                    except Exception as e:
                        display.console.print(f"[red]❌ Error analyzing segment {start_time:.1f}s-{end_time:.1f}s: {e}[/red]")
                        continue
            
            # Display segment results
            if results or metrics_results or transcription_results or speaker_results:
                display.show_segment_results(
                    results if analyzer else None, 
                    metrics_results if metrics_analyzer else None,
                    transcription_results if transcription_analyzer else None,
                    speaker_results if speaker_analyzer else None
                )
                
                # Show detailed metrics if requested
                if args.metrics_detailed and metrics_results:
                    display.show_audio_metrics_details(metrics_results)
                
                # Show detailed speaker analysis if requested
                if args.speaker_detailed and speaker_results:
                    display.show_speaker_summary(speaker_results)
            else:
                display.console.print("[red]❌ No segments could be analyzed.[/red]")
                return 1
            
            # Clean up segment files
            segmenter.cleanup_segments(segments)
            
        else:
            # Full audio analysis
            if analyzer:
                with Progress() as progress:
                    task = progress.add_task("[green]Analyzing emotions...", total=1)
                    emotion_scores = analyzer.analyze_audio(audio_path)
                    progress.advance(task)
                
                # Display single result
                display.show_single_result(emotion_scores, args.detailed)
            else:
                display.console.print("[yellow]⚠️ No analysis enabled for full audio mode. Use --emotion-analysis or --segment-mode with other analysis options.[/yellow]")
        
        # Clean up temporary audio file if created
        if not args.output_audio and os.path.exists(audio_path):
            os.unlink(audio_path)
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
