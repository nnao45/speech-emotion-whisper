from typing import Dict, List, Tuple
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


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