from typing import Dict, List, Tuple
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


class ResultDisplay:
    """Display emotion analysis results with rich formatting."""
    
    def __init__(self):
        self.console = Console()
        
        # Emotion label translation dictionary (English -> Japanese)
        self.emotion_translations = {
            "angry": "怒り",
            "disgust": "嫌悪", 
            "fearful": "恐怖",
            "happy": "喜び",
            "neutral": "平静",
            "sad": "悲しみ",
            "surprised": "驚き"
        }
    
    def _translate_emotion(self, emotion: str) -> str:
        """
        Translate emotion label to Japanese.
        
        Args:
            emotion: English emotion label
            
        Returns:
            Japanese emotion label
        """
        return self.emotion_translations.get(emotion.lower(), emotion)
    
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
        dominant_emotion_jp = self._translate_emotion(dominant_emotion)
        
        # Main result panel
        result_text = f"[bold red]{dominant_emotion_jp}[/bold red]\n"
        result_text += f"[yellow]信頼度: {confidence:.2%}[/yellow]"
        
        panel = Panel(
            result_text,
            title="[bold green]🎯 主要な感情[/bold green]",
            border_style="green"
        )
        self.console.print(panel)
        
        if detailed:
            self._show_detailed_scores(emotion_scores)
    
    def show_segment_results(self, results: List[Tuple[float, float, Dict[str, float]]] = None, 
                           metrics_results: List[Tuple[float, float, Dict[str, float]]] = None,
                           transcription_results: List[Tuple[float, float, Dict[str, str]]] = None,
                           speaker_results: List[Tuple[float, float, str]] = None):
        """Display results for segmented audio analysis."""
        # Handle case where no results are provided
        if not any([results, metrics_results, transcription_results, speaker_results]):
            self.console.print("[yellow]⚠️ No analysis results to display.[/yellow]")
            return
        
        if metrics_results and transcription_results and speaker_results and not results:
            # Show metrics, transcription, and speaker table (no emotions)
            table = Table(title="🎵 音声・文字起こし・話者タイムライン")
            table.add_column("時間", style="cyan", no_wrap=True)
            table.add_column("話者", style="bright_blue", no_wrap=True)
            table.add_column("テキスト", style="bright_white", no_wrap=True, max_width=30)
            table.add_column("ピッチ", style="yellow", no_wrap=True)
            table.add_column("音量", style="white", no_wrap=True)
            
            for i, ((start_time, end_time, audio_metrics), (_, _, transcription), (_, _, speaker)) in enumerate(zip(metrics_results, transcription_results, speaker_results)):
                table.add_row(
                    f"{start_time:.1f}s",
                    speaker,
                    transcription.get('preview_text', '[空]'),
                    f"{audio_metrics.get('pitch_mean', 0):.0f}Hz",
                    f"{audio_metrics.get('rms_mean', 0):.3f}"
                )
        elif results and metrics_results and transcription_results and speaker_results:
            # Show full combined table with all features
            table = Table(title="🎭 感情・音声・文字起こし・話者タイムライン")
            table.add_column("時間", style="cyan", no_wrap=True)
            table.add_column("話者", style="bright_blue", no_wrap=True)
            table.add_column("テキスト", style="bright_white", no_wrap=True, max_width=30)
            table.add_column("感情", style="red", no_wrap=True)
            table.add_column("信頼度", style="green", no_wrap=True)
            table.add_column("ピッチ", style="yellow", no_wrap=True)
            table.add_column("音量", style="white", no_wrap=True)
            
            for i, ((start_time, end_time, emotion_scores), (_, _, audio_metrics), (_, _, transcription), (_, _, speaker)) in enumerate(zip(results, metrics_results, transcription_results, speaker_results)):
                dominant_emotion, confidence = max(emotion_scores.items(), key=lambda x: x[1])
                dominant_emotion_jp = self._translate_emotion(dominant_emotion)
                
                table.add_row(
                    f"{start_time:.1f}s",
                    speaker,
                    transcription.get('preview_text', '[空]'),
                    dominant_emotion_jp,
                    f"{confidence:.0%}",
                    f"{audio_metrics.get('pitch_mean', 0):.0f}Hz",
                    f"{audio_metrics.get('rms_mean', 0):.3f}"
                )
        elif metrics_results and transcription_results and not results:
            # Show metrics and transcription table (no emotions)
            table = Table(title="🎵 音声メトリクス・文字起こしタイムライン")
            table.add_column("時間", style="cyan", no_wrap=True)
            table.add_column("テキスト", style="bright_white", no_wrap=True, max_width=30)
            table.add_column("ピッチ", style="yellow", no_wrap=True)
            table.add_column("明度", style="blue", no_wrap=True)
            table.add_column("音量", style="white", no_wrap=True)
            
            for i, ((start_time, end_time, audio_metrics), (_, _, transcription)) in enumerate(zip(metrics_results, transcription_results)):
                table.add_row(
                    f"{start_time:.1f}s",
                    transcription.get('preview_text', '[空]'),
                    f"{audio_metrics.get('pitch_mean', 0):.0f}Hz",
                    f"{audio_metrics.get('brightness_score', 0):.0f}",
                    f"{audio_metrics.get('rms_mean', 0):.3f}"
                )
        elif results and metrics_results and transcription_results:
            # Show combined emotion, audio metrics, and transcription table
            table = Table(title="🎭 感情・音声メトリクス・文字起こしタイムライン")
            table.add_column("時間", style="cyan", no_wrap=True)
            table.add_column("テキスト", style="bright_white", no_wrap=True, max_width=30)
            table.add_column("感情", style="red", no_wrap=True)
            table.add_column("信頼度", style="green", no_wrap=True)
            table.add_column("ピッチ", style="yellow", no_wrap=True)
            table.add_column("明度", style="blue", no_wrap=True)
            table.add_column("音量", style="white", no_wrap=True)
            
            for i, ((start_time, end_time, emotion_scores), (_, _, audio_metrics), (_, _, transcription)) in enumerate(zip(results, metrics_results, transcription_results)):
                dominant_emotion, confidence = max(emotion_scores.items(), key=lambda x: x[1])
                dominant_emotion_jp = self._translate_emotion(dominant_emotion)
                
                table.add_row(
                    f"{start_time:.1f}s",
                    transcription.get('preview_text', '[空]'),
                    dominant_emotion_jp,
                    f"{confidence:.0%}",
                    f"{audio_metrics.get('pitch_mean', 0):.0f}Hz",
                    f"{audio_metrics.get('brightness_score', 0):.0f}",
                    f"{audio_metrics.get('rms_mean', 0):.3f}"
                )
        elif transcription_results and speaker_results and not results:
            # Show transcription and speaker table (no emotions)
            table = Table(title="📝 文字起こし・話者タイムライン")
            table.add_column("時間", style="cyan", no_wrap=True)
            table.add_column("長さ", style="magenta")
            table.add_column("話者", style="bright_blue", no_wrap=True)
            table.add_column("テキスト", style="bright_white", no_wrap=True, max_width=30)
            
            for i, ((start_time, end_time, transcription), (_, _, speaker)) in enumerate(zip(transcription_results, speaker_results)):
                duration = end_time - start_time
                table.add_row(
                    f"{start_time:.1f}s",
                    f"{duration:.1f}s",
                    speaker,
                    transcription.get('preview_text', '[空]')
                )
        elif results and transcription_results and speaker_results:
            # Show emotion, transcription, and speaker table
            table = Table(title="🎭 感情・文字起こし・話者タイムライン")
            table.add_column("時間", style="cyan", no_wrap=True)
            table.add_column("長さ", style="magenta")
            table.add_column("話者", style="bright_blue", no_wrap=True)
            table.add_column("テキスト", style="bright_white", no_wrap=True, max_width=30)
            table.add_column("感情", style="red", no_wrap=True)
            table.add_column("信頼度", style="green")
            
            for i, ((start_time, end_time, emotion_scores), (_, _, transcription), (_, _, speaker)) in enumerate(zip(results, transcription_results, speaker_results)):
                dominant_emotion, confidence = max(emotion_scores.items(), key=lambda x: x[1])
                dominant_emotion_jp = self._translate_emotion(dominant_emotion)
                duration = end_time - start_time
                
                table.add_row(
                    f"{start_time:.1f}s",
                    f"{duration:.1f}s",
                    speaker,
                    transcription.get('preview_text', '[空]'),
                    dominant_emotion_jp,
                    f"{confidence:.1%}"
                )
        elif results and transcription_results:
            # Show emotion and transcription table
            table = Table(title="🎭 感情・文字起こしタイムライン")
            table.add_column("時間", style="cyan", no_wrap=True)
            table.add_column("長さ", style="magenta")
            table.add_column("テキスト", style="bright_white", no_wrap=True, max_width=30)
            table.add_column("感情", style="red", no_wrap=True)
            table.add_column("信頼度", style="green")
            
            for i, ((start_time, end_time, emotion_scores), (_, _, transcription)) in enumerate(zip(results, transcription_results)):
                dominant_emotion, confidence = max(emotion_scores.items(), key=lambda x: x[1])
                dominant_emotion_jp = self._translate_emotion(dominant_emotion)
                duration = end_time - start_time
                
                table.add_row(
                    f"{start_time:.1f}s",
                    f"{duration:.1f}s",
                    transcription.get('preview_text', '[空]'),
                    dominant_emotion_jp,
                    f"{confidence:.1%}"
                )
        elif results and metrics_results:
            # Show combined emotion and audio metrics table
            table = Table(title="🎭 感情・音声メトリクスタイムライン")
            table.add_column("時間", style="cyan", no_wrap=True)
            table.add_column("感情", style="red", no_wrap=True)
            table.add_column("信頼度", style="green", no_wrap=True)
            table.add_column("ピッチ", style="yellow", no_wrap=True)
            table.add_column("明度", style="blue", no_wrap=True)
            table.add_column("清涼度", style="magenta", no_wrap=True)
            table.add_column("音量", style="white", no_wrap=True)
            
            for i, ((start_time, end_time, emotion_scores), (_, _, audio_metrics)) in enumerate(zip(results, metrics_results)):
                dominant_emotion, confidence = max(emotion_scores.items(), key=lambda x: x[1])
                dominant_emotion_jp = self._translate_emotion(dominant_emotion)
                
                table.add_row(
                    f"{start_time:.1f}s",
                    dominant_emotion_jp,
                    f"{confidence:.0%}",
                    f"{audio_metrics.get('pitch_mean', 0):.0f}Hz",
                    f"{audio_metrics.get('brightness_score', 0):.0f}",
                    f"{audio_metrics.get('clarity_score', 0):.1f}",
                    f"{audio_metrics.get('rms_mean', 0):.3f}"
                )
        elif speaker_results:
            # Show speaker-only table
            table = Table(title="👥 話者タイムライン")
            table.add_column("時間", style="cyan", no_wrap=True)
            table.add_column("長さ", style="magenta")
            table.add_column("話者", style="bright_blue", no_wrap=True)
            
            for start_time, end_time, speaker in speaker_results:
                duration = end_time - start_time
                table.add_row(
                    f"{start_time:.1f}s",
                    f"{duration:.1f}s",
                    speaker
                )
        elif transcription_results:
            # Show transcription-only table (no emotions, metrics, or speakers)
            table = Table(title="📝 文字起こしタイムライン")
            table.add_column("時間", style="cyan", no_wrap=True)
            table.add_column("長さ", style="magenta")
            table.add_column("テキスト", style="bright_white", no_wrap=True, max_width=30)
            
            for start_time, end_time, transcription in transcription_results:
                duration = end_time - start_time
                table.add_row(
                    f"{start_time:.1f}s",
                    f"{duration:.1f}s",
                    transcription.get('preview_text', '[空]')
                )
        elif metrics_results:
            # Show metrics-only table
            table = Table(title="🎵 音声メトリクスタイムライン")
            table.add_column("時間", style="cyan", no_wrap=True)
            table.add_column("ピッチ", style="yellow", no_wrap=True)
            table.add_column("明度", style="blue", no_wrap=True)
            table.add_column("清涼度", style="magenta", no_wrap=True)
            table.add_column("音量", style="white", no_wrap=True)
            
            for start_time, end_time, audio_metrics in metrics_results:
                table.add_row(
                    f"{start_time:.1f}s",
                    f"{audio_metrics.get('pitch_mean', 0):.0f}Hz",
                    f"{audio_metrics.get('brightness_score', 0):.0f}",
                    f"{audio_metrics.get('clarity_score', 0):.1f}",
                    f"{audio_metrics.get('rms_mean', 0):.3f}"
                )
        elif results:
            # Show emotion-only table
            table = Table(title="🎭 感情分析タイムライン")
            table.add_column("時間", style="cyan", no_wrap=True)
            table.add_column("長さ", style="magenta")
            table.add_column("感情", style="red", no_wrap=True)
            table.add_column("信頼度", style="green")
            
            for start_time, end_time, emotion_scores in results:
                dominant_emotion, confidence = max(emotion_scores.items(), key=lambda x: x[1])
                dominant_emotion_jp = self._translate_emotion(dominant_emotion)
                duration = end_time - start_time
                
                table.add_row(
                    f"{start_time:.1f}s",
                    f"{duration:.1f}s",
                    dominant_emotion_jp,
                    f"{confidence:.1%}"
                )
        
        self.console.print(table)
        
        # Show summary statistics
        self._show_summary_stats(results, metrics_results, transcription_results, speaker_results)
    
    def _show_detailed_scores(self, emotion_scores: Dict[str, float]):
        """Show detailed emotion scores table."""
        table = Table(title="📊 詳細感情スコア")
        table.add_column("感情", style="cyan")
        table.add_column("スコア", style="green")
        table.add_column("バー", style="blue")
        
        for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
            emotion_jp = self._translate_emotion(emotion)
            bar_length = int(score * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            table.add_row(
                emotion_jp,
                f"{score:.2%}",
                bar
            )
        
        self.console.print(table)
    
    def _show_summary_stats(self, results: List[Tuple[float, float, Dict[str, float]]] = None, 
                          metrics_results: List[Tuple[float, float, Dict[str, float]]] = None,
                          transcription_results: List[Tuple[float, float, Dict[str, str]]] = None,
                          speaker_results: List[Tuple[float, float, str]] = None):
        """Show summary statistics for segment analysis."""
        # Use any available results to calculate basic stats
        all_results = [r for r in [results, metrics_results, transcription_results, speaker_results] if r]
        if not all_results:
            return
        
        # Use the first available result set for timing information
        timing_results = all_results[0]
        
        # Calculate emotion distribution (only if emotion results available)
        emotion_counts = {}
        total_duration = 0
        
        if results:
            for start_time, end_time, emotion_scores in results:
                dominant_emotion, _ = max(emotion_scores.items(), key=lambda x: x[1])
                emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
                total_duration += end_time - start_time
        else:
            # Calculate duration from other result types
            for start_time, end_time, _ in timing_results:
                total_duration += end_time - start_time
        
        # Create summary table
        table = Table(title="📈 統計サマリー")
        table.add_column("項目", style="cyan")
        table.add_column("値", style="green")
        
        table.add_row("総セグメント数", str(len(timing_results)))
        table.add_row("総時間", f"{total_duration:.1f}秒")
        table.add_row("平均セグメント長", f"{total_duration/len(timing_results):.1f}秒")
        
        # Most common emotion (only if emotion analysis was performed)
        if emotion_counts:
            most_common = max(emotion_counts.items(), key=lambda x: x[1])
            most_common_jp = self._translate_emotion(most_common[0])
            table.add_row("最頻感情", f"{most_common_jp} ({most_common[1]} セグメント)")
        
        # Add audio metrics summary if available
        if metrics_results:
            all_pitch = [metrics.get('pitch_mean', 0) for _, _, metrics in metrics_results if metrics.get('pitch_mean', 0) > 0]
            all_brightness = [metrics.get('brightness_score', 0) for _, _, metrics in metrics_results]
            all_clarity = [metrics.get('clarity_score', 0) for _, _, metrics in metrics_results]
            all_volume = [metrics.get('rms_mean', 0) for _, _, metrics in metrics_results]
            
            if all_pitch:
                table.add_row("平均ピッチ", f"{np.mean(all_pitch):.0f} Hz")
            if all_brightness:
                table.add_row("平均明度", f"{np.mean(all_brightness):.1f}")
            if all_clarity:
                table.add_row("平均清涼度", f"{np.mean(all_clarity):.1f}")
            if all_volume:
                table.add_row("平均音量", f"{np.mean(all_volume):.3f}")
        
        # Add transcription summary if available
        if transcription_results:
            languages = [t.get('language', 'unknown') for _, _, t in transcription_results]
            most_common_lang = max(set(languages), key=languages.count) if languages else 'unknown'
            
            # Count segments with successful transcription
            successful_transcriptions = sum(1 for _, _, t in transcription_results if t.get('full_text', '').strip())
            transcription_rate = successful_transcriptions / len(transcription_results) * 100 if transcription_results else 0
            
            table.add_row("検出言語", most_common_lang.upper())
            table.add_row("転写成功率", f"{transcription_rate:.1f}%")
        
        # Add speaker summary if available
        if speaker_results:
            speakers = [speaker for _, _, speaker in speaker_results]
            unique_speakers = list(set(speakers))
            num_speakers = len(unique_speakers)
            
            # Count segments per speaker
            speaker_counts = {}
            for speaker in speakers:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
            most_active_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0] if speaker_counts else "不明"
            
            table.add_row("検出話者数", str(num_speakers))
            table.add_row("最多発話者", f"{most_active_speaker} ({speaker_counts.get(most_active_speaker, 0)} セグメント)")
        
        self.console.print(table)
    
    def show_speaker_summary(self, speaker_results: List[Tuple[float, float, str]]):
        """Show detailed speaker analysis summary."""
        if not speaker_results:
            return
        
        # Calculate speaker statistics
        speaker_stats = {}
        total_duration = 0
        
        for start_time, end_time, speaker in speaker_results:
            duration = end_time - start_time
            total_duration += duration
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "total_time": 0,
                    "segment_count": 0,
                    "segments": []
                }
            
            speaker_stats[speaker]["total_time"] += duration
            speaker_stats[speaker]["segment_count"] += 1
            speaker_stats[speaker]["segments"].append((start_time, end_time))
        
        # Create speaker summary table
        table = Table(title="👥 話者分析サマリー")
        table.add_column("話者", style="bright_blue")
        table.add_column("発話時間", style="green")
        table.add_column("発話割合", style="yellow")
        table.add_column("セグメント数", style="cyan")
        table.add_column("平均長さ", style="magenta")
        
        for speaker, stats in sorted(speaker_stats.items()):
            percentage = (stats["total_time"] / total_duration * 100) if total_duration > 0 else 0
            avg_length = stats["total_time"] / stats["segment_count"] if stats["segment_count"] > 0 else 0
            
            table.add_row(
                speaker,
                f"{stats['total_time']:.1f}秒",
                f"{percentage:.1f}%",
                str(stats["segment_count"]),
                f"{avg_length:.1f}秒"
            )
        
        self.console.print(table)
    
    def show_audio_metrics_details(self, metrics_results: List[Tuple[float, float, Dict[str, float]]]):
        """Show detailed audio metrics analysis."""
        if not metrics_results:
            return
        
        # Pitch analysis table
        pitch_table = Table(title="🎵 ピッチ分析")
        pitch_table.add_column("時間", style="cyan")
        pitch_table.add_column("平均ピッチ", style="yellow")
        pitch_table.add_column("ピッチ範囲", style="magenta")
        pitch_table.add_column("有音率", style="green")
        
        for start_time, end_time, metrics in metrics_results:
            pitch_table.add_row(
                f"{start_time:.1f}s",
                f"{metrics.get('pitch_mean', 0):.0f} Hz",
                f"{metrics.get('pitch_range', 0):.0f} Hz",
                f"{metrics.get('voiced_ratio', 0):.1%}"
            )
        
        self.console.print(pitch_table)
        
        # Spectral analysis table
        spectral_table = Table(title="🌈 スペクトル分析")
        spectral_table.add_column("時間", style="cyan")
        spectral_table.add_column("明度", style="blue")
        spectral_table.add_column("清涼度", style="magenta")
        spectral_table.add_column("帯域幅", style="yellow")
        
        for start_time, end_time, metrics in metrics_results:
            spectral_table.add_row(
                f"{start_time:.1f}s",
                f"{metrics.get('brightness_score', 0):.1f}",
                f"{metrics.get('clarity_score', 0):.1f}",
                f"{metrics.get('spectral_bandwidth_mean', 0):.0f} Hz"
            )
        
        self.console.print(spectral_table)