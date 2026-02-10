"""
Whisper wrapper for MERLINE
Provides compatibility between different Whisper implementations
"""

import warnings
warnings.filterwarnings('ignore')

class WhisperModel:
    """Wrapper for Whisper models - uses openai-whisper (lightweight stable implementation)"""
    
    def __init__(self, model_name: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize Whisper model
        
        Args:
            model_name: Model name (e.g., "base.en", "tiny", "small")
            device: Device to load on ("cpu" or "cuda")
            compute_type: Compute type (e.g., "int8", "float32")
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        
        try:
            # Use openai-whisper (stable, no Rust dependencies)
            import whisper
            # Map device names: "cpu" -> "cpu", "cuda" -> "cuda"
            device_map = device if device in ["cpu", "cuda"] else "cpu"
            self.model = whisper.load_model(model_name, device=device_map)
            self.is_faster_whisper = False
            print(f"[WHISPER] Using openai-whisper with {self.model_name}")
        except Exception as e:
            print(f"[WHISPER] ERROR: Failed to load openai-whisper: {e}")
            print(f"\033[31m[WHISPER] SOLUTION: Please run 'install_dependencies.bat' to install missing dependencies.\033[0m")
            print(f"\033[33m[WHISPER] Or manually run: pip install openai-whisper\033[0m")
            raise RuntimeError(f"Whisper model failed to load: {e}\nPlease install dependencies by running 'install_dependencies.bat'")
    
    def transcribe(self, audio, language: str = "en", **kwargs):
        """
        Transcribe audio
        
        Args:
            audio: Audio data or file path
            language: Language code
            **kwargs: Additional arguments
        
        Returns:
            Transcription result
        """
        try:
            if self.is_faster_whisper:
                # faster-whisper returns segments
                segments, info = self.model.transcribe(
                    audio,
                    language=language,
                    **kwargs
                )
                # Combine segments into full text
                text = "".join([segment.text for segment in segments])
                return {
                    "text": text,
                    "language": info.language if hasattr(info, 'language') else language,
                }
            else:
                # openai-whisper returns dict with 'text' key
                result = self.model.transcribe(
                    audio,
                    language=language,
                    **kwargs
                )
                return result
        except Exception as e:
            print(f"[WHISPER] Transcription error: {e}")
            raise
