"""
VAD compatibility layer for MERLINE
Works with or without pyaudio
"""

import sounddevice as sd
import webrtcvad
import collections
import numpy as np
import time

RATE = 16000
CHUNK = 160

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    FORMAT = pyaudio.paInt16
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[VAD] PyAudio not available, using sounddevice fallback")
    FORMAT = None

class VADDetector:
    """Voice Activity Detector compatible with or without PyAudio"""
    
    def __init__(self, onSpeechStart, onSpeechEnd, sensitivity=0.4):
        self.channels = [1]
        self.mapping = [c - 1 for c in self.channels]
        self.device_info = sd.query_devices(None, "input")
        self.sample_rate = 16000
        self.interval_size = 10  # audio interval size in ms
        self.sensitivity = sensitivity
        self.block_size = int(self.sample_rate * self.interval_size / 1000)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)
        self.frameHistory = [False]
        self.block_since_last_spoke = 0
        self.onSpeechStart = onSpeechStart
        self.onSpeechEnd = onSpeechEnd
        self.voiced_frames = collections.deque(maxlen=1000)
        
        print(f"[VAD] Detector initialized (sensitivity: {sensitivity})")
    
    def write_wave(self, path, audio, sample_rate):
        """Write audio to WAV file"""
        import wave
        import contextlib
        
        with contextlib.closing(wave.open(path, "w")) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframesraw(audio)
    
    def voice_activity_detection(self, audio_data):
        """Detect voice activity in audio data"""
        try:
            return self.vad.is_speech(audio_data, self.sample_rate)
        except Exception as e:
            print(f"[VAD] Detection error: {e}")
            return False
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"[VAD] Audio stream status: {status}")
        
        audio_data = indata.tobytes() if isinstance(indata, np.ndarray) else indata
        detection = self.voice_activity_detection(audio_data)
        
        if self.frameHistory[-1] and detection:
            self.onSpeechStart()
        elif not detection and self.frameHistory[-1]:
            self.onSpeechEnd()
        
        self.frameHistory.append(detection)
    
    def start(self):
        """Start listening for voice activity"""
        print("[VAD] Starting voice activity detection...")
        self.stream = sd.InputStream(
            device=None,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self.audio_callback,
            dtype='int16'
        )
        self.stream.start()
    
    def stop(self):
        """Stop listening for voice activity"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            print("[VAD] Voice activity detection stopped")
    
    def is_speech(self, audio_data):
        """Check if audio contains speech"""
        try:
            return self.vad.is_speech(audio_data, self.sample_rate)
        except:
            return False
