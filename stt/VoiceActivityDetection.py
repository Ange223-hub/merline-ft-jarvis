import time
import wave
import webrtcvad
import contextlib
import collections
import numpy as np
import sounddevice as sd

RATE = 16000
CHUNK = 160
CHANNELS = 1

# Try to import pyaudio, fallback if not available
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    FORMAT = pyaudio.paInt16
except ImportError:
    PYAUDIO_AVAILABLE = False
    FORMAT = None
    print("[VAD] PyAudio not available, using sounddevice fallback")


class VADDetector:
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
        self.stream = None
        print(f"[VAD] Detector initialized (sensitivity: {sensitivity})")

    def write_wave(self, path, audio, sample_rate):
        with contextlib.closing(wave.open(path, "w")) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframesraw(audio)

    def voice_activity_detection(self, audio_data):
        try:
            return self.vad.is_speech(audio_data, self.sample_rate)
        except Exception as e:
            return False

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            pass  # Ignore status
        
        # Convert numpy array to bytes if needed
        if isinstance(indata, np.ndarray):
            audio_data = indata.astype(np.int16).tobytes()
        else:
            audio_data = indata
        
        detection = self.voice_activity_detection(audio_data)

        if self.frameHistory[-1] == True and detection == True:
            self.onSpeechStart()
            self.voiced_frames.append(audio_data)
            self.block_since_last_spoke = 0
        else:
            if (
                self.block_since_last_spoke
                == self.sensitivity * 10 * self.interval_size
            ):

                if len(self.voiced_frames) > 0:
                    samp = b"".join(self.voiced_frames)
                    self.onSpeechEnd(np.frombuffer(samp, dtype=np.int16))
                self.voiced_frames = []
            else:
                # if last block was not speech don't add
                if len(self.voiced_frames) > 0:
                    self.voiced_frames.append(audio_data)

            self.block_since_last_spoke += 1

        self.frameHistory.append(detection)

    def startListening(self):
        """Start listening using appropriate method"""
        if PYAUDIO_AVAILABLE:
            self._startListeningPyAudio()
        else:
            self._startListeningSoundDevice()
    
    def _startListeningPyAudio(self):
        """Listen using PyAudio (if available)"""
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.audio_callback(data, CHUNK, time.time(), None)
            except Exception as e:
                print(e)
                break
    
    def _startListeningSoundDevice(self):
        """Listen using SoundDevice (fallback)"""
        print("[VAD] Using sounddevice for audio capture...")
        
        with sd.InputStream(
            device=None,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self.audio_callback,
            dtype='int16'
        ) as stream:
            self.stream = stream
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":

    def onSpeechStart():
        print("Speech started")

    def onSpeechEnd(data):
        print("Speech ended")
        print(f"Data {data}")

    vad = VADDetector(onSpeechStart, onSpeechEnd)
    vad.startListening()
