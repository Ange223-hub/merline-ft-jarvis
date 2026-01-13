import time
import librosa
import threading
import sounddevice as sd
import numpy as np
import os
from queue import Queue
from playsound import playsound
from melo.api import TTS
from stt.VoiceActivityDetection import VADDetector
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from faster_whisper import WhisperModel
from pydantic import BaseModel

# Augmenter le timeout pour le téléchargement des modèles
os.environ["HF_HUB_READ_TIMEOUT"] = "120"

# Système Prompt original de JARVIS-MLX (adapté pour Merline)
master = "You are a helpful assistant designed to run offline with decent latency, you are open source. Answer the following input from the user in no more than three sentences. Address them as Sir or Stephane at all times. Only respond with the dialogue, nothing else."

class ChatMLMessage(BaseModel):
    role: str
    content: str

class Client:
    def __init__(self, startListening=True, history: list[ChatMLMessage] = []):
        self.greet()
        self.listening = False
        self.history = history
        # VAD threshold and sensitivity
        self.vad = VADDetector(self.onSpeechStart, self.onSpeechEnd, sensitivity=0.4)
        self.vad_data = Queue()
        
        print("\033[33mLoading models (this may take a while, especially on first run)...\033[0m")
        # TTS - English voice
        self.tts = TTS(language="EN_NEWEST", device="cpu") 
        
        # STT - Faster Whisper
        self.stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        
        # LLM - Qwen 2.5 0.5B (Speed Optimized)
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"\033[33mLoading {model_id}...\033[0m")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        print("\033[32mModels loaded successfully!\033[0m")

        # Initial Introduction
        self.speak("Welcome back, Sir. Merline is at your service.")

        if startListening:
            self.toggleListening()
            self.startListening()
            t = threading.Thread(target=self.transcription_loop)
            t.daemon = True
            t.start()

    def greet(self):
        print()
        print("\033[36mWelcome to MERLINE\n\nOptimized for Windows\033[0m")
        print("\n\033[34m--- Audio Devices ---")
        try:
            input_dev = sd.query_devices(kind='input')
            output_dev = sd.query_devices(kind='output')
            print(sd.query_devices())
            print(f"\033[32mCurrent Input: {sd.default.device[0]} ({input_dev['name']})")
            print(f"Current Output: {sd.default.device[1]} ({output_dev['name']})\033[0m")
            
            if "Stereo Mix" in input_dev['name']:
                print("\n\033[31m[WARNING] 'Stereo Mix' is selected as input. \nMerline likely won't hear your voice. Please change your default Windows microphone!\033[0m")
        except Exception as e:
            print(f"Could not list audio devices: {e}")
        print("---------------------\033[0m\n")

    def startListening(self):
        t = threading.Thread(target=self.vad.startListening)
        t.daemon = True
        t.start()

    def toggleListening(self):
        if not self.listening:
            print()
            try:
                playsound("beep.mp3")
            except Exception:
                pass
            print("\033[36mListening...\033[0m")

        while not self.vad_data.empty():
            self.vad_data.get()

        self.listening = not self.listening

    def onSpeechStart(self):
        pass

    def onSpeechEnd(self, data):
        if data.any():
            if len(data) > 8000:
                print(f"\033[34m[DEBUG] Speech detected ({len(data)} samples)\033[0m")
                self.vad_data.put(data)
            else:
                print(f"\033[30m[DEBUG] Speech ignored (too short: {len(data)} samples)\033[0m")

    def addToHistory(self, content: str, role: str):
        if role == "user":
            print(f"\033[32mUser: {content}\033[0m")
        else:
            print(f"\033[33mMerline: {content}\033[0m")

        self.history.append(ChatMLMessage(content=content, role=role))

    def transcription_loop(self):
        while True:
            if not self.vad_data.empty():
                data = self.vad_data.get()
                if self.listening:
                    self.toggleListening()
                    
                    segments, info = self.stt_model.transcribe(data, beam_size=5, language="en")
                    transcribed_text = "".join([segment.text for segment in segments]).strip()
                    
                    if not transcribed_text:
                        self.toggleListening()
                        continue

                    self.addToHistory(transcribed_text, "user")

                    print("\033[35mThinking...\033[0m")
                    prompt = f"<|im_start|>system\n{master}<|im_end|>\n<|im_start|>user\n{transcribed_text}<|im_end|>\n<|im_start|>assistant\n"
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    response = response.strip()
                    
                    self.addToHistory(response, "assistant")
                    self.speak(response)

    def speak(self, text):
        try:
            print("\033[35mSpeaking...\033[0m")
            try:
                speaker_id = self.tts.hps.data.spk2id["EN-Newest"]
            except Exception:
                speaker_id = 0
                
            data = self.tts.tts_to_file(
                text,
                speaker_id,
                speed=1.0,
                quiet=True
            )
            
            if data is not None and len(data) > 0:
                max_amp = np.max(np.abs(data))
                print(f"\033[34m[DEBUG] Audio generated (amplitude max: {max_amp:.4f})\033[0m")
                
                trimmed_audio, _ = librosa.effects.trim(data, top_db=20)
                sd.play(trimmed_audio, 44100)
                sd.wait() # Force wait for completion
                time.sleep(0.5)
            else:
                print("\033[31m[DEBUG] Audio data is empty!\033[0m")
        except Exception as e:
            print(f"\033[31mSpeech Error: {e}\033[0m")

        if self.listening:
            self.toggleListening()

if __name__ == "__main__":
    try:
        mc = Client(startListening=True, history=[])
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Merline...")

