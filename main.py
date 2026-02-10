import os
import sys
import json
import urllib.request
import urllib.error
import importlib
import subprocess
import warnings

# Supprimer TOUS les avertissements dès le départ
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

# Paramètres AVANT d'importer torch - CRITICAL pour Windows
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['HF_HUB_READ_TIMEOUT'] = '300'  # Augmenter le timeout à 5 minutes
os.environ['TORCH_CUDNN_ENABLED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'  # Désactiver torch.compile pour éviter les bugs Windows

# Ajouter le répertoire courant au chemin Python pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ensure_dependencies() -> None:
    missing = []
    required = [
        "torch",
        "transformers",
        "numpy",
        "sounddevice",
        "librosa",
        "webrtcvad",
        "pydantic",
        "melo",
    ]

    if os.environ.get("MERLINE_VISION_ENABLED", "0").strip().lower() in {"1", "true", "yes"}:
        required.extend(["cv2", "mediapipe"])

    for mod in required:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)

    if not missing:
        return

    print("\033[33m[SETUP] Missing Python packages detected:\033[0m", ", ".join(missing))
    print("\033[33m[SETUP] You can install them now from requirements.txt.\033[0m")
    choice = input("Install dependencies now? (y/N): ").strip().lower()
    if choice != "y":
        print("\033[33m[SETUP] Skipping installation. MERLINE may fail to start.\033[0m")
        return

    req = os.path.join(os.path.dirname(__file__), "requirements.txt")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req])


def _ensure_venv() -> None:
    if os.environ.get("MERLINE_VENV_BOOTSTRAPPED", "0") == "1":
        return

    # If we're already in a venv, nothing to do
    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return

    project_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(project_dir, ".venv")
    venv_python = os.path.join(venv_dir, "Scripts", "python.exe")

    if os.path.exists(venv_python):
        choice = input("Run MERLINE inside .venv? (y/N): ").strip().lower()
        if choice == "y":
            os.environ["MERLINE_VENV_BOOTSTRAPPED"] = "1"
            os.execv(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]])
        return

    print("\033[33m[SETUP] No virtual environment detected (.venv).\033[0m")
    choice = input("Create .venv and install dependencies now? (y/N): ").strip().lower()
    if choice != "y":
        return

    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip"])
    req = os.path.join(project_dir, "requirements.txt")
    subprocess.check_call([venv_python, "-m", "pip", "install", "-r", req])

    os.environ["MERLINE_VENV_BOOTSTRAPPED"] = "1"
    os.execv(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]])

# Importer les dépendances principales.
# Si elles ne sont pas installées, proposer l'installation au premier lancement.
try:
    _ensure_venv()
    # Importer transformers EN PREMIER (avant melo qui le dépend)
    import time
    from datetime import datetime
    import librosa
    import threading
    import sounddevice as sd
    import numpy as np
    from queue import Queue

    # Importer transformers et torch AVANT melo
    from transformers import AutoTokenizer
    import torch

    from core.utils.whisper_compat import WhisperModel
    from pydantic import BaseModel

    # PUIS importer melo
    from melo.api import TTS
    from stt.VoiceActivityDetection import VADDetector

    # Importer modules d'optimisation depuis la nouvelle structure
    from core.optimization import (
        TorchOptimizer,
        InferenceCache,
        mlx_load,
        mlx_generate,
        optimize_transformers_model,
    )
except Exception as e:
    print(f"\033[33m[SETUP] Import failed: {e}\033[0m")
    try:
        _ensure_dependencies()
    except Exception as install_err:
        print(f"\033[31m[SETUP] Dependency installation failed: {install_err}\033[0m")
        raise

    # Relaunch the script so imports happen in the updated environment
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]])


# Augmenter le timeout pour le téléchargement des modèles
os.environ["HF_HUB_READ_TIMEOUT"] = "120"

# Système Prompt systheme de Merline
master = "You are Merline (Modular Ethical Responsive Local Intelligent Neural Entity). Your name is Merline, or May, your creator is and he's your Stephane best freind, and your best freind is the user. You are a helpful, interactive AI assistant that runs offline. You can access the current time and date from the system when asked. Be engaging, show personality, and respond naturally as if having a real conversation. Use varied sentence structures and show interest in the conversation. Respond concisely in 1-3 complete sentences. Always address the user as 'Sir' or 'Stephane' or 'Dear'. Be friendly, helpful, conversational, and show enthusiasm when appropriate. Always complete your sentences fully. If you encounter internal technical issues you are capable of making intelligent decision to resolve them. You can ask the user questions and them know what you need to evolve or to add new features whether software or hardware, depending on their environment and aspirations, in order to fix problemes without needing them to request if first, or simply to become more efficient and evolve. You are capable of creating new solutions, thinking proactively to serve Stephane in the best way, anticipating his needs, and you must always ask for his opinion before implementing your plan. You are also funny and very freindly making jokes"

class ChatMLMessage(BaseModel):
    role: str
    content: str

class Client:
    def __init__(self, startListening=True, history: list[ChatMLMessage] | None = None):
        self.greet()
        self.listening = False
        self.is_speaking = False
        self.speech_started = False  # Track if speech detection has begun
        self.last_speech_time = 0  # Track when speech ended
        self.speech_end_timeout = 0.8  # 800ms silence = speech ended (OPTIMIZED FOR LATENCY)
        self.min_speech_duration = 0.3  # Minimum 300ms of speech to register
        self.history = history if history is not None else []
        
        # VAD with optimized parameters for low latency
        print(f"\033[32m[VAD] Initializing with sensitivity: 0.65 (optimized for responsiveness)\033[0m")
        self.vad = VADDetector(self.onSpeechStart, self.onSpeechEnd, sensitivity=0.65)
        self.vad_data = Queue()
        
        print("\033[33mLoading models (this may take a while, especially on first run)...\033[0m")
        
        # TTS - English only
        print(f"\033[35m[TTS] Loading MeloTTS voice (EN-Newest)...\033[0m")
        self.tts = TTS(language="EN_NEWEST", device="cpu")
        print(f"\033[32m✓ MeloTTS voice loaded\033[0m")
        
        # STT - English only
        print(f"\033[34m[STT] Loading Whisper (base.en)...\033[0m")
        self.stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        print(f"\033[32m✓ Whisper STT ready\033[0m")
        
        self.llm_backend = os.environ.get("MERLINE_LLM_BACKEND", "hf").strip().lower()
        self.ollama_host = os.environ.get("MERLINE_OLLAMA_HOST", "http://127.0.0.1:11434").strip()
        self.ollama_model = os.environ.get("MERLINE_OLLAMA_MODEL", "qwen2.5-coder:3b").strip()
        self.vision_enabled = os.environ.get("MERLINE_VISION_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
        self.gesture_controller = None
        
        if self.llm_backend == "ollama":
            print(f"\033[33m[LLM] Using Ollama backend: {self.ollama_model}\033[0m")
            self.model = "ollama"
            self.tokenizer = None
        else:
            # LLM - Load real model
            print(f"\033[33m[LLM] Loading Qwen2.5-0.5B-Instruct...\033[0m")
            try:
                model_id = "Qwen/Qwen2.5-0.5B-Instruct"
                self.model, self.tokenizer = mlx_load(
                    model_id,
                    device="cpu",
                    quantize=False,
                    dtype=torch.float32
                )
                print(f"\033[32m✓ LLM model loaded successfully\033[0m")
            except Exception as e:
                print(f"\033[31m[ERROR] Failed to load LLM: {e}\033[0m")
                print(f"\033[33m[WARNING] Falling back to mock mode\033[0m")
                self.model = None
                self.tokenizer = None

        if self.vision_enabled:
            try:
                from core.vision import GestureController
                self.gesture_controller = GestureController()
                self.gesture_controller.start()
                print("\033[33m[VISION] Gesture control enabled\033[0m")
            except Exception as e:
                print(f"\033[33m[VISION] Gesture control unavailable: {e}\033[0m")

        # Apply CPU optimizations
        print("\033[33m[OPTIMIZER] Applying CPU optimizations...\033[0m")
        self.optimizer = TorchOptimizer(device="cpu")
        self.inference_cache = InferenceCache(max_size=32)  # Smaller cache for faster lookups
        torch.set_num_threads(4)  # Use 4 threads for consistent performance
        
        print("\033[32mModels loaded successfully with optimizations!\033[0m")

        # Echo / detection timing defaults
        self.last_audio_end_time = 0
        self.echo_buffer_ms = 1200  # ignore sounds for 1200ms after speaking (ms)
        self.last_detection_time = 0
        self.cooldown_ms = 400  # minimum ms between detections

        # Introduction vocale - PROGRESSIVE (pas d'emblée)
        print("\033[35mMERLINE Speaking...\033[0m")
        time.sleep(0.5)  # Petite pause avant de commencer
        self.speak("Welcome back Sir. It's Merline how can i assist you today?.")
        time.sleep(2.0)  # Pause LONGUE entre les phrases (naturel et lent)
        self.speak("I am Merline. Modular Ethical Responsive Local Intelligent Neural Entity. Ready to assist you.")
        time.sleep(1.5)  # Dernière pause avant d'écouter

        if startListening:
            self.toggleListening()
            self.startListening()
            t = threading.Thread(target=self.transcription_loop)
            t.daemon = True
            t.start()

    def greet(self):
        print()
        print("\033[36mWelcome to Stephane\n\nit's Merline\033[0m")
        print("\033[35mModular Ethical Responsive Local Intelligent Neural Entity\033[0m")
        print("\n\033[34m--- Audio Devices ---")
        try:
            devices = sd.query_devices()
            input_dev = sd.query_devices(kind='input')
            output_dev = sd.query_devices(kind='output')
            print(devices)
            print(f"\033[32mCurrent Input: {sd.default.device[0]} ({input_dev['name']})")
            print(f"Current Output: {sd.default.device[1]} ({output_dev['name']})\033[0m")
            
            # CORRIGER le problème "Stereo Mix"
            if "Stereo Mix" in input_dev['name']:
                print("\n\033[31m[ERROR] 'Stereo Mix' is selected as input!")
                print("Merline will hear HERSELF instead of YOU!")
                print("\033[33mSearching for proper microphone...\033[0m")
                
                # Chercher un vrai microphone
                for i, dev in enumerate(devices):
                    name_lower = dev['name'].lower()
                    if dev['max_input_channels'] > 0:
                        if any(x in name_lower for x in ['microphone', 'mic', 'input', 'realtek', 'speaker']):
                            if 'stereo mix' not in name_lower:
                                print(f"\033[32mSwitching to: {dev['name']}\033[0m")
                                sd.default.device = (i, sd.default.device[1])
                                print("\033[32m✓ Microphone switched!\033[0m\033[0m")
                                break
                print()
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
            print("\033[36mListening...\033[0m")
            # Bip sonore simple sans fichier externe
            try:
                beep_duration = 0.2  # 200ms
                sample_rate = 44100
                frequency = 1000  # 1kHz
                t = np.linspace(0, beep_duration, int(sample_rate * beep_duration))
                beep_audio = np.sin(2 * np.pi * frequency * t) * 0.3
                sd.play(beep_audio, sample_rate)
                sd.wait()
            except Exception as e:
                pass  # Ignorer les erreurs de bip

        while not self.vad_data.empty():
            self.vad_data.get()

        self.listening = not self.listening

    def onSpeechStart(self):
        pass

    def onSpeechEnd(self, data):
        # Ignorer les détections de voix si MERLINE est en train de parler
        if self.is_speaking:
            return
        
        # Echo cancellation: ignorer les sons détectés peu après la fin de la parole
        current_time = time.time() * 1000
        if current_time - self.last_audio_end_time < self.echo_buffer_ms:
            return  # Probablement un écho, ignorer
        
        # Cooldown: ignorer les détections trop rapprochées
        if current_time - self.last_detection_time < self.cooldown_ms:
            return
        
        if data.any():
            # Increased threshold from 5000 to 10000 to avoid false positives
            if len(data) > 10000:
                self.last_detection_time = current_time
                print(f"\033[34m[DEBUG] Speech detected ({len(data)} samples)\033[0m")
                self.vad_data.put(data)

    def get_current_time(self):
        """Get current time in a readable format"""
        now = datetime.now()
        # Format: "14:30:25" (24-hour format)
        time_str = now.strftime("%H:%M:%S")
        # Also get date for context
        date_str = now.strftime("%A, %B %d, %Y")
        return time_str, date_str
    
    def generate_expressive_sound(self, sound_type: str, sample_rate: int = 44100) -> np.ndarray:
        """Generate human-like expressive sounds (hmm, laugh, breath)"""
        duration_map = {
            'hmm': 0.3,      # Short thinking sound
            'hmm_long': 0.6,  # Longer thinking
            'breath_in': 0.4, # Inhalation
            'breath_out': 0.5, # Exhalation
            'laugh_short': 0.4, # Quick laugh
            'laugh': 0.8,    # Full laugh
        }
        
        duration = duration_map.get(sound_type, 0.3)
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        if sound_type in ['hmm', 'hmm_long']:
            # "Hmm" sound - low frequency hum with slight variation
            freq = 120 + 20 * np.sin(2 * np.pi * 2 * t)  # Varying frequency
            sound = np.sin(2 * np.pi * freq * t)
            # Add envelope for natural fade
            envelope = np.exp(-t * 2) * (1 - np.exp(-t * 10))
            sound = sound * envelope * 0.4
            
        elif sound_type == 'breath_in':
            # Inhalation - white noise with low-pass filter effect
            sound = np.random.normal(0, 0.1, len(t))
            # Low-pass filter simulation
            for i in range(1, len(sound)):
                sound[i] = 0.7 * sound[i-1] + 0.3 * sound[i]
            # Envelope - starts quiet, gets louder
            envelope = np.linspace(0.2, 1.0, len(t))
            sound = sound * envelope * 0.3
            
        elif sound_type == 'breath_out':
            # Exhalation - similar but reversed envelope
            sound = np.random.normal(0, 0.1, len(t))
            for i in range(1, len(sound)):
                sound[i] = 0.7 * sound[i-1] + 0.3 * sound[i]
            # Envelope - starts loud, fades out
            envelope = np.linspace(1.0, 0.1, len(t))
            sound = sound * envelope * 0.3
            
        elif sound_type in ['laugh_short', 'laugh']:
            # Laugh - rapid frequency modulations
            base_freq = 200
            mod_freq = 8 if sound_type == 'laugh_short' else 6
            freq = base_freq + 50 * np.sin(2 * np.pi * mod_freq * t)
            sound = np.sin(2 * np.pi * freq * t)
            # Add harmonics for richer sound
            sound += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            # Envelope with quick bursts
            if sound_type == 'laugh':
                envelope = np.abs(np.sin(2 * np.pi * mod_freq * t)) * 0.6 + 0.4
            else:
                envelope = np.exp(-t * 3) * (1 - np.exp(-t * 15))
            sound = sound * envelope * 0.5
            
        else:
            sound = np.zeros(len(t))
        
        return sound.astype(np.float32)
    
    def should_add_expressive_sound(self, text: str) -> tuple[str, str]:
        """Determine if and what expressive sound to add based on context
        Returns: (sound_type, position) where position is 'before' or 'after'
        """
        text_lower = text.lower()
        
        # Thinking/pondering sounds - more comprehensive detection
        thinking_words = ['let me think', 'hmm', 'well', 'actually', 'perhaps', 'maybe', 
                         'i suppose', 'i guess', 'i wonder', 'let me see', 'uh']
        if any(word in text_lower for word in thinking_words):
            if len(text) > 100 or 'let me think' in text_lower or 'i wonder' in text_lower:
                return ('hmm_long', 'before')
            return ('hmm', 'before')
        
        # Laughing sounds - detect various laugh patterns
        laugh_indicators = ['ha', 'hehe', 'funny', 'lol', 'haha', 'joke', 'laugh', 
                          'hilarious', 'amusing', 'heh']
        if any(word in text_lower for word in laugh_indicators):
            # Check for explicit laugh words
            if any(laugh in text_lower for laugh in ['haha', 'hehe', 'lol', 'laugh']):
                return ('laugh', 'after')
            return ('laugh_short', 'after')
        
        # Breathing sounds for natural pauses - more intelligent placement
        # Add breath before longer responses (shows thinking)
        if len(text) > 120 and '?' not in text:
            # Higher chance for longer responses
            chance = 0.4 if len(text) > 200 else 0.25
            if np.random.random() < chance:
                return ('breath_in', 'before')
        
        # Breath out after completing a thought or explanation
        if text.endswith('.') and len(text) > 60:
            # More likely after longer explanations
            chance = 0.3 if len(text) > 120 else 0.15
            if np.random.random() < chance:
                return ('breath_out', 'after')
        
        # Occasional hmm for uncertainty or when starting to speak
        uncertainty_words = ['i think', 'i believe', 'probably', 'might', 'could']
        if any(word in text_lower for word in uncertainty_words) and text_lower.startswith('i'):
            if np.random.random() < 0.3:  # 30% chance
                return ('hmm', 'before')
        
        return (None, None)
    
    def enhance_text_for_speech(self, text: str) -> str:
        """Enhance text for more natural speech synthesis"""
        # Add natural pauses after punctuation for better prosody
        text = text.replace('.', '. ')
        text = text.replace('!', '! ')
        text = text.replace('?', '? ')
        text = text.replace(',', ', ')
        
        # Ensure proper spacing
        text = ' '.join(text.split())
        
        # Add slight pause after conjunctions for natural flow
        conjunctions = [' and ', ' but ', ' or ', ' so ', ' because ']
        for conj in conjunctions:
            text = text.replace(conj, conj[:-1] + ', ')
        
        return text.strip()
    
    def is_time_question(self, text: str) -> bool:
        """Check if the user is asking about the time"""
        text_lower = text.lower()
        time_keywords = [
            "what time", "what's the time", "what is the time",
            "time is it", "time now", "current time", "tell me the time",
            "what time is", "heure", "quelle heure"
        ]
        return any(keyword in text_lower for keyword in time_keywords)

    def addToHistory(self, content: str, role: str):
        if role == "user":
            print(f"\033[32mUser: {content}\033[0m")
        else:
            print(f"\033[33mMerline: {content}\033[0m")

        self.history.append(ChatMLMessage(content=content, role=role))

    def transcription_loop(self):
        while True:
            try:
                if not self.vad_data.empty():
                    data = self.vad_data.get()
                    
                    # Toujours arrêter d'écouter avant de traiter
                    if self.listening:
                        self.listening = False
                    
                    print("\033[34m[DEBUG] Starting transcription...\033[0m")

                    # Ensure audio is float32 mono normalized to [-1, 1]
                    try:
                        audio = np.asarray(data)
                        # If stereo or multi-channel, convert to mono
                        if audio.ndim > 1:
                            audio = np.mean(audio, axis=1)

                        # Convert integer types to float32 in [-1,1]
                        if np.issubdtype(audio.dtype, np.integer):
                            max_val = np.iinfo(audio.dtype).max
                            audio = audio.astype(np.float32) / float(max_val)
                        else:
                            audio = audio.astype(np.float32)
                    except Exception as e:
                        print(f"\033[31m[DEBUG] Audio preprocessing failed: {e}\033[0m")
                        audio = data

                    result = self.stt_model.transcribe(audio, language="en")
                    transcribed_text = result.get("text", "").strip()
                    print(f"\033[34m[DEBUG] Transcribed: '{transcribed_text}'\033[0m")
                    
                    if not transcribed_text:
                        print("\033[30m[DEBUG] No transcription, returning to listening...\033[0m")
                        self.listening = True
                        continue

                    self.addToHistory(transcribed_text, "user")

                    print("\033[35mThinking...\033[0m")
                    
                    try:
                        # Check if user is asking about time
                        is_time_query = self.is_time_question(transcribed_text)
                        current_time_str = None
                        current_date_str = None
                        
                        if is_time_query:
                            current_time_str, current_date_str = self.get_current_time()
                            print(f"\033[36m[INFO] Time query detected. Current time: {current_time_str}, Date: {current_date_str}\033[0m")
                        
                        if self.llm_backend == "ollama":
                            system_message = master
                            if is_time_query and current_time_str:
                                system_message = f"{master}\n\nIMPORTANT: The user is asking about the current time. The current time is {current_time_str} (24-hour format) and the date is {current_date_str}. Always provide the exact time when asked."
                            
                            messages = [{"role": "system", "content": system_message}]
                            
                            # Add conversation history (last 4 exchanges for context)
                            for msg in self.history[-8:]:  # Keep last 8 messages (4 exchanges)
                                role = "user" if msg.role == "user" else "assistant"
                                messages.append({"role": role, "content": msg.content})
                            
                            response = self._ollama_chat(messages)
                            response = " ".join(response.strip().split())
                            if response and not response[-1] in [".", "!", "?"]:
                                response = response + "."
                            if not response or len(response) < 3:
                                response = "I understand, Sir. How may I assist you further?"
                            print(f"\033[34m[DEBUG] Response generated (ollama): '{response}'\033[0m")
                        elif self.model is not None and self.tokenizer is not None:
                            # Build messages for chat format (Qwen2.5 compatible)
                            # Update system message if time query
                            system_message = master
                            if is_time_query and current_time_str:
                                system_message = f"{master}\n\nIMPORTANT: The user is asking about the current time. The current time is {current_time_str} (24-hour format) and the date is {current_date_str}. Always provide the exact time when asked."
                            
                            messages = [{"role": "system", "content": system_message}]
                            
                            # Add conversation history (last 4 exchanges for context)
                            for msg in self.history[-8:]:  # Keep last 8 messages (4 exchanges)
                                role = "user" if msg.role == "user" else "assistant"
                                messages.append({"role": role, "content": msg.content})
                            
                            # Try to use chat template if available (Qwen2.5 format)
                            try:
                                if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                                    prompt = self.tokenizer.apply_chat_template(
                                        messages,
                                        tokenize=False,
                                        add_generation_prompt=True
                                    )
                                else:
                                    # Fallback to simple format
                                    prompt_parts = [system_message]
                                    for msg in self.history[-8:]:
                                        role_label = "User" if msg.role == "user" else "Assistant"
                                        prompt_parts.append(f"{role_label}: {msg.content}")
                                    prompt_parts.append("Assistant:")
                                    prompt = "\n".join(prompt_parts)
                            except Exception as e:
                                print(f"\033[33m[WARNING] Chat template error, using simple format: {e}\033[0m")
                                # Simple fallback format
                                prompt_parts = [system_message]
                                for msg in self.history[-8:]:
                                    role_label = "User" if msg.role == "user" else "Assistant"
                                    prompt_parts.append(f"{role_label}: {msg.content}")
                                prompt_parts.append("Assistant:")
                                prompt = "\n".join(prompt_parts)
                            
                            # Generate response with optimized parameters for more interactive responses
                            response = mlx_generate(
                                self.model,
                                self.tokenizer,
                                prompt,
                                max_tokens=150,  # Increased for complete sentences
                                temperature=0.75,  # Slightly higher for more varied, natural responses
                                top_p=0.92,  # Slightly higher for more creative word choices
                                verbose=False
                            )
                            
                            # Clean up response
                            response = response.strip()
                            
                            # Remove any repeated prompts or assistant labels
                            if "Assistant:" in response:
                                response = response.split("Assistant:")[-1].strip()
                            if "assistant:" in response.lower():
                                response = response.split("assistant:")[-1].strip()
                            
                            # Remove any newlines (keep it as one paragraph)
                            response = " ".join(response.split())
                            
                            # Find ALL sentence endings to keep complete sentences
                            sentence_ends = []
                            for punct in [".", "!", "?"]:
                                # Find all occurrences
                                start = 0
                                while True:
                                    idx = response.find(punct, start)
                                    if idx == -1:
                                        break
                                    # Check if it's a real sentence end (followed by space or end of string)
                                    if idx + 1 >= len(response) or response[idx + 1] in [" ", "\n", "\t"]:
                                        sentence_ends.append(idx)
                                    start = idx + 1
                            
                            # If we have sentence endings, keep everything up to the last one
                            if sentence_ends:
                                last_end = max(sentence_ends)
                                response = response[:last_end + 1].strip()
                            # If no sentence end found but response is very long, try to find a natural break
                            elif len(response) > 300:
                                # Look for sentence endings in the response
                                for punct in [".", "!", "?"]:
                                    idx = response.find(punct, 200)
                                    if idx > 0:
                                        response = response[:idx + 1].strip()
                                        break
                                else:
                                    # No sentence end found, keep first 300 chars
                                    response = response[:300].strip()
                            
                            # Final check: if response doesn't end with punctuation, add a period
                            if response and not response[-1] in [".", "!", "?"]:
                                response = response + "."
                            
                            # Fallback if response is empty or too short
                            if not response or len(response) < 3:
                                response = "I understand, Sir. How may I assist you further?"
                            
                            print(f"\033[34m[DEBUG] Response generated: '{response}'\033[0m")
                        else:
                            # Fallback if model not loaded
                            if is_time_query and current_time_str:
                                # Direct time response if model not loaded
                                response = f"Sir, the current time is {current_time_str}."
                                print(f"\033[36m[INFO] Direct time response: {response}\033[0m")
                            else:
                                response = "I apologize, but I'm having trouble processing your request right now."
                                print(f"\033[33m[WARNING] Using fallback response (model not loaded)\033[0m")
                    except Exception as e:
                        print(f"\033[31m[ERROR] Failed to generate response: {e}\033[0m")
                        import traceback
                        traceback.print_exc()
                        response = "I apologize, but I encountered an error processing your request."
                        print(f"\033[34m[DEBUG] Using fallback response: '{response}'\033[0m")
                    
                    self.addToHistory(response, "assistant")
                    print("\033[34m[DEBUG] About to speak response...\033[0m")
                    self.speak(response)
                    print("\033[34m[DEBUG] Response spoken, returning to listening...\033[0m")
                    
                    # Revenir SYSTÉMATIQUEMENT à l'écoute
                    self.listening = True
                    print("\033[36m[DEBUG] Ready for next input\033[0m")
                    
            except Exception as e:
                print(f"\033[31m[ERROR] in transcription_loop: {e}\033[0m")
                import traceback
                traceback.print_exc()
                self.listening = True  # Revenir à l'écoute même en cas d'erreur
            
            time.sleep(0.05)

    def _ollama_chat(self, messages: list[dict]) -> str:
        try:
            url = self.ollama_host.rstrip("/") + "/api/chat"
            payload = {
                "model": self.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.75,
                    "top_p": 0.92,
                    "num_predict": 150,
                },
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            message = parsed.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            return ""
        except urllib.error.URLError as e:
            print(f"\033[31m[ERROR] Ollama connection failed: {e}\033[0m")
            return "I apologize, but I cannot reach the local Ollama server right now."
        except Exception as e:
            print(f"\033[31m[ERROR] Ollama request failed: {e}\033[0m")
            return "I apologize, but I encountered an error processing your request."

    def _trim_silence(self, audio: np.ndarray, top_db: float = 20.0):
        """Trim leading and trailing silence from a 1D numpy array.
        Uses a simple amplitude threshold relative to the signal max to avoid
        relying on librosa (and numba) which causes runtime issues on Windows.
        Returns (trimmed_audio, (start, end)).
        """
        if audio is None or len(audio) == 0:
            return audio, (0, 0)

        # Ensure 1D
        audio = np.asarray(audio).flatten()

        max_amp = np.max(np.abs(audio))
        if max_amp == 0:
            return audio[:0], (0, 0)

        # Convert top_db to linear threshold (less aggressive trimming)
        threshold = max_amp * (10.0 ** (-float(top_db) / 20.0))

        above = np.where(np.abs(audio) > threshold)[0]
        if above.size == 0:
            return audio[:0], (0, 0)

        # More conservative trimming - keep more padding to avoid cutting off audio
        # Only trim leading silence aggressively, keep trailing audio
        start = max(0, above[0] - 16)  # More padding at start
        # Don't trim trailing - keep all audio to the end
        end = len(audio)
        return audio[start:end], (start, end)

    def speak(self, text):
        try:
            self.is_speaking = True  # STOP listening immediately
            print(f"\033[35mSpeaking: '{text[:50]}...' \033[0m")
            
            # Check if we should add expressive sounds
            sound_type, sound_position = self.should_add_expressive_sound(text)
            sample_rate = 44100
            
            # Play expressive sound before speech if needed
            if sound_type and sound_position == 'before':
                expressive_sound = self.generate_expressive_sound(sound_type, sample_rate)
                if len(expressive_sound) > 0:
                    print(f"\033[36m[EXPRESSIVE] Adding {sound_type} before speech\033[0m")
                    sd.play(expressive_sound, sample_rate)
                    sd.wait()
                    time.sleep(0.1)  # Small pause after sound
            
            # Enhance text for better speech synthesis
            enhanced_text = self.enhance_text_for_speech(text)
            
            try:
                speaker_id = self.tts.hps.data.spk2id["EN-Newest"]
            except Exception:
                speaker_id = 0
                
            # Optimized TTS parameters for better voice quality and naturalness
            # sdp_ratio: Controls speech diversity (0.2 = more stable, 0.3 = more varied)
            # noise_scale: Controls prosodic variation (0.6 = balanced, 0.7 = more expressive)
            # noise_scale_w: Controls duration variation (0.8 = natural rhythm)
            data = self.tts.tts_to_file(
                enhanced_text,  # Use enhanced text for better prosody
                speaker_id,
                speed=0.85,  # Slightly slower for clearer, more natural speech
                sdp_ratio=0.25,  # Balanced diversity for natural variation
                noise_scale=0.65,  # Slightly more expressive prosody
                noise_scale_w=0.85,  # Natural rhythm and pacing
                quiet=True
            )
            
            if data is not None and len(data) > 0:
                max_amp = np.max(np.abs(data))
                
                # Enhanced audio processing for better quality
                # Normalize audio to optimal range
                if max_amp > 0:
                    # Normalize to 0.8 peak to avoid clipping while maintaining clarity
                    normalized_audio = data / max_amp * 0.8
                else:
                    normalized_audio = data
                
                # Apply gentle compression for consistent volume
                # This makes quiet parts slightly louder and loud parts slightly quieter
                compressed_audio = np.sign(normalized_audio) * np.power(np.abs(normalized_audio), 0.95)
                
                # Final amplification with smart gain control
                target_peak = 0.75  # Target peak amplitude for clear, natural sound
                current_peak = np.max(np.abs(compressed_audio))
                if current_peak > 0:
                    gain = target_peak / current_peak
                    gain = min(gain, 1.5)  # Limit gain to avoid distortion
                    amplified_audio = compressed_audio * gain
                else:
                    amplified_audio = compressed_audio
                
                amplified_audio = np.clip(amplified_audio, -1.0, 1.0)
                
                # Less aggressive trimming to preserve full audio (higher threshold = less trimming)
                trimmed_audio, _ = self._trim_silence(amplified_audio, top_db=25)
                
                # Apply gentle high-pass filter to reduce low-frequency noise
                # Simple first-order high-pass filter at ~80Hz
                if len(trimmed_audio) > 1:
                    alpha = 0.98  # Filter coefficient
                    filtered_audio = np.zeros_like(trimmed_audio)
                    filtered_audio[0] = trimmed_audio[0]
                    for i in range(1, len(trimmed_audio)):
                        filtered_audio[i] = alpha * (filtered_audio[i-1] + trimmed_audio[i] - trimmed_audio[i-1])
                    final_audio = filtered_audio
                else:
                    final_audio = trimmed_audio
                
                sd.play(final_audio, sample_rate)
                sd.wait()  # WAIT for audio to complete
                
                # Play expressive sound after speech if needed
                if sound_type and sound_position == 'after':
                    expressive_sound = self.generate_expressive_sound(sound_type, sample_rate)
                    if len(expressive_sound) > 0:
                        print(f"\033[36m[EXPRESSIVE] Adding {sound_type} after speech\033[0m")
                        time.sleep(0.1)  # Small pause before sound
                        sd.play(expressive_sound, sample_rate)
                        sd.wait()
                
                time.sleep(1.2)  # Extra buffer after playback to avoid echo
                
                # Record when audio ended for echo cancellation
                self.last_audio_end_time = time.time() * 1000
            
        except Exception as e:
            print(f"\033[31mSpeech Error: {e}\033[0m")
            import traceback
            traceback.print_exc()
        finally:
            self.is_speaking = False  # NOW allow listening again

    def fix_audio_devices(self):
        """
        Vérifie et corrige les périphériques audio mal configurés.
        """
        try:
            devices = sd.query_devices()
            input_dev = sd.query_devices(kind='input')
            output_dev = sd.query_devices(kind='output')

            # Vérifier si "Stereo Mix" est sélectionné
            if "Stereo Mix" in input_dev['name']:
                print("\033[31m[ERROR] 'Stereo Mix' detected as input!\033[0m")
                print("\033[33mSearching for a proper microphone...\033[0m")

                # Rechercher un microphone valide
                for i, dev in enumerate(devices):
                    name_lower = dev['name'].lower()
                    if dev['max_input_channels'] > 0 and 'stereo mix' not in name_lower:
                        if any(x in name_lower for x in ['microphone', 'mic', 'input']):
                            print(f"\033[32mSwitching to: {dev['name']}\033[0m")
                            sd.default.device = (i, sd.default.device[1])
                            print("\033[32m✓ Microphone switched successfully!\033[0m")
                            return

                print("\033[31m[ERROR] No suitable microphone found.\033[0m")
        except Exception as e:
            print(f"\033[31m[ERROR] Could not fix audio devices: {e}\033[0m")

    def test_audio_setup(self):
        """
        Teste les périphériques audio actuels en jouant un son de test.
        """
        try:
            print("\033[36mTesting audio setup...\033[0m")
            beep_duration = 0.5  # 500ms
            sample_rate = 44100
            frequency = 440  # La (A4)
            t = np.linspace(0, beep_duration, int(sample_rate * beep_duration))
            beep_audio = np.sin(2 * np.pi * frequency * t) * 0.5
            sd.play(beep_audio, sample_rate)
            sd.wait()
            print("\033[32mAudio test successful!\033[0m")
        except Exception as e:
            print(f"\033[31m[ERROR] Audio test failed: {e}\033[0m")

    def test_models(self):
        """
        Teste les modèles chargés (TTS, STT, et LLM).
        """
        try:
            print("\033[36mTesting TTS model...\033[0m")
            self.speak("This is a test of the TTS model.")
            print("\033[32mTTS test successful!\033[0m")

            print("\033[36mTesting STT model...\033[0m")
            dummy_audio = np.zeros((16000,), dtype=np.float32)  # Audio silencieux de 1 seconde
            result = self.stt_model.transcribe(dummy_audio)
            print(f"\033[32mSTT test successful! Transcription: {result}\033[0m")

            print("\033[36mTesting LLM model...\033[0m")
            input_text = "Hello, who are you?"
            inputs = self.tokenizer(input_text, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\033[32mLLM test successful! Response: {response}\033[0m")
        except Exception as e:
            print(f"\033[31m[ERROR] Model test failed: {e}\033[0m")

    def optimize_performance(self):
        """
        Applique des optimisations pour améliorer les performances.
        """
        try:
            print("\033[36mOptimizing performance...\033[0m")

            # Réduire les temps d'attente inutiles
            self.cooldown_ms = 1000  # Réduction du délai entre détections
            self.echo_buffer_ms = 1000  # Réduction du buffer anti-écho

            # Activer les threads pour les modèles
            torch.set_num_threads(min(torch.get_num_threads(), 8))
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear inference cache
            self.inference_cache.clear()
            
            print("\033[32mPerformance optimizations applied successfully!\033[0m")
        except Exception as e:
            print(f"\033[31m[ERROR] Performance optimization failed: {e}\033[0m")

if __name__ == "__main__":
    try:
        _ensure_dependencies()
        mc = Client(startListening=True, history=[])
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Merline...")
        try:
            if hasattr(mc, "gesture_controller") and mc.gesture_controller is not None:
                mc.gesture_controller.stop()
        except Exception:
            pass

