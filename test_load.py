import torch
import transformers
from faster_whisper import WhisperModel
import sounddevice as sd

print("--- Test de Merline ---")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")

try:
    print("Test de chargement de Faster-Whisper (cpu, int8)...")
    # Utilise un modèle minuscule pour le test rapide
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("Succès: Faster-Whisper chargé.")
except Exception as e:
    print(f"Erreur Faster-Whisper: {e}")

try:
    print("Vérification des périphériques audio...")
    devices = sd.query_devices()
    print(f"Nombre de périphériques détectés: {len(devices)}")
except Exception as e:
    print(f"Erreur Audio: {e}")

print("--- Fin du test ---")
