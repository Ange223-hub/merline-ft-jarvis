#!/usr/bin/env python3
"""
MERLINE - Optimized Launch Script
Lance MERLINE avec des optimisations pour réduire le temps de démarrage
"""

import sys
import os

# Ajouter le répertoire courant au PATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*50)
print("MERLINE v1.0 - Lancement Optimisé")
print("="*50)

# Configuration d'optimisation avant les imports lourds
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print("\n1. Importation des modules principaux...")

try:
    # Désactiver les warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Désactiver les logs
    import logging
    logging.disable(logging.CRITICAL)
    
    print("   ✓ Suppression des warnings et logs")
    
    # Importer les dépendances légères d'abord
    import time
    import threading
    from queue import Queue
    
    print("   ✓ Dépendances légères chargées")
    
    # Importer PyTorch/Transformers (plus lourd)
    print("   • Chargement de PyTorch... (ceci peut prendre quelques secondes)")
    import torch
    import transformers
    from transformers import AutoTokenizer
    
    print("   ✓ PyTorch et Transformers chargés")
    
except ImportError as e:
    print(f"   ✗ Erreur d'importation: {e}")
    sys.exit(1)

print("\n2. Importation des modules MERLINE...")

try:
    from core.optimization import (
        TorchOptimizer,
        InferenceCache,
        mlx_load,
        mlx_generate,
    )
    from core.utils.whisper_compat import WhisperModel
    from stt.VoiceActivityDetection import VADDetector
    from melo.api import TTS
    from pydantic import BaseModel
    
    print("   ✓ Tous les modules MERLINE chargés")
    
except ImportError as e:
    print(f"   ✗ Erreur: {e}")
    sys.exit(1)

print("\n3. Initialisation du client MERLINE...\n")

try:
    # Import du client principal
    from main import Client
    
    # Lancer le client
    mc = Client(startListening=True, history=[])
    
except Exception as e:
    print(f"\n✗ Erreur lors du lancement du client:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n[AIDE] Problèmes courants:")
    print("  - Vérifiez que le microphone est connecté")
    print("  - Vérifiez l'espace disque disponible")
    print("  - Redémarrez Python et réessayez")
    
    sys.exit(1)
