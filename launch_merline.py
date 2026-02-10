#!/usr/bin/env python3
"""
Script de lancement simple pour MERLINE
"""
import os
import sys
import time

# Désactiver numba JIT
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['HF_HUB_READ_TIMEOUT'] = '120'

# Ajouter le répertoire courant au chemin Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Lancement de MERLINE...")
    print("1. Importation des modules...", flush=True)
    try:
        from main import Client
        print("2. ✓ Tous les imports sont chargés avec succès!", flush=True)
        print("3. Initialisation du client MERLINE...\n", flush=True)
        mc = Client(startListening=True, history=[])
        
        print("4. MERLINE est prêt!", flush=True)
        # Boucle principale
        while True:
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nArrêt de MERLINE...")
                break
    except Exception as e:
        print(f"✗ Erreur: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
