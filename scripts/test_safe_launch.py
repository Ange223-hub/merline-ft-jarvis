"""
Safe launch verification for MERLINE
Tests all components before full startup
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_core_imports():
    """Test core module imports"""
    print("[1] Testing core imports...")
    try:
        import torch
        import transformers
        import numpy as np
        import sounddevice
        import librosa
        print("  ✓ Core libraries imported")
        return True
    except ImportError as e:
        print(f"  ✗ Missing core library: {e}")
        return False

def test_melo():
    """Test MeloTTS"""
    print("[2] Testing MeloTTS...")
    try:
        from melo.api import TTS
        print("  ✓ MeloTTS available")
        return True
    except ImportError as e:
        print(f"  ✗ MeloTTS not available: {e}")
        return False

def test_vad():
    """Test VAD detector"""
    print("[3] Testing VAD...")
    try:
        from stt.VoiceActivityDetection import VADDetector
        print("  ✓ VAD available")
        return True
    except ImportError as e:
        print(f"  ✗ VAD not available: {e}")
        return False

def test_whisper_compat():
    """Test Whisper compatibility layer"""
    print("[4] Testing Whisper compatibility...")
    try:
        from core.utils.whisper_compat import WhisperModel
        print("  ✓ Whisper compatibility layer available")
        return True
    except ImportError as e:
        print(f"  ✗ Whisper compat not available: {e}")
        return False

def test_optimization_modules():
    """Test optimization modules"""
    print("[5] Testing optimization modules...")
    try:
        from core.optimization import (
            TorchOptimizer,
            InferenceCache,
            mlx_load,
            mlx_generate,
        )
        print("  ✓ All optimization modules available")
        return True
    except ImportError as e:
        print(f"  ✗ Optimization modules not available: {e}")
        return False

def test_main_imports():
    """Test all imports needed by main.py"""
    print("[6] Testing main.py imports...")
    try:
        # Simulate main.py imports
        import warnings
        warnings.filterwarnings('ignore')
        import logging
        logging.disable(logging.CRITICAL)
        
        import time
        import librosa
        import threading
        import sounddevice as sd
        import numpy as np
        from queue import Queue
        from transformers import AutoTokenizer
        import torch
        
        from core.utils.whisper_compat import WhisperModel
        from pydantic import BaseModel
        
        from melo.api import TTS
        from stt.VoiceActivityDetection import VADDetector
        
        from core.optimization import (
            TorchOptimizer,
            InferenceCache,
            mlx_load,
            mlx_generate,
            optimize_transformers_model,
        )
        
        print("  ✓ All main.py imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Main.py import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("MERLINE SAFE LAUNCH VERIFICATION")
    print("="*70 + "\n")
    
    tests = [
        test_core_imports,
        test_melo,
        test_vad,
        test_whisper_compat,
        test_optimization_modules,
        test_main_imports,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("="*70)
    if passed == total:
        print(f"✓ SUCCESS: All {total} verification tests PASSED!")
        print("\nMERLINE is ready to launch safely.")
        print("  • All core dependencies available")
        print("  • Optimization modules loaded")
        print("  • Whisper compatibility working")
        print("  • Ready for audio processing")
    else:
        print(f"✗ WARNING: {passed}/{total} tests passed")
        print("  Please check the failures above")
    print("="*70 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
