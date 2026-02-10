"""
Comprehensive test script for MERLINE
Tests all components and provides performance diagnostics
"""

import sys
import os

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports"""
    print("\n[TEST 1] Basic Imports")
    print("-" * 50)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✓ Transformers")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        return False
    
    try:
        from faster_whisper import WhisperModel
        print("✓ Faster Whisper")
    except ImportError as e:
        print(f"✗ Faster Whisper: {e}")
        return False
    
    try:
        from melo.api import TTS
        print("✓ MeloTTS")
    except ImportError as e:
        print(f"✗ MeloTTS: {e}")
        return False
    
    return True


def test_optimization_modules():
    """Test optimization modules"""
    print("\n[TEST 2] Optimization Modules")
    print("-" * 50)
    
    try:
        from optimization import TorchOptimizer, InferenceCache, OptimizedModelWrapper
        print("✓ optimization module")
    except ImportError as e:
        print(f"✗ optimization: {e}")
        return False
    
    try:
        from mlx_lm_replacement import load, generate
        print("✓ mlx_lm_replacement module")
    except ImportError as e:
        print(f"✗ mlx_lm_replacement: {e}")
        return False
    
    try:
        from performance_tuning import SystemAnalyzer, PerformanceTuner, MemoryOptimizer
        print("✓ performance_tuning module")
    except ImportError as e:
        print(f"✗ performance_tuning: {e}")
        return False
    
    return True


def test_system_analysis():
    """Test system analysis"""
    print("\n[TEST 3] System Analysis")
    print("-" * 50)
    
    try:
        from performance_tuning import SystemAnalyzer
        
        info = SystemAnalyzer.get_system_info()
        print(f"✓ CPU Cores: {info['cpu_count']}")
        print(f"✓ RAM: {info['ram_available']:.1f}GB / {info['ram_total']:.1f}GB available")
        print(f"✓ CUDA Available: {info['cuda_available']}")
        
        return True
    except Exception as e:
        print(f"✗ System analysis failed: {e}")
        return False


def test_performance_tuning():
    """Test performance tuning"""
    print("\n[TEST 4] Performance Tuning")
    print("-" * 50)
    
    try:
        from performance_tuning import PerformanceTuner
        
        device = PerformanceTuner.get_optimal_device()
        print(f"✓ Optimal device: {device}")
        
        dtype = PerformanceTuner.get_optimal_dtype(device)
        print(f"✓ Optimal dtype: {dtype}")
        
        batch_size = PerformanceTuner.get_optimal_batch_size(device)
        print(f"✓ Optimal batch size: {batch_size}")
        
        return True
    except Exception as e:
        print(f"✗ Performance tuning failed: {e}")
        return False


def test_cache_system():
    """Test inference cache"""
    print("\n[TEST 5] Inference Cache")
    print("-" * 50)
    
    try:
        import torch
        from optimization import InferenceCache
        
        cache = InferenceCache(max_size=10)
        
        # Test basic operations
        test_tensor = torch.tensor([[1, 2, 3]])
        test_result = "test_result"
        
        cache.set(test_tensor, test_result)
        retrieved = cache.get(test_tensor)
        
        assert retrieved == test_result, "Cache retrieval failed"
        print("✓ Cache set/get operations")
        
        cache.clear()
        assert len(cache) == 0, "Cache clear failed"
        print("✓ Cache clear operation")
        
        return True
    except Exception as e:
        print(f"✗ Cache test failed: {e}")
        return False


def test_main_py_syntax():
    """Test main.py syntax"""
    print("\n[TEST 6] Main.py Syntax")
    print("-" * 50)
    
    try:
        import py_compile
        py_compile.compile('main.py', doraise=True)
        print("✓ main.py syntax is valid")
        return True
    except Exception as e:
        print(f"✗ main.py syntax error: {e}")
        return False


def test_mock_model_loading():
    """Test model loading without actually downloading"""
    print("\n[TEST 7] Model Loading Setup")
    print("-" * 50)
    
    try:
        from mlx_lm_replacement import load
        import inspect
        
        # Test function signature
        sig = inspect.signature(load)
        params = list(sig.parameters.keys())
        
        required_params = ['model_id', 'device', 'quantize', 'dtype']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        
        print("✓ load() function signature is correct")
        print(f"  Parameters: {params}")
        
        return True
    except Exception as e:
        print(f"✗ Model loading setup failed: {e}")
        return False


def test_vad_setup():
    """Test VAD (Voice Activity Detection) setup"""
    print("\n[TEST 8] VAD Setup")
    print("-" * 50)
    
    try:
        from stt.VoiceActivityDetection import VADDetector
        import sounddevice as sd
        
        # Just test that we can import and query devices
        devices = sd.query_devices()
        input_dev = sd.query_devices(kind='input')
        
        print(f"✓ Audio devices detected: {len(devices)} total")
        print(f"✓ Input device: {input_dev['name']}")
        
        return True
    except Exception as e:
        print(f"✗ VAD setup failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("MERLINE COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Optimization Modules", test_optimization_modules),
        ("System Analysis", test_system_analysis),
        ("Performance Tuning", test_performance_tuning),
        ("Cache System", test_cache_system),
        ("Main.py Syntax", test_main_py_syntax),
        ("Model Loading Setup", test_mock_model_loading),
        ("VAD Setup", test_vad_setup),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("-"*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! MERLINE is ready to run.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
    
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
