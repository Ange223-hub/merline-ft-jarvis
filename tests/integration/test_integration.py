"""
Integration test for MERLINE optimization
Tests all core components working together
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("[TEST] Testing module imports...")
    try:
        from core.optimization import (
            TorchOptimizer,
            InferenceCache,
            mlx_load,
            mlx_generate,
            SystemAnalyzer,
            PerformanceTuner,
            MemoryOptimizer,
        )
        print("✓ All optimization modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_optimizer():
    """Test TorchOptimizer"""
    print("[TEST] Testing TorchOptimizer...")
    try:
        from core.optimization import TorchOptimizer
        optimizer = TorchOptimizer(device="cpu")
        print("✓ TorchOptimizer initialized")
        return True
    except Exception as e:
        print(f"✗ TorchOptimizer failed: {e}")
        return False

def test_cache():
    """Test InferenceCache"""
    print("[TEST] Testing InferenceCache...")
    try:
        from core.optimization import InferenceCache
        import torch
        cache = InferenceCache(max_size=10)
        test_input = torch.randn(1, 5)
        cache.set(test_input, "cached_result")
        assert cache.get(test_input) == "cached_result"
        print("✓ InferenceCache working correctly")
        return True
    except Exception as e:
        print(f"✗ InferenceCache failed: {e}")
        return False

def test_system_analyzer():
    """Test SystemAnalyzer"""
    print("[TEST] Testing SystemAnalyzer...")
    try:
        from core.optimization import SystemAnalyzer
        info = SystemAnalyzer.get_system_info()
        assert 'cpu_count' in info
        assert 'ram_total' in info
        assert 'torch_version' in info
        print(f"✓ SystemAnalyzer working (CPU cores: {info['cpu_count']}, RAM: {info['ram_total']:.1f}GB)")
        return True
    except Exception as e:
        print(f"✗ SystemAnalyzer failed: {e}")
        return False

def test_performance_tuner():
    """Test PerformanceTuner"""
    print("[TEST] Testing PerformanceTuner...")
    try:
        from core.optimization import PerformanceTuner
        device = PerformanceTuner.get_optimal_device()
        dtype = PerformanceTuner.get_optimal_dtype(device)
        batch_size = PerformanceTuner.get_optimal_batch_size(device)
        print(f"✓ PerformanceTuner working (Device: {device}, Batch: {batch_size})")
        return True
    except Exception as e:
        print(f"✗ PerformanceTuner failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("MERLINE INTEGRATION TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_imports,
        test_optimizer,
        test_cache,
        test_system_analyzer,
        test_performance_tuner,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print("="*60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
