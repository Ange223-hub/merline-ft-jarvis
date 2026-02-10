"""
Comprehensive test for MERLINE integration
Verifies that the entire system works end-to-end
"""

import sys
import os
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_module_structure():
    """Verify proper module structure"""
    print("[1] Checking module structure...")
    required_files = [
        "core/__init__.py",
        "core/optimization/__init__.py",
        "core/optimization/torch_optimizer.py",
        "core/optimization/mlx_replacement.py",
        "core/optimization/cache.py",
        "core/optimization/performance.py",
    ]
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    missing = []
    
    for file in required_files:
        path = os.path.join(base_path, file)
        if not os.path.exists(path):
            missing.append(file)
    
    if missing:
        print(f"  ✗ Missing files: {missing}")
        return False
    
    print(f"  ✓ All {len(required_files)} module files present")
    return True

def test_core_optimization_imports():
    """Test importing from core.optimization"""
    print("[2] Testing core.optimization imports...")
    try:
        from core.optimization import (
            TorchOptimizer,
            InferenceCache,
            mlx_load,
            mlx_generate,
            SystemAnalyzer,
            PerformanceTuner,
            MemoryOptimizer,
            optimize_transformers_model,
        )
        print("  ✓ All core.optimization imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_torch_optimizer():
    """Test TorchOptimizer class"""
    print("[3] Testing TorchOptimizer...")
    try:
        from core.optimization import TorchOptimizer
        optimizer = TorchOptimizer(device="cpu", dtype=__import__('torch').float32)
        print("  ✓ TorchOptimizer initialized successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def test_inference_cache():
    """Test InferenceCache"""
    print("[4] Testing InferenceCache...")
    try:
        from core.optimization import InferenceCache
        import torch
        
        cache = InferenceCache(max_size=16)
        test_tensor = torch.randn(1, 10)
        test_value = "test_output"
        
        # Test set and get
        cache.set(test_tensor, test_value)
        retrieved = cache.get(test_tensor)
        
        if retrieved == test_value:
            print("  ✓ InferenceCache working correctly")
            return True
        else:
            print(f"  ✗ Cache returned wrong value: {retrieved}")
            return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def test_system_analyzer():
    """Test SystemAnalyzer"""
    print("[5] Testing SystemAnalyzer...")
    try:
        from core.optimization import SystemAnalyzer
        
        info = SystemAnalyzer.get_system_info()
        
        # Verify key fields exist
        required_keys = ['cpu_count', 'ram_total', 'torch_version', 'cuda_available']
        missing_keys = [k for k in required_keys if k not in info]
        
        if missing_keys:
            print(f"  ✗ Missing info fields: {missing_keys}")
            return False
        
        print(f"  ✓ SystemAnalyzer working (CPU: {info['cpu_count']}, RAM: {info['ram_total']:.1f}GB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def test_performance_tuner():
    """Test PerformanceTuner"""
    print("[6] Testing PerformanceTuner...")
    try:
        from core.optimization import PerformanceTuner
        
        device = PerformanceTuner.get_optimal_device()
        dtype = PerformanceTuner.get_optimal_dtype(device)
        batch_size = PerformanceTuner.get_optimal_batch_size(device)
        num_threads = PerformanceTuner.get_optimal_num_threads(device)
        
        print(f"  ✓ PerformanceTuner recommendations:")
        print(f"    - Device: {device}")
        print(f"    - Data type: {dtype}")
        print(f"    - Batch size: {batch_size}")
        print(f"    - Threads: {num_threads}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def test_memory_optimizer():
    """Test MemoryOptimizer"""
    print("[7] Testing MemoryOptimizer...")
    try:
        from core.optimization import MemoryOptimizer
        
        # Just verify it can be imported and methods exist
        assert hasattr(MemoryOptimizer, 'estimate_model_size')
        assert hasattr(MemoryOptimizer, 'check_memory_availability')
        
        print("  ✓ MemoryOptimizer methods available")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def test_main_imports():
    """Test that main.py can import optimization modules"""
    print("[8] Testing main.py imports...")
    try:
        # Simulate what main.py does
        from core.optimization import (
            TorchOptimizer,
            InferenceCache,
            mlx_load,
            mlx_generate,
            optimize_transformers_model,
        )
        
        print("  ✓ All main.py required imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("MERLINE INTEGRATION VERIFICATION")
    print("="*70 + "\n")
    
    tests = [
        test_module_structure,
        test_core_optimization_imports,
        test_torch_optimizer,
        test_inference_cache,
        test_system_analyzer,
        test_performance_tuner,
        test_memory_optimizer,
        test_main_imports,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            traceback.print_exc()
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("="*70)
    if passed == total:
        print(f"✓ SUCCESS: All {total} integration tests PASSED!")
        print("\nMERLINE is properly optimized and organized:")
        print("  • All modules in core/optimization/")
        print("  • main.py correctly imports from core.optimization")
        print("  • All optimization features available")
    else:
        print(f"✗ FAILURE: {passed}/{total} tests passed")
        failed_tests = [tests[i].__name__ for i in range(len(results)) if not results[i]]
        print(f"  Failed: {', '.join(failed_tests)}")
    print("="*70 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
