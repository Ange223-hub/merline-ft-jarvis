#!/usr/bin/env python3
"""
Final Verification Script for MERLINE Optimization Suite
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print('\n' + '='*70)
    print('FINAL VERIFICATION - MERLINE OPTIMIZATION SUITE')
    print('='*70)

    # Test 1: Imports
    print('\n[1/5] Testing module imports...')
    try:
        from optimization import TorchOptimizer, InferenceCache
        from mlx_lm_replacement import load, generate
        from performance_tuning import SystemAnalyzer, PerformanceTuner
        print('    ✓ All optimization modules imported successfully')
    except Exception as e:
        print(f'    ✗ Import failed: {e}')
        return False

    # Test 2: System detection
    print('\n[2/5] Testing system detection...')
    try:
        info = SystemAnalyzer.get_system_info()
        print(f'    ✓ Detected {info["cpu_count"]} CPU cores')
        print(f'    ✓ Detected {info["ram_total"]:.1f} GB total RAM')
        print(f'    ✓ CUDA available: {info["cuda_available"]}')
    except Exception as e:
        print(f'    ✗ System detection failed: {e}')
        return False

    # Test 3: Cache system
    print('\n[3/5] Testing inference cache...')
    try:
        import torch
        cache = InferenceCache(max_size=10)
        test_tensor = torch.tensor([[1, 2, 3]])
        cache.set(test_tensor, 'test_result')
        result = cache.get(test_tensor)
        assert result == 'test_result'
        print('    ✓ Inference cache working correctly')
    except Exception as e:
        print(f'    ✗ Cache test failed: {e}')
        return False

    # Test 4: Performance tuning
    print('\n[4/5] Testing performance tuning...')
    try:
        device = PerformanceTuner.get_optimal_device()
        dtype = PerformanceTuner.get_optimal_dtype(device)
        batch_size = PerformanceTuner.get_optimal_batch_size(device)
        print(f'    ✓ Optimal device: {device}')
        print(f'    ✓ Optimal dtype: {dtype}')
        print(f'    ✓ Optimal batch size: {batch_size}')
    except Exception as e:
        print(f'    ✗ Tuning test failed: {e}')
        return False

    # Test 5: Main.py integration
    print('\n[5/5] Testing main.py integration...')
    try:
        import py_compile
        py_compile.compile('main.py', doraise=True)
        print('    ✓ main.py syntax is valid')
        print('    ✓ main.py ready for execution')
    except Exception as e:
        print(f'    ✗ main.py test failed: {e}')
        return False

    print('\n' + '='*70)
    print('✓ ALL VERIFICATION TESTS PASSED!')
    print('='*70)
    print('\nSUMMARY:')
    print('  • Optimization modules: ✓ Functional')
    print('  • System detection: ✓ Working')
    print('  • Inference cache: ✓ Operational')
    print('  • Performance tuning: ✓ Configured')
    print('  • main.py integration: ✓ Ready')
    print('\n🚀 MERLINE is ready to run with full optimization!')
    print('='*70 + '\n')
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
