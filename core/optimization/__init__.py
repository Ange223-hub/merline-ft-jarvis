"""
Optimization module - Performance enhancements for MERLINE
Includes quantization, caching, and mixed precision
"""

from .torch_optimizer import (
    TorchOptimizer,
    OptimizedModelWrapper,
    BatchProcessor,
    optimize_transformers_model,
    InferenceCache,
    get_optimizer
)
from .cache import InferenceCache as CacheSystem
from .mlx_replacement import load as mlx_load, generate as mlx_generate, stream_generate, load_cached, clear_cache
from .performance import (
    SystemAnalyzer,
    PerformanceTuner,
    MemoryOptimizer,
    InferenceProfiler,
    get_profiler,
    setup_merline_performance
)

__all__ = [
    # Torch optimization
    'TorchOptimizer',
    'OptimizedModelWrapper',
    'BatchProcessor',
    'optimize_transformers_model',
    'get_optimizer',
    # Cache
    'InferenceCache',
    'CacheSystem',
    # MLX replacement
    'mlx_load',
    'mlx_generate',
    'stream_generate',
    'load_cached',
    'clear_cache',
    # Performance tuning
    'SystemAnalyzer',
    'PerformanceTuner',
    'MemoryOptimizer',
    'InferenceProfiler',
    'get_profiler',
    'setup_merline_performance',
]

