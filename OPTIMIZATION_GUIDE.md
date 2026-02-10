# MERLINE - Optimization Guide

## Overview

MERLINE has been enhanced with comprehensive performance optimizations for PyTorch-based inference on both CPU and GPU systems.

## New Modules

### 1. `optimization.py` - Core Optimization Framework

Provides essential optimization utilities:

- **TorchOptimizer**: Device-specific optimization manager
  - CPU: Thread optimization, denormal handling
  - GPU: cuDNN benchmark mode, cache management
  
- **InferenceCache**: Smart caching for inference results
  - Reduces redundant computations
  - Configurable cache size (default: 128 entries)
  
- **OptimizedModelWrapper**: Integrated model optimization wrapper
  - Automatic quantization
  - Cache management
  - Device management

- **BatchProcessor**: Efficient batch processing
  - Optimal batch size calculation
  - Memory-efficient processing

### 2. `mlx_lm_replacement.py` - Drop-in MLX_LM Replacement

Replaces the missing `mlx_lm` library with PyTorch equivalents:

- **load()**: Load pretrained models from HuggingFace
  - Compatible with mlx_lm interface
  - Automatic model ID mapping
  - Quantization support

- **generate()**: Text generation with mlx_lm compatibility
  - Temperature sampling
  - Top-p (nucleus) sampling
  - Verbose output option

- **stream_generate()**: Streaming generation for real-time output

- **Model Caching**: Avoid redundant downloads

### 3. `performance_tuning.py` - Advanced Performance Analysis

Provides system-aware optimization:

- **SystemAnalyzer**: System capability detection
  - CPU/GPU information
  - RAM analysis
  - CUDA capability detection

- **PerformanceTuner**: Automatic optimization configuration
  - Optimal device selection
  - Precision optimization (FP32 vs FP16)
  - Thread count optimization
  - Batch size recommendations

- **MemoryOptimizer**: Memory management
  - Model size estimation
  - Inference memory requirements
  - Memory availability checking

- **InferenceProfiler**: Performance profiling
  - Execution time tracking
  - Statistical analysis

## Usage Examples

### Basic Optimization Setup

```python
from optimization import TorchOptimizer
from performance_tuning import PerformanceTuner

# Get optimal settings
settings = PerformanceTuner.configure_optimal_settings()
print(f"Device: {settings['device']}")
print(f"Dtype: {settings['dtype']}")
print(f"Batch size: {settings['batch_size']}")
```

### Loading Models with Optimization

```python
from mlx_lm_replacement import load, generate

# Load model with quantization
model, tokenizer = load(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device="cpu",
    quantize=True
)

# Generate text
response = generate(
    model, tokenizer,
    prompt="What is AI?",
    max_tokens=100,
    temperature=0.7
)
print(response)
```

### Using Inference Cache

```python
from optimization import InferenceCache
import torch

# Create cache
cache = InferenceCache(max_size=64)

# Store result
cache.set(input_tensor, result)

# Retrieve result
cached_result = cache.get(input_tensor)
```

### System Analysis

```python
from performance_tuning import SystemAnalyzer

# Print system information
SystemAnalyzer.print_system_info()

# Get detailed info as dict
info = SystemAnalyzer.get_system_info()
print(f"Available RAM: {info['ram_available']} GB")
print(f"CPU cores: {info['cpu_count']}")
```

## Performance Improvements

### Memory Optimization
- **Dynamic Quantization**: Reduces model size by ~75% (FP32 → INT8)
- **Inference Caching**: Eliminates redundant computations
- **Gradient Checkpointing**: Reduces memory usage during inference

### Speed Optimization
- **CPU Threading**: Optimized for multi-core utilization
- **CUDA Optimization**: cuDNN benchmark and memory management
- **Batch Processing**: Efficient large-scale inference
- **Mixed Precision**: FP16 on CUDA, FP32 on CPU

### Stability
- **Automatic Fallbacks**: Graceful degradation on errors
- **Memory Checks**: Validates sufficient memory before operations
- **Error Handling**: Comprehensive exception handling

## Compatibility

### Device Support
- ✓ CPU (all systems)
- ✓ NVIDIA CUDA (with cuDNN)
- ✓ Apple Silicon (with MLX library)
- ✓ Intel Arc GPU (experimental)

### Model Support
The `mlx_lm_replacement` supports all HuggingFace models:
- Phi-3 series
- Qwen series
- Llama series
- Mistral series
- And many more...

## Testing

Run the comprehensive test suite:

```bash
python test_merline_full.py
```

This will verify:
- All module imports
- System detection
- Performance tuning
- Cache operations
- Model loading setup
- Audio device configuration

## Performance Metrics (Example)

On a typical CPU system (12 cores, 8GB RAM):
- **Model Loading**: ~30-60 seconds (first run)
- **Inference Time**: 500-1000ms per response (Qwen 0.5B)
- **Memory Usage**: ~2-3 GB for inference
- **Cache Hit Rate**: 20-40% for repetitive queries

## Troubleshooting

### Model Loading Fails
- Check available disk space
- Verify internet connection
- Try `quantize=False` for fallback

### Out of Memory Errors
- Reduce batch size
- Enable quantization
- Close other applications
- Use streaming generation

### Slow Performance
- Check CPU usage (should be >70% utilization)
- Verify thread count optimization
- Enable CUDA if available
- Use inference caching

## Future Optimizations

- [ ] ONNX Runtime support
- [ ] TensorRT integration (NVIDIA)
- [ ] ROCm support (AMD GPU)
- [ ] Quantized model weights
- [ ] KV cache optimization
- [ ] Flash Attention support

## References

- [PyTorch Optimization Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [CUDA Performance Tuning](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [TorchScript Export](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
