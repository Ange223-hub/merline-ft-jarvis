"""
Optimization module for MERLINE - Advanced PyTorch performance enhancements
Handles quantization, mixed precision, caching, and device management.
"""

import torch
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

class TorchOptimizer:
    """
    Comprehensive PyTorch optimization manager for CPU/CUDA inference
    """
    
    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_cuda = device == "cuda" and torch.cuda.is_available()
        self._setup_device_optimizations()
    
    def _setup_device_optimizations(self):
        """Configure device-specific optimizations"""
        torch.set_grad_enabled(False)  # Disable gradient computation for inference
        
        if self.use_cuda:
            # CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            print("[OPTIMIZER] CUDA optimizations enabled")
        else:
            # CPU optimizations
            torch.set_num_threads(min(torch.get_num_threads(), 8))
            torch.set_flush_denormal(True)
            print("[OPTIMIZER] CPU optimizations enabled")
    
    @staticmethod
    def quantize_dynamic(model: torch.nn.Module, 
                        quantize_layers: Tuple = (torch.nn.Linear,),
                        dtype: torch.qint8 = torch.qint8) -> torch.nn.Module:
        """
        Apply dynamic quantization to reduce model size and improve inference speed
        
        Args:
            model: PyTorch model to quantize
            quantize_layers: Tuple of layer types to quantize
            dtype: Quantization data type (default: int8)
        
        Returns:
            Quantized model
        """
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {layer_type: {
                    'activation': torch.qint8,
                    'weight': dtype
                } for layer_type in quantize_layers},
                dtype=dtype
            )
            print(f"[OPTIMIZER] Dynamic quantization applied (dtype={dtype})")
            return quantized_model
        except Exception as e:
            print(f"[OPTIMIZER] Quantization failed: {e}, returning original model")
            return model
    
    @staticmethod
    def enable_mixed_precision(model: torch.nn.Module) -> torch.nn.Module:
        """
        Enable automatic mixed precision (AMP) for better performance
        
        Args:
            model: PyTorch model
        
        Returns:
            Model with mixed precision enabled
        """
        try:
            model.half()  # Convert to float16
            print("[OPTIMIZER] Mixed precision (FP16) enabled")
            return model
        except Exception as e:
            print(f"[OPTIMIZER] Mixed precision setup failed: {e}")
            return model
    
    @staticmethod
    def enable_gradient_checkpointing(model: torch.nn.Module):
        """
        Enable gradient checkpointing to reduce memory usage
        
        Args:
            model: PyTorch model
        """
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("[OPTIMIZER] Gradient checkpointing enabled")
    
    @staticmethod
    def enable_model_cpu_offload(model: torch.nn.Module):
        """
        Enable CPU offloading for large models on GPU
        
        Args:
            model: PyTorch model
        """
        if hasattr(model, 'enable_model_cpu_offload'):
            model.enable_model_cpu_offload()
            print("[OPTIMIZER] CPU offload enabled")
    
    @staticmethod
    def export_to_torchscript(model: torch.nn.Module, 
                             example_input: torch.Tensor,
                             output_path: str = "model_optimized.pt") -> Optional[torch.jit.ScriptModule]:
        """
        Export model to TorchScript for optimized inference
        
        Args:
            model: PyTorch model
            example_input: Example input tensor for tracing
            output_path: Path to save the TorchScript model
        
        Returns:
            TorchScript module or None if conversion fails
        """
        try:
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(output_path)
            print(f"[OPTIMIZER] TorchScript model exported to {output_path}")
            return traced_model
        except Exception as e:
            print(f"[OPTIMIZER] TorchScript export failed: {e}")
            return None
    
    @staticmethod
    def clear_cache():
        """Clear GPU/CPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[OPTIMIZER] Cache cleared")


class InferenceCache:
    """
    Intelligent caching system for inference results to improve performance
    """
    
    def __init__(self, max_size: int = 128):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
    
    def _hash_input(self, input_ids: torch.Tensor) -> str:
        """Generate hash from input tensor"""
        return hash(input_ids.cpu().numpy().tobytes()).__str__()
    
    def get(self, input_ids: torch.Tensor) -> Optional[Any]:
        """Retrieve cached result"""
        key = self._hash_input(input_ids)
        return self.cache.get(key)
    
    def set(self, input_ids: torch.Tensor, result: Any):
        """Cache result"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (FIFO)
            self.cache.pop(next(iter(self.cache)))
        
        key = self._hash_input(input_ids)
        self.cache[key] = result
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
    
    def __len__(self) -> int:
        return len(self.cache)


class OptimizedModelWrapper:
    """
    Wrapper for models with integrated optimizations
    """
    
    def __init__(self, model: torch.nn.Module, 
                 device: str = "cpu",
                 quantize: bool = True,
                 use_cache: bool = True,
                 mixed_precision: bool = False):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = TorchOptimizer(device)
        self.use_cache = use_cache
        self.cache = InferenceCache() if use_cache else None
        
        # Apply optimizations
        if quantize:
            self.model = self.optimizer.quantize_dynamic(self.model)
        
        if mixed_precision and device == "cpu":
            # Mixed precision is less effective on CPU, but still useful
            pass  # Skip for now, enable only on CUDA
        
        self.model.eval()
        print(f"[MODEL] Optimized model loaded on {device}")
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass with caching"""
        if self.use_cache and 'input_ids' in kwargs:
            cached_result = self.cache.get(kwargs['input_ids'])
            if cached_result is not None:
                print("[CACHE] Cache hit!")
                return cached_result
        
        with torch.no_grad():
            result = self.model(*args, **kwargs)
        
        if self.use_cache and 'input_ids' in kwargs:
            self.cache.set(kwargs['input_ids'], result)
        
        return result
    
    def generate(self, *args, **kwargs) -> Any:
        """Generate with optimizations"""
        with torch.no_grad():
            return self.model.generate(*args, **kwargs)
    
    def clear_cache(self):
        """Clear inference cache"""
        if self.cache:
            self.cache.clear()
            print("[CACHE] Inference cache cleared")
    
    def __getattr__(self, name):
        """Delegate to underlying model"""
        return getattr(self.model, name)


class BatchProcessor:
    """
    Efficient batch processing for multiple inputs
    """
    
    @staticmethod
    def process_batch(model: torch.nn.Module,
                     inputs: list,
                     batch_size: int = 8,
                     device: str = "cpu") -> list:
        """
        Process multiple inputs efficiently in batches
        
        Args:
            model: PyTorch model
            inputs: List of input samples
            batch_size: Batch size for processing
            device: Device to process on
        
        Returns:
            List of outputs
        """
        results = []
        device = torch.device(device)
        
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                batch_results = model(batch)
                results.extend(batch_results)
        
        return results


def optimize_transformers_model(model, device: str = "cpu", 
                                quantize: bool = True) -> torch.nn.Module:
    """
    Optimize a transformers model for inference
    
    Args:
        model: Transformers model
        device: Device to optimize for
        quantize: Whether to apply quantization
    
    Returns:
        Optimized model
    """
    model.to(device)
    model.eval()
    
    # Disable unnecessary features
    if hasattr(model, 'config'):
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = True  # Enable KV cache
    
    # Apply optimizations
    optimizer = TorchOptimizer(device)
    
    if quantize:
        model = optimizer.quantize_dynamic(model)
    
    # Enable gradient checkpointing if available (reduces memory)
    optimizer.enable_gradient_checkpointing(model)
    
    print(f"[OPTIMIZER] Transformers model optimized for {device}")
    return model


# Global optimization context
_global_optimizer = None

def get_optimizer(device: str = "cpu") -> TorchOptimizer:
    """Get global optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = TorchOptimizer(device)
    return _global_optimizer
