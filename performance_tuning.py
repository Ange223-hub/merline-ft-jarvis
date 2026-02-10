"""
Advanced performance tuning for MERLINE
Provides adaptive optimization based on system capabilities
"""

import torch
import os
import sys
import time
from typing import Optional, Dict, Any
import psutil

class SystemAnalyzer:
    """Analyzes system capabilities for optimal configuration"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get detailed system information"""
        info = {
            'cpu_count': os.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_total': psutil.virtual_memory().total / (1024**3),  # GB
            'ram_available': psutil.virtual_memory().available / (1024**3),  # GB
            'ram_percent': psutil.virtual_memory().percent,
            'disk_total': psutil.disk_usage('/').total / (1024**3),  # GB
            'disk_free': psutil.disk_usage('/').free / (1024**3),  # GB
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'torch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            info['cuda_current_device'] = torch.cuda.current_device()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['cuda_device_capability'] = torch.cuda.get_device_capability(0)
            info['cuda_total_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    @staticmethod
    def print_system_info():
        """Print formatted system information"""
        info = SystemAnalyzer.get_system_info()
        
        print("\n" + "="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        print(f"CPU Cores: {info['cpu_count']}")
        print(f"CPU Usage: {info['cpu_percent']:.1f}%")
        print(f"RAM Total: {info['ram_total']:.1f} GB")
        print(f"RAM Available: {info['ram_available']:.1f} GB")
        print(f"RAM Usage: {info['ram_percent']:.1f}%")
        print(f"Disk Free: {info['disk_free']:.1f} GB / {info['disk_total']:.1f} GB")
        print(f"PyTorch Version: {info['torch_version']}")
        print(f"CUDA Available: {info['cuda_available']}")
        
        if info['cuda_available']:
            print(f"CUDA Device Count: {info['cuda_device_count']}")
            print(f"CUDA Device: {info['cuda_device_name']}")
            print(f"CUDA Compute Capability: {info['cuda_device_capability']}")
            print(f"CUDA Total Memory: {info['cuda_total_memory']:.1f} GB")
        
        print("="*60 + "\n")
        
        return info


class PerformanceTuner:
    """Automatically tunes performance based on system capabilities"""
    
    @staticmethod
    def get_optimal_device() -> str:
        """Determine optimal device (cuda or cpu)"""
        if torch.cuda.is_available():
            print("[TUNER] CUDA device detected, using GPU for inference")
            return "cuda"
        else:
            print("[TUNER] CUDA not available, using CPU")
            return "cpu"
    
    @staticmethod
    def get_optimal_dtype(device: str) -> torch.dtype:
        """Get optimal data type based on device"""
        if device == "cuda" and torch.cuda.is_available():
            # Use float16 (half precision) on CUDA for speed
            print("[TUNER] Using float16 (half precision) on CUDA")
            return torch.float16
        else:
            # Use float32 (full precision) on CPU for accuracy
            print("[TUNER] Using float32 (full precision) on CPU")
            return torch.float32
    
    @staticmethod
    def get_optimal_batch_size(device: str) -> int:
        """Determine optimal batch size based on available memory"""
        if device == "cuda":
            cuda_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if cuda_memory_gb < 4:
                return 4
            elif cuda_memory_gb < 8:
                return 8
            else:
                return 16
        else:
            ram_available_gb = psutil.virtual_memory().available / (1024**3)
            if ram_available_gb < 4:
                return 2
            elif ram_available_gb < 8:
                return 4
            else:
                return 8
    
    @staticmethod
    def get_optimal_num_threads(device: str) -> int:
        """Determine optimal number of threads"""
        if device == "cuda":
            return 1  # CUDA handles threading
        else:
            # Use half the available cores for CPU
            cpu_count = os.cpu_count() or 4
            optimal = max(2, cpu_count // 2)
            print(f"[TUNER] Optimal CPU threads: {optimal}")
            return optimal
    
    @staticmethod
    def configure_optimal_settings(device: Optional[str] = None):
        """Apply optimal settings to PyTorch"""
        if device is None:
            device = PerformanceTuner.get_optimal_device()
        
        dtype = PerformanceTuner.get_optimal_dtype(device)
        num_threads = PerformanceTuner.get_optimal_num_threads(device)
        batch_size = PerformanceTuner.get_optimal_batch_size(device)
        
        # Apply settings
        if device == "cpu":
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(1)
            
            # Enable optimizations
            torch.set_float32_matmul_precision('high')
            torch.set_flush_denormal(True)
            
            print(f"[TUNER] CPU optimizations enabled:")
            print(f"  - Threads: {num_threads}")
            print(f"  - Interop threads: 1")
        
        elif device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            
            print("[TUNER] CUDA optimizations enabled:")
            print(f"  - cudnn.benchmark: True")
            print(f"  - Deterministic: False")
        
        return {
            'device': device,
            'dtype': dtype,
            'num_threads': num_threads,
            'batch_size': batch_size,
        }


class MemoryOptimizer:
    """Optimizes memory usage during inference"""
    
    @staticmethod
    def enable_memory_efficient_attention(model: torch.nn.Module):
        """Enable memory-efficient attention computation"""
        try:
            # This only works on newer PyTorch versions
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                model.config.use_sdpa = True
                print("[MEMORY] Scaled dot-product attention enabled")
        except Exception as e:
            pass
    
    @staticmethod
    def estimate_model_size(model: torch.nn.Module) -> float:
        """Estimate model size in MB"""
        param_count = sum(p.numel() for p in model.parameters())
        buffer_count = sum(b.numel() for b in model.buffers())
        total_elements = param_count + buffer_count
        
        # Assuming float32 = 4 bytes per element
        size_mb = (total_elements * 4) / (1024 * 1024)
        return size_mb
    
    @staticmethod
    def estimate_inference_memory(model: torch.nn.Module, 
                                 input_size: int = 2048,
                                 device: str = "cpu") -> float:
        """Estimate memory needed for inference"""
        model_size = MemoryOptimizer.estimate_model_size(model)
        
        # Rough estimate: model size + input + output buffers
        # Input: 4 bytes * input_size (tokens)
        # Output: 4 bytes * vocab_size (usually 50k for LLMs)
        input_memory = (4 * input_size) / (1024 * 1024)
        output_memory = (4 * 50000) / (1024 * 1024)
        
        total_mb = model_size + input_memory + output_memory
        return total_mb
    
    @staticmethod
    def check_memory_availability(required_mb: float, device: str = "cpu") -> bool:
        """Check if sufficient memory is available"""
        if device == "cuda":
            available_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        else:
            available_mb = psutil.virtual_memory().available / (1024**2)
        
        has_enough = available_mb > (required_mb * 1.2)  # 20% buffer
        
        print(f"[MEMORY] Required: {required_mb:.1f} MB")
        print(f"[MEMORY] Available: {available_mb:.1f} MB")
        print(f"[MEMORY] Sufficient: {'Yes' if has_enough else 'No'}")
        
        return has_enough


class InferenceProfiler:
    """Profiles inference performance"""
    
    def __init__(self):
        self.timings = {}
    
    def profile(self, name: str, func, *args, **kwargs):
        """Profile a function's execution time"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        if name not in self.timings:
            self.timings[name] = []
        
        self.timings[name].append(elapsed)
        
        return result, elapsed
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get profiling statistics for a function"""
        if name not in self.timings:
            return {}
        
        times = self.timings[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
        }
    
    def print_report(self):
        """Print profiling report"""
        print("\n" + "="*60)
        print("INFERENCE PROFILING REPORT")
        print("="*60)
        
        for name in self.timings:
            stats = self.get_stats(name)
            print(f"\n{name}:")
            print(f"  Calls: {stats['count']}")
            print(f"  Total: {stats['total']:.3f}s")
            print(f"  Mean:  {stats['mean']:.3f}s")
            print(f"  Min:   {stats['min']:.3f}s")
            print(f"  Max:   {stats['max']:.3f}s")
        
        print("\n" + "="*60 + "\n")


# Global profiler instance
_global_profiler = InferenceProfiler()

def get_profiler() -> InferenceProfiler:
    """Get global profiler instance"""
    return _global_profiler


def setup_merline_performance():
    """Complete setup for MERLINE performance optimization"""
    print("\n" + "="*60)
    print("MERLINE PERFORMANCE SETUP")
    print("="*60)
    
    # Analyze system
    SystemAnalyzer.print_system_info()
    
    # Configure optimal settings
    settings = PerformanceTuner.configure_optimal_settings()
    
    print("\nOptimal Configuration:")
    print(f"  Device: {settings['device']}")
    print(f"  Data Type: {settings['dtype']}")
    print(f"  Batch Size: {settings['batch_size']}")
    
    print("\n" + "="*60 + "\n")
    
    return settings
