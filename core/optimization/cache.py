"""
Caching system for MERLINE inference
"""

from typing import Optional, Any, Dict
import torch

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
