"""
mlx_lm_replacement.py - Drop-in replacement for mlx_lm using PyTorch

This module provides compatible functions to replace mlx_lm for loading and 
generating text with LLMs on CPU/CUDA devices.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')

class LLMModel:
    """Wrapper for LLM models compatible with mlx_lm interface"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate output from input_ids"""
        with torch.no_grad():
            outputs = self.model.generate(input_ids, **kwargs)
        return outputs
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        self.device = device
        return self


def load(model_id: str, 
         device: str = "cpu",
         quantize: bool = False,
         dtype: torch.dtype = torch.float32) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a pretrained language model and tokenizer
    
    Compatible with mlx_lm.load() interface
    
    Args:
        model_id: HuggingFace model identifier (e.g., "mlx-community/Phi-3-mini-4k-instruct-8bit")
        device: Device to load model on ("cpu" or "cuda")
        quantize: Whether to apply quantization
        dtype: Data type for model (torch.float32, torch.float16, etc.)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"[LLM LOADER] Loading model: {model_id}")
    print(f"[LLM LOADER] Device: {device}, Dtype: {dtype}")
    
    try:
        # Handle model variants
        actual_model_id = model_id
        
        # Map common mlx model IDs to HuggingFace equivalents
        model_mappings = {
            "mlx-community/Phi-3-mini-4k-instruct-8bit": "microsoft/Phi-3-mini-4k-instruct",
            "mlx-community/Meta-Llama-3-8B-Instruct-4bit": "meta-llama/Llama-2-7b-chat-hf",
            "mlx-community/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
        }
        
        if model_id in model_mappings:
            actual_model_id = model_mappings[model_id]
            print(f"[LLM LOADER] Mapped {model_id} -> {actual_model_id}")
        
        # Load tokenizer
        print("[LLM LOADER] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(actual_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        print("[LLM LOADER] Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            actual_model_id,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        model = model.to(device)
        model.eval()
        
        # Apply quantization if requested
        if quantize:
            print("[LLM LOADER] Applying quantization...")
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        
        print(f"[LLM LOADER] ✓ Model loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        print(f"[LLM LOADER] Error loading model: {e}")
        raise


def generate(model: torch.nn.Module,
             tokenizer: AutoTokenizer,
             prompt: str,
             max_tokens: int = 100,
             temperature: float = 0.7,
             top_p: float = 0.9,
             verbose: bool = False,
             **kwargs) -> str:
    """
    Generate text using the model
    
    Compatible with mlx_lm.generate() interface
    
    Args:
        model: PyTorch language model
        tokenizer: Tokenizer for the model
        prompt: Input prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling parameter
        verbose: Whether to print debug info
        **kwargs: Additional generation arguments
    
    Returns:
        Generated text
    """
    device = next(model.parameters()).device
    
    if verbose:
        print(f"[LLM GENERATE] Generating from prompt: '{prompt[:50]}...'")
        print(f"[LLM GENERATE] Device: {device}")
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        if verbose:
            print(f"[LLM GENERATE] Input tokens: {inputs['input_ids'].shape}")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        if verbose:
            print(f"[LLM GENERATE] Generated: '{generated_text[:50]}...'")
        
        return generated_text
    
    except Exception as e:
        print(f"[LLM GENERATE] Error during generation: {e}")
        raise


def stream_generate(model: torch.nn.Module,
                   tokenizer: AutoTokenizer,
                   prompt: str,
                   max_tokens: int = 100,
                   temperature: float = 0.7,
                   verbose: bool = False,
                   **kwargs):
    """
    Generate text with streaming output
    
    Args:
        model: PyTorch language model
        tokenizer: Tokenizer for the model
        prompt: Input prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        verbose: Whether to print debug info
        **kwargs: Additional generation arguments
    
    Yields:
        Generated tokens as strings
    """
    device = next(model.parameters()).device
    
    if verbose:
        print(f"[LLM STREAM] Starting stream generation")
    
    try:
        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
        
        # Generate token by token
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Decode and yield
                token_str = tokenizer.decode(next_token[0])
                yield token_str
                
                # Append to input for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
    
    except Exception as e:
        print(f"[LLM STREAM] Error during stream generation: {e}")
        raise


# Model cache for faster loading
_model_cache = {}

def load_cached(model_id: str, device: str = "cpu") -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load model with caching to avoid reloading
    
    Args:
        model_id: HuggingFace model identifier
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    cache_key = f"{model_id}_{device}"
    
    if cache_key in _model_cache:
        print(f"[LLM CACHE] Loading model from cache: {cache_key}")
        return _model_cache[cache_key]
    
    model, tokenizer = load(model_id, device)
    _model_cache[cache_key] = (model, tokenizer)
    
    return model, tokenizer


def clear_cache():
    """Clear model cache"""
    global _model_cache
    _model_cache.clear()
    print("[LLM CACHE] Cache cleared")
