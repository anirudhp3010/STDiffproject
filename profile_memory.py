#!/usr/bin/env python3
"""
Memory profiling script for STDiff training.
Add this to your training script to track memory usage.
"""

import torch
import gc
from typing import Dict, List

def get_memory_usage(device: torch.device = None) -> Dict[str, float]:
    """
    Get current GPU memory usage in MB.
    
    Args:
        device: CUDA device (default: current device)
        
    Returns:
        Dictionary with allocated, reserved, and free memory in MB
    """
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
    reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    
    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'max_allocated_mb': max_allocated,
        'free_mb': reserved - allocated
    }

def print_memory_usage(label: str = "", device: torch.device = None):
    """Print current memory usage with a label."""
    mem = get_memory_usage(device)
    print(f"{label:30s} | Allocated: {mem['allocated_mb']:7.1f} MB | "
          f"Reserved: {mem['reserved_mb']:7.1f} MB | "
          f"Max: {mem['max_allocated_mb']:7.1f} MB")

def track_tensor_memory(tensors: Dict[str, torch.Tensor], label: str = ""):
    """
    Calculate and print memory usage of specific tensors.
    
    Args:
        tensors: Dictionary of {name: tensor}
        label: Label for this group of tensors
    """
    total = 0
    print(f"\n{label}:")
    print("-" * 60)
    for name, tensor in tensors.items():
        if tensor is not None:
            size_mb = tensor.numel() * tensor.element_size() / 1024**2
            total += size_mb
            print(f"  {name:30s}: {size_mb:7.2f} MB {list(tensor.shape)}")
    print(f"  {'TOTAL':30s}: {total:7.2f} MB")
    print("-" * 60)

def profile_training_step(func):
    """
    Decorator to profile memory usage during a training step.
    
    Usage:
        @profile_training_step
        def training_step(batch):
            ...
    """
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        
        print_memory_usage("Before step")
        
        # Run the function
        result = func(*args, **kwargs)
        
        print_memory_usage("After step")
        print_memory_usage("Peak memory")
        
        return result
    return wrapper

class MemoryTracker:
    """Context manager to track memory usage in a code block."""
    
    def __init__(self, label: str = ""):
        self.label = label
        self.start_mem = None
        self.end_mem = None
        
    def __enter__(self):
        torch.cuda.synchronize()
        gc.collect()
        self.start_mem = get_memory_usage()
        print(f"\n>>> Entering: {self.label}")
        print_memory_usage("  Start", device=torch.cuda.current_device())
        return self
        
    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.end_mem = get_memory_usage()
        print_memory_usage("  End", device=torch.cuda.current_device())
        
        if self.start_mem:
            delta = self.end_mem['allocated_mb'] - self.start_mem['allocated_mb']
            print(f"  Memory delta: {delta:+.1f} MB")
        print(f"<<< Exiting: {self.label}\n")

# Example usage in training script:
"""
from profile_memory import MemoryTracker, print_memory_usage, track_tensor_memory

# At the start of training
print_memory_usage("After model init")

# In training loop
for step, batch in enumerate(train_dataloader):
    with MemoryTracker("Batch loading"):
        Vo, Vp, Vo_last_frame, idx_o, idx_p = batch
        clean_images = Vp.flatten(0, 1)
    
    with MemoryTracker("Noise generation"):
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
    
    with MemoryTracker("Model forward"):
        model_output = model(Vo, idx_o, idx_p, noisy_images, timesteps, Vp, Vo_last_frame)
    
    with MemoryTracker("Loss computation"):
        loss = compute_loss(model_output, noise)
    
    with MemoryTracker("Backward pass"):
        accelerator.backward(loss)
    
    # Track specific tensors
    track_tensor_memory({
        'Vo': Vo,
        'Vp': Vp,
        'clean_images': clean_images,
        'noise': noise,
        'noisy_images': noisy_images,
        'model_output': model_output.sample if hasattr(model_output, 'sample') else model_output
    }, "Key tensors")
    
    if step == 0:
        break  # Just profile first step
"""

if __name__ == "__main__":
    # Test the memory tracking
    print("Memory Profiling Utilities")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"CUDA Device: {device}")
        print_memory_usage("Initial")
        
        # Create some test tensors
        with MemoryTracker("Creating test tensors"):
            x = torch.randn(10, 1, 64, 2048, device='cuda')
            y = torch.randn(10, 1, 64, 2048, device='cuda')
        
        track_tensor_memory({'x': x, 'y': y}, "Test tensors")
        
        print_memory_usage("After test tensors")
    else:
        print("CUDA not available")

