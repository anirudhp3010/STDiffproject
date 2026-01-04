# Why 45 GB GPU Memory When Peak Shows 1-2 GB?

## The Problem
- **PyTorch peak memory**: 1-2 GB
- **nvidia-smi GPU usage**: 45 GB
- **Discrepancy**: ~43 GB unaccounted for

## Common Causes (Ranked by Likelihood)

### 1. **Multiple Training Processes** (Most Likely)
**What happens:**
- Multiple training scripts running simultaneously
- Each process allocates its own GPU memory
- PyTorch only tracks memory for the current process

**Check:**
```bash
nvidia-smi
# Look for multiple python processes
ps aux | grep python | grep train
```

**Solution:**
- Kill duplicate processes: `pkill -f train_stdiff.py`
- Use `nvidia-smi` to see all processes using GPU

### 2. **DataLoader Workers** (Very Common)
**Your config shows:** `num_workers: 2`

**What happens:**
- Each DataLoader worker is a separate process
- If workers accidentally touch CUDA (e.g., data preprocessing on GPU), each worker allocates GPU memory
- 2 workers × ~20 GB each = 40 GB

**Check:**
```python
# In your dataset.py, check if any operations use .cuda() or .to('cuda')
# Common culprits:
# - Transforms applied on GPU
# - Data normalization on GPU
# - Augmentation on GPU
```

**Solution:**
```yaml
# In config file:
num_workers: 0  # Disable multiprocessing workers
```

### 3. **Shared GPU Cluster**
**What happens:**
- Multiple users sharing the same GPU
- Other users' processes using GPU memory
- Your process only sees its own allocation

**Check:**
```bash
nvidia-smi
# Look at "Processes" section - see all PIDs and users
```

**Solution:**
- Request exclusive GPU access
- Use `CUDA_VISIBLE_DEVICES` to isolate GPU

### 4. **Jupyter Notebooks / Interactive Sessions**
**What happens:**
- Jupyter kernels keep GPU memory allocated
- Previous cells' variables still in GPU memory
- Notebooks don't release memory until kernel restart

**Check:**
```bash
ps aux | grep jupyter
```

**Solution:**
- Restart Jupyter kernels
- Use `torch.cuda.empty_cache()` in notebooks
- Close unused notebooks

### 5. **CUDA Context Overhead**
**What happens:**
- First CUDA operation creates a large context
- Context includes cuDNN, cuBLAS libraries
- Can be 1-5 GB just for initialization

**Check:**
```python
import torch
torch.cuda.init()
# Check memory before doing anything
torch.cuda.memory_allocated()  # Should be ~0
torch.cuda.memory_reserved()    # Might be 1-5 GB (context)
```

### 6. **Memory Fragmentation**
**What happens:**
- PyTorch's caching allocator reserves large blocks
- Even if you only use 2 GB, allocator might reserve 20+ GB
- Memory is "reserved" but not "allocated"

**Check:**
```python
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
# Large gap = fragmentation
```

**Solution:**
```python
torch.cuda.empty_cache()  # Release unused cached memory
```

### 7. **Model Checkpoints / State Dicts**
**What happens:**
- Large checkpoints loaded into GPU memory
- EMA model (full copy of model)
- Optimizer states (Adam stores 2x model size)

**Your case:**
- EMA enabled: `use_ema: True` (line 72)
- This creates a full model copy = ~200-400 MB
- But still doesn't explain 45 GB

### 8. **Previous Training Runs Not Cleared**
**What happens:**
- Previous training didn't release GPU memory
- Process crashed but GPU memory not freed
- Zombie processes holding memory

**Check:**
```bash
# Find zombie processes
ps aux | grep python | grep -v grep
# Check if they're actually running or zombie
```

**Solution:**
```bash
# Kill all Python processes (be careful!)
pkill -f python
# Or more specific:
pkill -f train_stdiff
```

## Diagnostic Steps

### Step 1: Run the diagnostic script
```bash
cd /home/anirudh/STDiffProject
python diagnose_memory.py
```

This will show:
- Actual GPU memory from nvidia-smi
- PyTorch's view of memory
- All processes using CUDA
- Memory discrepancy

### Step 2: Check nvidia-smi directly
```bash
nvidia-smi
# Look at:
# - Memory usage per process
# - Multiple processes?
# - Other users?
```

### Step 3: Check for multiple Python processes
```bash
ps aux | grep python
# Count how many training processes are running
```

### Step 4: Check DataLoader workers
```python
# In your training script, add:
print(f"DataLoader num_workers: {train_dataloader.num_workers}")
# If > 0, each worker might allocate GPU memory
```

## Quick Fixes

### Fix 1: Disable DataLoader Workers
```yaml
# In kitti_range_train_config.yaml
num_workers: 0  # Change from 2 to 0
```

### Fix 2: Clear GPU Memory
```python
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

### Fix 3: Kill Other Processes
```bash
# See what's using GPU
nvidia-smi

# Kill specific process (replace PID)
kill -9 <PID>

# Or kill all training processes
pkill -f train_stdiff
```

### Fix 4: Use Exclusive GPU
```bash
# In your SLURM script or before training
export CUDA_VISIBLE_DEVICES=0
# This isolates to single GPU
```

## Expected Memory Breakdown

For your current config:
- **Model parameters**: ~100-200 MB (fp16)
- **Model activations**: ~150-200 MB
- **Gradients**: ~100-200 MB (fp16)
- **EMA model**: ~100-200 MB
- **Input data**: ~5-10 MB
- **Intermediate tensors**: ~30-50 MB
- **Total expected**: ~500-900 MB

**If you're seeing 45 GB, something else is using the GPU!**

## Most Likely Culprit

Based on your setup:
1. **Multiple training processes** (80% probability)
2. **DataLoader workers with GPU operations** (15% probability)
3. **Other users on shared GPU** (5% probability)

Run `diagnose_memory.py` to identify the exact cause!

