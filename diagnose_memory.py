#!/usr/bin/env python3
"""
Diagnostic script to find where GPU memory is actually being used.
Helps identify the discrepancy between reported peak memory and actual GPU usage.
"""

import torch
import subprocess
import os
import sys
from collections import defaultdict

def get_nvidia_smi_memory():
    """Get actual GPU memory usage from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,memory.free', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpus.append({
                    'index': int(parts[0]),
                    'used_mb': int(parts[1]),
                    'total_mb': int(parts[2]),
                    'free_mb': int(parts[3])
                })
        return gpus
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
        return []

def get_pytorch_memory(device_id=0):
    """Get PyTorch's view of GPU memory."""
    if not torch.cuda.is_available():
        return None
    
    device = torch.device(f'cuda:{device_id}')
    return {
        'allocated_mb': torch.cuda.memory_allocated(device) / 1024**2,
        'reserved_mb': torch.cuda.memory_reserved(device) / 1024**2,
        'max_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024**2,
        'max_reserved_mb': torch.cuda.max_memory_reserved(device) / 1024**2,
    }

def get_cuda_processes():
    """Get all processes using CUDA."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
             '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                if len(parts) >= 3:
                    try:
                        processes.append({
                            'pid': int(parts[0]),
                            'name': parts[1],
                            'memory_mb': int(parts[2].replace(' MiB', ''))
                        })
                    except:
                        pass
        return processes
    except Exception as e:
        print(f"Error getting CUDA processes: {e}")
        return []

def get_process_details(pid):
    """Get details about a specific process."""
    try:
        # Get command line
        with open(f'/proc/{pid}/cmdline', 'r') as f:
            cmdline = f.read().replace('\x00', ' ').strip()
        
        # Get memory usage
        with open(f'/proc/{pid}/status', 'r') as f:
            status = f.read()
            for line in status.split('\n'):
                if line.startswith('VmRSS:'):
                    rss_mb = int(line.split()[1]) / 1024  # Convert KB to MB
                    return {'cmdline': cmdline, 'rss_mb': rss_mb}
    except:
        pass
    return None

def analyze_memory_discrepancy():
    """Main function to analyze memory discrepancy."""
    print("=" * 80)
    print("GPU MEMORY DIAGNOSTIC")
    print("=" * 80)
    
    # 1. Check nvidia-smi (actual GPU memory)
    print("\n1. NVIDIA-SMI GPU Memory (Actual Hardware Usage):")
    print("-" * 80)
    gpu_info = get_nvidia_smi_memory()
    for gpu in gpu_info:
        print(f"GPU {gpu['index']}:")
        print(f"  Used:    {gpu['used_mb']:>10,} MB ({gpu['used_mb']/1024:.2f} GB)")
        print(f"  Free:    {gpu['free_mb']:>10,} MB ({gpu['free_mb']/1024:.2f} GB)")
        print(f"  Total:   {gpu['total_mb']:>10,} MB ({gpu['total_mb']/1024:.2f} GB)")
        print(f"  Usage:   {gpu['used_mb']/gpu['total_mb']*100:.1f}%")
    
    # 2. Check PyTorch's view
    print("\n2. PyTorch Memory Tracking (This Process Only):")
    print("-" * 80)
    if torch.cuda.is_available():
        pytorch_mem = get_pytorch_memory()
        if pytorch_mem:
            print(f"  Allocated:     {pytorch_mem['allocated_mb']:>10,.1f} MB")
            print(f"  Reserved:      {pytorch_mem['reserved_mb']:>10,.1f} MB")
            print(f"  Max Allocated: {pytorch_mem['max_allocated_mb']:>10,.1f} MB")
            print(f"  Max Reserved:  {pytorch_mem['max_reserved_mb']:>10,.1f} MB")
            
            if gpu_info:
                discrepancy = gpu_info[0]['used_mb'] - pytorch_mem['reserved_mb']
                print(f"\n  ⚠️  DISCREPANCY: {discrepancy:>10,.1f} MB ({discrepancy/1024:.2f} GB)")
                print(f"     (nvidia-smi shows {gpu_info[0]['used_mb']/1024:.2f} GB used,")
                print(f"      but PyTorch only shows {pytorch_mem['reserved_mb']/1024:.2f} GB)")
    else:
        print("  CUDA not available")
    
    # 3. Check all CUDA processes
    print("\n3. All Processes Using CUDA:")
    print("-" * 80)
    processes = get_cuda_processes()
    if processes:
        total_cuda_memory = 0
        for proc in processes:
            total_cuda_memory += proc['memory_mb']
            details = get_process_details(proc['pid'])
            print(f"  PID {proc['pid']:>6}: {proc['name']:20s} - {proc['memory_mb']:>8,} MB")
            if details:
                print(f"           Command: {details['cmdline'][:80]}")
                print(f"           RAM:     {details['rss_mb']:>8,.1f} MB")
        print(f"\n  Total CUDA memory from processes: {total_cuda_memory:>10,} MB ({total_cuda_memory/1024:.2f} GB)")
    else:
        print("  No CUDA processes found (or nvidia-smi query failed)")
    
    # 4. Check for multiple Python processes
    print("\n4. Python Processes on This System:")
    print("-" * 80)
    try:
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        python_procs = []
        for line in result.stdout.split('\n'):
            if 'python' in line.lower() and 'diagnose_memory' not in line:
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pid = int(parts[1])
                        mem_mb = float(parts[5])  # RSS in KB, convert to MB
                        cmd = ' '.join(parts[10:])
                        python_procs.append({
                            'pid': pid,
                            'memory_mb': mem_mb,
                            'cmd': cmd[:100]
                        })
                    except:
                        pass
        
        if python_procs:
            for proc in sorted(python_procs, key=lambda x: x['memory_mb'], reverse=True)[:10]:
                print(f"  PID {proc['pid']:>6}: {proc['memory_mb']:>8,.1f} MB - {proc['cmd']}")
        else:
            print("  No other Python processes found")
    except Exception as e:
        print(f"  Error checking processes: {e}")
    
    # 5. Check for Jupyter/notebook processes
    print("\n5. Jupyter/Notebook Processes:")
    print("-" * 80)
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'jupyter'], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"  Found {len(pids)} Jupyter process(es): {', '.join(pids)}")
            print("  ⚠️  Jupyter kernels can hold GPU memory even when idle!")
        else:
            print("  No Jupyter processes found")
    except:
        print("  Could not check for Jupyter processes")
    
    # 6. Recommendations
    print("\n6. Recommendations:")
    print("-" * 80)
    if gpu_info and pytorch_mem:
        discrepancy = gpu_info[0]['used_mb'] - pytorch_mem['reserved_mb']
        if discrepancy > 1000:  # More than 1 GB discrepancy
            print("  ⚠️  Large memory discrepancy detected!")
            print("\n  Possible causes:")
            print("  1. Multiple training processes running simultaneously")
            print("  2. DataLoader workers loading data into GPU (num_workers > 0)")
            print("  3. Other users/processes using the same GPU")
            print("  4. Jupyter notebooks holding GPU memory")
            print("  5. Previous training runs that didn't release memory")
            print("  6. CUDA context from other libraries (OpenCV, etc.)")
            print("\n  Solutions:")
            print("  - Check: nvidia-smi to see all processes")
            print("  - Set num_workers=0 in DataLoader to avoid GPU memory in workers")
            print("  - Kill other Python processes: pkill -f python")
            print("  - Restart Jupyter kernels if using notebooks")
            print("  - Use: torch.cuda.empty_cache() to clear PyTorch cache")
            print("  - Check if other users are on the same GPU (shared cluster)")

def check_dataloader_workers():
    """Check if DataLoader workers might be using GPU memory."""
    print("\n7. DataLoader Worker Memory Check:")
    print("-" * 80)
    print("  ⚠️  If num_workers > 0, each worker process can allocate GPU memory!")
    print("  Check your config file for 'num_workers' setting.")
    print("  Recommendation: Set num_workers=0 if GPU memory is limited.")

if __name__ == "__main__":
    analyze_memory_discrepancy()
    check_dataloader_workers()
    
    print("\n" + "=" * 80)
    print("To clear PyTorch's cached memory, run:")
    print("  torch.cuda.empty_cache()")
    print("  torch.cuda.reset_peak_memory_stats()")
    print("=" * 80)

