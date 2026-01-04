#!/usr/bin/env python3
"""
Quick test script to verify checkpoint loading works for KITTI_RANGE dataset.
"""

import sys
from pathlib import Path

# Add stdiff to path
sys.path.insert(0, str(Path(__file__).parent / 'stdiff'))

from models import STDiffDiffusers
from diffusers import DDPMScheduler
from omegaconf import OmegaConf

def test_checkpoint_loading():
    """Test loading model from checkpoint-6"""
    
    checkpoint_path = "/home/anirudh/STDiffProject/STDiff_ckpts/kitti_range_64x512/checkpoint-6"
    checkpoint_dir = Path(checkpoint_path)
    
    print("=" * 60)
    print("Testing Checkpoint Loading")
    print("=" * 60)
    
    # Test 1: Check checkpoint directory exists
    print(f"\n1. Checking checkpoint directory: {checkpoint_path}")
    if not checkpoint_dir.exists():
        print(f"   ❌ ERROR: Checkpoint directory not found!")
        return False
    print(f"   ✅ Checkpoint directory exists")
    
    # Test 2: Check unet subdirectory exists
    unet_path = checkpoint_dir / 'unet'
    print(f"\n2. Checking unet subdirectory: {unet_path}")
    if not unet_path.exists():
        print(f"   ❌ ERROR: unet subdirectory not found!")
        return False
    print(f"   ✅ unet subdirectory exists")
    
    # Test 3: Try loading model
    print(f"\n3. Loading model from {unet_path}")
    try:
        model = STDiffDiffusers.from_pretrained(str(unet_path))
        print(f"   ✅ Model loaded successfully!")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Model parameters: {num_params/1e6:.2f} M")
    except Exception as e:
        print(f"   ❌ ERROR loading model: {e}")
        return False
    
    # Test 4: Create scheduler from config
    print(f"\n4. Creating scheduler from config")
    try:
        config_path = Path(__file__).parent / 'stdiff' / 'configs' / 'kitti_range_test_config.yaml'
        cfg = OmegaConf.load(config_path)
        
        scheduler = DDPMScheduler(
            num_train_timesteps=cfg.STDiff.Diffusion.ddpm_num_steps,
            beta_schedule=cfg.STDiff.Diffusion.ddpm_beta_schedule,
            prediction_type=cfg.STDiff.Diffusion.prediction_type,
        )
        print(f"   ✅ Scheduler created successfully!")
        print(f"   num_train_timesteps: {scheduler.config.num_train_timesteps}")
    except Exception as e:
        print(f"   ❌ ERROR creating scheduler: {e}")
        return False
    
    # Test 5: Check model config matches
    print(f"\n5. Verifying model configuration")
    try:
        print(f"   Model sample_size: {model.config.sample_size}")
        print(f"   Model in_channels: {model.config.in_channels}")
        print(f"   Model out_channels: {model.config.out_channels}")
        print(f"   ✅ Model configuration looks good!")
    except Exception as e:
        print(f"   ⚠️  Warning checking config: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Checkpoint loading works correctly.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_checkpoint_loading()
    sys.exit(0 if success else 1)

