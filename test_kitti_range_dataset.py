#!/usr/bin/env python3
"""
Quick test script to verify KITTI range image dataset loading
"""
import sys
sys.path.insert(0, '/csehome/pydah/STDiffProject/stdiff/utils')

from pathlib import Path
import numpy as np
from dataset import KITTIRangeImageDataset, VidResize, VidToTensor
from torchvision import transforms

# Test dataset loading
KITTI_dir = '/scratch/pydah/kitti/processed_data'
test_folder_ids = [8, 9, 10]  # Sequences 08, 09, 10 for testing

# Use proper video transforms (VidResize and VidToTensor for lists of images)
transform = transforms.Compose([
    VidResize((64, 2048)),  # Resize to model input size
    VidToTensor(),  # Convert list of PIL Images to tensor (T, C, H, W)
])

print("Creating KITTI Range Image Dataset...")
try:
    dataset = KITTIRangeImageDataset(
        KITTI_dir=KITTI_dir,
        test_folder_ids=test_folder_ids,
        transform=transform,
        train=True,
        val=False,
        num_observed_frames=5,
        num_predict_frames=5
    )
    
    print(f"✓ Dataset created successfully!")
    print(f"  Number of training clips: {len(dataset.train_clips)}")
    
    # Try loading one sample
    print("\nTesting data loading...")
    clip_dataset = dataset.__getClips__([dataset.train_folders[0]])
    print(f"  Clips in first folder: {len(clip_dataset)}")
    
    if len(clip_dataset) > 0:
        print(f"  First clip has {len(clip_dataset[0])} frames")
        print(f"  First frame path: {clip_dataset[0][0]}")
        
        # Try loading the actual data
        from dataset import RangeImageClipDataset
        range_dataset = RangeImageClipDataset(
            num_observed_frames=5,
            num_predict_frames=5,
            clips=clip_dataset[:1],  # Just first clip
            transform=transform,
            color_mode='grey_scale'
        )
        
        past, future, past_mask, future_mask = range_dataset[0]
        print(f"\n✓ Successfully loaded sample!")
        print(f"  Past clip shape: {past.shape}")
        print(f"  Future clip shape: {future.shape}")
        print(f"  Past mask shape: {past_mask.shape}")
        print(f"  Future mask shape: {future_mask.shape}")
        print(f"  Valid pixels in past: {past_mask.sum().item()}/{past_mask.numel()}")
        print(f"  Valid pixels in future: {future_mask.sum().item()}/{future_mask.numel()}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed! Dataset is ready for training.")

