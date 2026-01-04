"""
Script to load and display saved range image evaluation results.
"""

import torch
import numpy as np
from pathlib import Path
import argparse


def load_and_display_results(results_dir):
    """Load and display saved evaluation results."""
    results_dir = Path(results_dir)
    
    print("="*80)
    print("LOADING SAVED RANGE IMAGE EVALUATION RESULTS")
    print("="*80)
    print(f"Results directory: {results_dir}\n")
    
    # 1. Load summary statistics
    summary_path = results_dir / 'range_loss_summary.pt'
    if summary_path.exists():
        print("1. SUMMARY STATISTICS (range_loss_summary.pt)")
        print("-"*80)
        summary = torch.load(summary_path)
        
        print(f"Total samples: {summary['num_samples']}")
        print(f"Number of frames: {summary['num_frames']}")
        print(f"Image shape: {summary['image_shape']}")
        print(f"Used mask: {summary.get('used_mask', False)}")
        
        print("\nOverall Metrics:")
        print(f"  Overall L1 Loss (with mask): {summary['overall_frame_loss']:.6f}")
        if 'overall_frame_loss_no_mask' in summary and summary['overall_frame_loss_no_mask'] is not None:
            print(f"  Overall L1 Loss (without mask): {summary['overall_frame_loss_no_mask']:.6f}")
            print(f"  Difference: {summary['overall_frame_loss_no_mask'] - summary['overall_frame_loss']:.6f}")
        print(f"  Overall pixel loss: {summary['overall_pixel_loss']:.6f}")
        
        print("\nPer-Frame Loss (WITH mask):")
        print(f"{'Frame':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-"*80)
        for i in range(summary['num_frames']):
            print(f"{i:<8} "
                  f"{summary['per_frame_loss_mean'][i]:<12.6f} "
                  f"{summary['per_frame_loss_std'][i]:<12.6f} "
                  f"{summary['per_frame_loss_min'][i]:<12.6f} "
                  f"{summary['per_frame_loss_max'][i]:<12.6f}")
        
        if 'per_frame_loss_mean_no_mask' in summary and summary['per_frame_loss_mean_no_mask'] is not None:
            print("\nPer-Frame Loss (WITHOUT mask):")
            print(f"{'Frame':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Diff':<12}")
            print("-"*80)
            for i in range(summary['num_frames']):
                diff = summary['per_frame_loss_mean_no_mask'][i] - summary['per_frame_loss_mean'][i]
                print(f"{i:<8} "
                      f"{summary['per_frame_loss_mean_no_mask'][i]:<12.6f} "
                      f"{summary['per_frame_loss_std_no_mask'][i]:<12.6f} "
                      f"{summary['per_frame_loss_min_no_mask'][i]:<12.6f} "
                      f"{summary['per_frame_loss_max_no_mask'][i]:<12.6f} "
                      f"{diff:<12.6f}")
    else:
        print(f"❌ Summary file not found: {summary_path}")
    
    # 2. Load per-pixel loss maps
    pixel_maps_path = results_dir / 'per_pixel_loss_maps.npz'
    if pixel_maps_path.exists():
        print("\n\n2. PER-PIXEL LOSS MAPS (per_pixel_loss_maps.npz)")
        print("-"*80)
        pixel_maps = np.load(pixel_maps_path)
        
        print(f"Available arrays: {list(pixel_maps.keys())}")
        for key in pixel_maps.keys():
            arr = pixel_maps[key]
            print(f"\n  {key}:")
            print(f"    Shape: {arr.shape}")
            print(f"    Range: [{arr.min():.6f}, {arr.max():.6f}]")
            print(f"    Mean: {arr.mean():.6f}")
            print(f"    Std: {arr.std():.6f}")
    else:
        print(f"\n❌ Per-pixel maps file not found: {pixel_maps_path}")
    
    # 3. Load per-frame losses
    frame_losses_path = results_dir / 'per_frame_losses.npy'
    if frame_losses_path.exists():
        print("\n\n3. PER-FRAME LOSSES (per_frame_losses.npy)")
        print("-"*80)
        frame_losses = np.load(frame_losses_path)
        
        print(f"Shape: {frame_losses.shape}  # (num_samples, num_frames)")
        print(f"Range: [{frame_losses.min():.6f}, {frame_losses.max():.6f}]")
        print(f"Mean: {frame_losses.mean():.6f}")
        print(f"Std: {frame_losses.std():.6f}")
        print(f"\nPer-frame statistics:")
        print(f"{'Frame':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-"*80)
        for i in range(frame_losses.shape[1]):
            print(f"{i:<8} "
                  f"{frame_losses[:, i].mean():<12.6f} "
                  f"{frame_losses[:, i].std():<12.6f} "
                  f"{frame_losses[:, i].min():<12.6f} "
                  f"{frame_losses[:, i].max():<12.6f}")
    else:
        print(f"\n❌ Per-frame losses file not found: {frame_losses_path}")
    
    print("\n" + "="*80)
    print("To load in Python:")
    print("="*80)
    print("import torch")
    print("import numpy as np")
    print(f"summary = torch.load('{summary_path}')")
    print(f"pixel_maps = np.load('{pixel_maps_path}')")
    print(f"frame_losses = np.load('{frame_losses_path}')")


def main():
    parser = argparse.ArgumentParser(description='Display saved range image evaluation results')
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to directory containing saved results (range_loss_summary.pt, etc.)'
    )
    
    args = parser.parse_args()
    load_and_display_results(args.results_dir)


if __name__ == '__main__':
    main()

