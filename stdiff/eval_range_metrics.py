"""
Evaluation script for calculating range image loss metrics.
Computes L1 loss per frame per pixel across all future predictions.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse


def calculate_range_image_loss(preds_dir, sample_idx=0, global_min=None, global_max=None, use_original_scale=False):
    """
    Calculate range image L1 loss per frame per pixel across all predictions.
    
    Args:
        preds_dir: Path to directory containing Preds_*.pt files
        sample_idx: Which sample to use from predictions (default: 0)
        global_min: Global minimum value used for normalization (required for denormalization)
        global_max: Global maximum value used for normalization (required for denormalization)
        use_original_scale: If True, denormalize predictions and ground truth back to original scale using global min/max
        
    Returns:
        Dictionary containing various loss metrics
    """
    preds_dir = Path(preds_dir)
    preds_files = sorted(list(preds_dir.glob("Preds_*.pt")))
    
    if len(preds_files) == 0:
        raise ValueError(f"No Preds_*.pt files found in {preds_dir}")
    
    print(f"Found {len(preds_files)} prediction files")
    
    # Try to get global min/max from first file if not provided
    if global_min is None or global_max is None:
        first_file_data = torch.load(preds_files[0])
        if 'global_min' in first_file_data and 'global_max' in first_file_data:
            global_min = first_file_data['global_min']
            global_max = first_file_data['global_max']
            print(f"Found global min/max in saved files: [{global_min:.6f}, {global_max:.6f}]")
        elif use_original_scale:
            raise ValueError("global_min and global_max must be provided (either as arguments or saved in prediction files) when use_original_scale=True")
    
    # Accumulators for losses
    all_frame_losses = []  # List of tensors: (num_frames,) for each batch (with mask if available)
    all_frame_losses_no_mask = []  # List of tensors: (num_frames,) for each batch (without mask)
    all_pixel_losses = []  # List of tensors: (H, W) for each batch
    total_samples = 0
    total_frames = 0
    used_mask = False  # Track if masks were used (will be set per file)
    
    # Process each prediction file
    for file_idx, file in enumerate(tqdm(preds_files, desc="Processing prediction files")):
        data = torch.load(file)
        
        # Debug: Print data structure for first file
        if file_idx == 0:
            print(f"\n[DEBUG] First file: {file.name}")
            print(f"  Keys in data: {list(data.keys())}")
            if 'g_Vp' in data:
                print(f"  g_Vp shape: {data['g_Vp'].shape}")
                print(f"  g_Vp range: [{data['g_Vp'].min().item():.6f}, {data['g_Vp'].max().item():.6f}], mean: {data['g_Vp'].mean().item():.6f}")
            if 'g_Preds' in data:
                print(f"  g_Preds shape: {data['g_Preds'].shape}")
                print(f"  g_Preds range: [{data['g_Preds'].min().item():.6f}, {data['g_Preds'].max().item():.6f}], mean: {data['g_Preds'].mean().item():.6f}")
            if 'Vo' in data:
                print(f"  Vo shape: {data['Vo'].shape}")
                print(f"  Vo range: [{data['Vo'].min().item():.6f}, {data['Vo'].max().item():.6f}], mean: {data['Vo'].mean().item():.6f}")
            if 'g_Vp_mask' in data:
                print(f"  g_Vp_mask shape: {data['g_Vp_mask'].shape}")
                print(f"  g_Vp_mask valid pixels: {data['g_Vp_mask'].sum().item()} / {data['g_Vp_mask'].numel()}")
            if 'global_min' in data and 'global_max' in data:
                print(f"  global_min: {data['global_min']:.6f}, global_max: {data['global_max']:.6f}")
        
        # Extract ground truth and predictions
        if 'g_Vp' not in data:
            raise KeyError(f"Key 'g_Vp' not found in {file}. Available keys: {list(data.keys())}")
        if 'g_Preds' not in data:
            raise KeyError(f"Key 'g_Preds' not found in {file}. Available keys: {list(data.keys())}")
            
        Vp = data['g_Vp']  # Ground truth: (N, Tp, C, H, W)
        preds = data['g_Preds']  # Predictions: (N, sample_num, Tp, C, H, W)
        
        # Use the specified sample from predictions
        if preds.shape[1] > sample_idx:
            preds = preds[:, sample_idx, ...]  # (N, Tp, C, H, W)
        else:
            print(f"Warning: sample_idx {sample_idx} not available, using sample 0")
            preds = preds[:, 0, ...]  # (N, Tp, C, H, W)
        
        # Ensure same number of frames
        num_frames = min(Vp.shape[1], preds.shape[1])
        Vp = Vp[:, :num_frames, ...]
        preds = preds[:, :num_frames, ...]
        
        # Check value ranges before conversion
        vp_min, vp_max = Vp.min().item(), Vp.max().item()
        pred_min, pred_max = preds.min().item(), preds.max().item()
        
        # Debug: Print shapes and ranges after loading
        if file_idx == 0:
            print(f"\n[DEBUG] After loading (before range conversion):")
            print(f"  Vp shape: {Vp.shape}, range: [{vp_min:.6f}, {vp_max:.6f}], mean: {Vp.mean().item():.6f}")
            print(f"  preds shape: {preds.shape}, range: [{pred_min:.6f}, {pred_max:.6f}], mean: {preds.mean().item():.6f}")
            print(f"  num_frames: {num_frames}")
            if global_min is not None and global_max is not None:
                print(f"  Global min/max: [{global_min:.6f}, {global_max:.6f}]")
        
        # Convert Vp from [-1, 1] to [0, 1] to match preds range
        # Vp comes from dataset with norm_transform: x * 2 - 1 (converts [0,1] -> [-1,1])
        # preds come from pipeline: (image / 2 + 0.5).clamp(0, 1) (converts [-1,1] -> [0,1])
        # So we need to convert Vp: (Vp + 1) / 2 to get [0, 1]
        # Check if Vp is in [-1, 1] range (has negative values or max > 1.1)
        if vp_min < 0 or vp_max > 1.1:
            if file_idx == 0:
                print(f"  Converting Vp from [-1, 1] to [0, 1] range (detected range: [{vp_min:.6f}, {vp_max:.6f}])")
            Vp = (Vp + 1.0) / 2.0  # Convert [-1, 1] -> [0, 1]
            Vp = Vp.clamp(0, 1)  # Ensure in valid range
        elif file_idx == 0:
            print(f"  Vp already in [0, 1] range, no conversion needed")
        
        # Ensure preds are also in [0, 1] range (should already be, but clamp for safety)
        if pred_min < 0 or pred_max > 1.1:
            if file_idx == 0:
                print(f"  Warning: preds range [{pred_min:.6f}, {pred_max:.6f}] outside [0, 1], clamping")
            preds = preds.clamp(0, 1)
        
        # Check ranges after conversion
        if file_idx == 0:
            print(f"\n[DEBUG] After range conversion (before denormalization):")
            print(f"  Vp shape: {Vp.shape}, range: [{Vp.min().item():.6f}, {Vp.max().item():.6f}], mean: {Vp.mean().item():.6f}")
            print(f"  preds shape: {preds.shape}, range: [{preds.min().item():.6f}, {preds.max().item():.6f}], mean: {preds.mean().item():.6f}")
        
        # Denormalize from [0, 1] back to original scale using global min/max
        # Both Vp and preds are now in [0, 1] normalized range
        # To get back to original scale: value * (global_max - global_min) + global_min
        if use_original_scale:
            if global_min is None or global_max is None:
                raise ValueError("global_min and global_max must be provided when use_original_scale=True")
            
            # Denormalize: [0, 1] -> [global_min, global_max]
            Vp = Vp * (global_max - global_min) + global_min
            preds = preds * (global_max - global_min) + global_min
            
            if file_idx == 0:
                print(f"\n[DEBUG] After denormalization using global min/max [{global_min:.6f}, {global_max:.6f}]:")
                print(f"  Vp range: [{Vp.min().item():.6f}, {Vp.max().item():.6f}], mean: {Vp.mean().item():.6f}")
                print(f"  preds range: [{preds.min().item():.6f}, {preds.max().item():.6f}], mean: {preds.mean().item():.6f}")
        
        # Get mask if available
        if 'g_Vp_mask' in data:
            Vp_mask = data['g_Vp_mask']  # (N, Tp, H, W) - bool tensor: 1 for valid, 0 for invalid
            # Ensure mask matches the number of frames
            if Vp_mask.shape[1] > num_frames:
                Vp_mask = Vp_mask[:, :num_frames, ...]
            elif Vp_mask.shape[1] < num_frames:
                # Pad mask if needed
                pad_frames = num_frames - Vp_mask.shape[1]
                Vp_mask = torch.cat([Vp_mask, Vp_mask[:, -1:, ...].repeat(1, pad_frames, 1, 1)], dim=1)
            if not used_mask:
                used_mask = True  # Update global flag if mask is found
        else:
            Vp_mask = None
        
        # Calculate L1 loss per pixel: (N, Tp, C, H, W)
        l1_loss_per_pixel = F.l1_loss(preds, Vp, reduction='none')
        
        # Average across channels if C > 1
        if l1_loss_per_pixel.shape[2] > 1:
            l1_loss_per_pixel = l1_loss_per_pixel.mean(dim=2)  # (N, Tp, H, W)
        else:
            l1_loss_per_pixel = l1_loss_per_pixel.squeeze(2)  # (N, Tp, H, W)
        
        # IMPORTANT: Compute no_mask version BEFORE applying mask
        # Metric 1 (no_mask): L1 loss per frame WITHOUT mask (average across ALL pixels)
        # Shape: (N, Tp)
        l1_loss_per_frame_no_mask = l1_loss_per_pixel.mean(dim=(2, 3))  # Average over H, W (all pixels)
        
        # Apply mask if available (set invalid pixels to 0 loss)
        if Vp_mask is not None:
            # Expand mask to match loss shape: (N, Tp, H, W)
            Vp_mask_expanded = Vp_mask.float()  # Convert bool to float for multiplication
            l1_loss_per_pixel_masked = l1_loss_per_pixel * Vp_mask_expanded
        else:
            l1_loss_per_pixel_masked = l1_loss_per_pixel  # No mask, use original
        
        # Debug: Print loss statistics for first file
        if file_idx == 0:
            print(f"\n[DEBUG] L1 loss statistics (first file):")
            print(f"  l1_loss_per_pixel (unmasked) shape: {l1_loss_per_pixel.shape}")
            print(f"  l1_loss_per_pixel (unmasked) range: [{l1_loss_per_pixel.min().item():.6f}, {l1_loss_per_pixel.max().item():.6f}]")
            if Vp_mask is not None:
                valid_pixels = Vp_mask_expanded.sum().item()
                total_pixels = Vp_mask_expanded.numel()
                # l1_loss_per_pixel_masked already has invalid pixels set to 0, so just sum and divide
                masked_loss = l1_loss_per_pixel_masked.sum() / valid_pixels if valid_pixels > 0 else 0
                print(f"  Using mask: {valid_pixels}/{total_pixels} valid pixels")
                print(f"  Masked L1 loss mean (valid pixels only): {masked_loss.item():.6f}")
                print(f"  Unmasked L1 loss mean (all pixels): {l1_loss_per_pixel.mean().item():.6f}")
            else:
                print(f"  L1 loss mean (all pixels): {l1_loss_per_pixel.mean().item():.6f}")
        
        # Metric 1 (with mask): L1 loss per frame (average across VALID pixels for each frame)
        # Shape: (N, Tp)
        if Vp_mask is not None:
            # Average only over valid pixels (with mask)
            # Use l1_loss_per_pixel_masked which already has invalid pixels set to 0
            valid_pixels_per_frame = Vp_mask_expanded.sum(dim=(2, 3))  # (N, Tp) - number of valid pixels per frame
            l1_loss_per_frame = l1_loss_per_pixel_masked.sum(dim=(2, 3)) / valid_pixels_per_frame.clamp(min=1)  # Average over H, W (valid only)
        else:
            l1_loss_per_frame = l1_loss_per_pixel.mean(dim=(2, 3))  # Average over H, W (all pixels)
        
        # Metric 2: L1 loss per pixel (average across frames for each pixel)
        # Shape: (N, H, W)
        if Vp_mask is not None:
            # Average only over valid frames for each pixel
            # Use l1_loss_per_pixel_masked which already has invalid pixels set to 0
            valid_frames_per_pixel = Vp_mask_expanded.sum(dim=1)  # (N, H, W) - number of valid frames per pixel
            l1_loss_per_pixel_loc = l1_loss_per_pixel_masked.sum(dim=1) / valid_frames_per_pixel.clamp(min=1)  # Average over Tp (valid only)
        else:
            l1_loss_per_pixel_loc = l1_loss_per_pixel.mean(dim=1)  # Average over Tp (all frames)
        
        # Store results
        all_frame_losses.append(l1_loss_per_frame.cpu())
        all_frame_losses_no_mask.append(l1_loss_per_frame_no_mask.cpu())
        all_pixel_losses.append(l1_loss_per_pixel_loc.cpu())
        
        total_samples += Vp.shape[0]
        total_frames += Vp.shape[0] * num_frames
    
    # Concatenate all batches
    all_frame_losses = torch.cat(all_frame_losses, dim=0)  # (total_samples, Tp) - with mask
    all_frame_losses_no_mask = torch.cat(all_frame_losses_no_mask, dim=0)  # (total_samples, Tp) - without mask
    all_pixel_losses = torch.cat(all_pixel_losses, dim=0)  # (total_samples, H, W)
    
    # Calculate statistics
    num_frames = all_frame_losses.shape[1]
    H, W = all_pixel_losses.shape[1], all_pixel_losses.shape[2]
    
    # Per-frame statistics WITH mask (across all samples)
    frame_loss_mean = all_frame_losses.mean(dim=0).numpy()  # (Tp,)
    frame_loss_std = all_frame_losses.std(dim=0).numpy()  # (Tp,)
    frame_loss_min = all_frame_losses.min(dim=0)[0].numpy()  # (Tp,)
    frame_loss_max = all_frame_losses.max(dim=0)[0].numpy()  # (Tp,)
    
    # Per-frame statistics WITHOUT mask (across all samples)
    frame_loss_mean_no_mask = all_frame_losses_no_mask.mean(dim=0).numpy()  # (Tp,)
    frame_loss_std_no_mask = all_frame_losses_no_mask.std(dim=0).numpy()  # (Tp,)
    frame_loss_min_no_mask = all_frame_losses_no_mask.min(dim=0)[0].numpy()  # (Tp,)
    frame_loss_max_no_mask = all_frame_losses_no_mask.max(dim=0)[0].numpy()  # (Tp,)
    
    # Overall frame loss (average across all frames and samples)
    overall_frame_loss = all_frame_losses.mean().item()
    overall_frame_loss_no_mask = all_frame_losses_no_mask.mean().item()
    
    # Per-pixel statistics (across all samples)
    pixel_loss_mean = all_pixel_losses.mean(dim=0).numpy()  # (H, W)
    pixel_loss_std = all_pixel_losses.std(dim=0).numpy()  # (H, W)
    pixel_loss_min = all_pixel_losses.min(dim=0)[0].numpy()  # (H, W)
    pixel_loss_max = all_pixel_losses.max(dim=0)[0].numpy()  # (H, W)
    
    # Overall pixel loss (average across all pixels and samples)
    overall_pixel_loss = all_pixel_losses.mean().item()
    
    results = {
        'overall_frame_loss': overall_frame_loss,
        'overall_frame_loss_no_mask': overall_frame_loss_no_mask,
        'overall_pixel_loss': overall_pixel_loss,
        'per_frame_loss_mean': frame_loss_mean,
        'per_frame_loss_std': frame_loss_std,
        'per_frame_loss_min': frame_loss_min,
        'per_frame_loss_max': frame_loss_max,
        'per_frame_loss_mean_no_mask': frame_loss_mean_no_mask,
        'per_frame_loss_std_no_mask': frame_loss_std_no_mask,
        'per_frame_loss_min_no_mask': frame_loss_min_no_mask,
        'per_frame_loss_max_no_mask': frame_loss_max_no_mask,
        'per_pixel_loss_mean': pixel_loss_mean,
        'per_pixel_loss_std': pixel_loss_std,
        'per_pixel_loss_min': pixel_loss_min,
        'per_pixel_loss_max': pixel_loss_max,
        'num_samples': total_samples,
        'num_frames': num_frames,
        'image_shape': (H, W),
        'all_frame_losses': all_frame_losses.numpy(),
        'all_frame_losses_no_mask': all_frame_losses_no_mask.numpy(),
        'all_pixel_losses': all_pixel_losses.numpy(),
        'global_min': global_min if use_original_scale else None,
        'global_max': global_max if use_original_scale else None,
        'use_original_scale': use_original_scale,
        'used_mask': used_mask,  # Whether masks were used in loss calculation
    }
    
    return results


def print_results(results):
    """Print evaluation results in a readable format."""
    print("\n" + "="*80)
    print("RANGE IMAGE L1 LOSS EVALUATION RESULTS")
    print("="*80)
    print(f"\nTotal samples: {results['num_samples']}")
    print(f"Number of future frames: {results['num_frames']}")
    print(f"Image shape: {results['image_shape']}")
    if results.get('used_mask'):
        print(f"Note: Losses computed using VALID pixels only (invalid pixels masked out)")
    else:
        print(f"Note: Losses computed on ALL pixels (no mask available)")
    
    print("\n" + "-"*80)
    print("OVERALL METRICS")
    print("-"*80)
    if results.get('used_mask'):
        print(f"Overall L1 Loss WITH mask (averaged across all frames and pixels): {results['overall_frame_loss']:.6f}")
        print(f"Overall L1 Loss WITHOUT mask (averaged across all frames and pixels): {results['overall_frame_loss_no_mask']:.6f}")
        print(f"Difference (no_mask - with_mask): {results['overall_frame_loss_no_mask'] - results['overall_frame_loss']:.6f}")
    else:
        print(f"Overall L1 Loss (averaged across all frames and pixels): {results['overall_frame_loss']:.6f}")
    print(f"Overall L1 Loss (averaged across all pixels and frames): {results['overall_pixel_loss']:.6f}")
    if results.get('use_original_scale') and results.get('global_min') is not None:
        print(f"Note: Losses are computed in original scale [{results['global_min']:.6f}, {results['global_max']:.6f}] meters")
    
    print("\n" + "-"*80)
    if results.get('used_mask'):
        print("PER-FRAME METRICS WITH MASK (averaged across valid pixels only)")
    else:
        print("PER-FRAME METRICS (averaged across pixels)")
    print("-"*80)
    print(f"{'Frame':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*80)
    for frame_idx in range(results['num_frames']):
        print(f"{frame_idx:<8} "
              f"{results['per_frame_loss_mean'][frame_idx]:<12.6f} "
              f"{results['per_frame_loss_std'][frame_idx]:<12.6f} "
              f"{results['per_frame_loss_min'][frame_idx]:<12.6f} "
              f"{results['per_frame_loss_max'][frame_idx]:<12.6f}")
    
    if results.get('used_mask'):
        print("\n" + "-"*80)
        print("PER-FRAME METRICS WITHOUT MASK (averaged across ALL pixels)")
        print("-"*80)
        print(f"{'Frame':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Diff':<12}")
        print("-"*80)
        for frame_idx in range(results['num_frames']):
            diff = results['per_frame_loss_mean_no_mask'][frame_idx] - results['per_frame_loss_mean'][frame_idx]
            print(f"{frame_idx:<8} "
                  f"{results['per_frame_loss_mean_no_mask'][frame_idx]:<12.6f} "
                  f"{results['per_frame_loss_std_no_mask'][frame_idx]:<12.6f} "
                  f"{results['per_frame_loss_min_no_mask'][frame_idx]:<12.6f} "
                  f"{results['per_frame_loss_max_no_mask'][frame_idx]:<12.6f} "
                  f"{diff:<12.6f}")
    
    print("\n" + "-"*80)
    print("PER-PIXEL METRICS (averaged across frames)")
    print("-"*80)
    H, W = results['image_shape']
    pixel_mean = results['per_pixel_loss_mean']
    print(f"Pixel loss statistics:")
    print(f"  Mean across all pixels: {pixel_mean.mean():.6f}")
    print(f"  Std across all pixels: {pixel_mean.std():.6f}")
    print(f"  Min pixel loss: {pixel_mean.min():.6f}")
    print(f"  Max pixel loss: {pixel_mean.max():.6f}")
    print(f"\nPer-pixel loss map shape: ({H}, {W})")
    print("(Use the saved numpy arrays for detailed per-pixel analysis)")


def save_results(results, output_dir):
    """Save results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary statistics
    summary = {
        'overall_frame_loss': results['overall_frame_loss'],
        'overall_frame_loss_no_mask': results.get('overall_frame_loss_no_mask'),
        'overall_pixel_loss': results['overall_pixel_loss'],
        'per_frame_loss_mean': results['per_frame_loss_mean'],
        'per_frame_loss_std': results['per_frame_loss_std'],
        'per_frame_loss_min': results['per_frame_loss_min'],
        'per_frame_loss_max': results['per_frame_loss_max'],
        'per_frame_loss_mean_no_mask': results.get('per_frame_loss_mean_no_mask'),
        'per_frame_loss_std_no_mask': results.get('per_frame_loss_std_no_mask'),
        'per_frame_loss_min_no_mask': results.get('per_frame_loss_min_no_mask'),
        'per_frame_loss_max_no_mask': results.get('per_frame_loss_max_no_mask'),
        'num_samples': results['num_samples'],
        'num_frames': results['num_frames'],
        'image_shape': results['image_shape'],
        'used_mask': results.get('used_mask', False),
    }
    torch.save(summary, output_dir / 'range_loss_summary.pt')
    
    # Save per-pixel loss maps
    np.savez(
        output_dir / 'per_pixel_loss_maps.npz',
        mean=results['per_pixel_loss_mean'],
        std=results['per_pixel_loss_std'],
        min=results['per_pixel_loss_min'],
        max=results['per_pixel_loss_max'],
    )
    
    # Save per-frame losses (optional - can be large)
    np.save(output_dir / 'per_frame_losses.npy', results['all_frame_losses'])
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - range_loss_summary.pt: Summary statistics")
    print(f"  - per_pixel_loss_maps.npz: Per-pixel loss maps")
    print(f"  - per_frame_losses.npy: Per-frame losses for all samples")


def main():
    parser = argparse.ArgumentParser(description='Evaluate range image L1 loss metrics')
    parser.add_argument(
        '--preds_dir',
        type=str,
        required=True,
        help='Path to directory containing Preds_*.pt files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Path to save evaluation results (default: same as preds_dir)'
    )
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=0,
        help='Which sample to use from predictions (default: 0, first sample)'
    )
    parser.add_argument(
        '--global_min',
        type=float,
        default=None,
        help='Global minimum value used for normalization (required for denormalization to original scale)'
    )
    parser.add_argument(
        '--global_max',
        type=float,
        default=None,
        help='Global maximum value used for normalization (required for denormalization to original scale)'
    )
    parser.add_argument(
        '--use_original_scale',
        action='store_true',
        default=True,
        help='Denormalize predictions and ground truth back to original scale using global min/max (default: True)'
    )
    parser.add_argument(
        '--no_original_scale',
        dest='use_original_scale',
        action='store_false',
        help='Keep data in normalized [0, 1] range instead of denormalizing to original scale'
    )
    
    args = parser.parse_args()
    
    preds_dir = Path(args.preds_dir)
    if not preds_dir.exists():
        raise ValueError(f"Directory does not exist: {preds_dir}")
    
    output_dir = Path(args.output_dir) if args.output_dir else preds_dir
    
    # Calculate metrics
    results = calculate_range_image_loss(
        preds_dir, 
        sample_idx=args.sample_idx, 
        global_min=args.global_min,
        global_max=args.global_max,
        use_original_scale=args.use_original_scale
    )
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, output_dir)


if __name__ == '__main__':
    main()

