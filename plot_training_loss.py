#!/usr/bin/env python3
"""
Parse TensorBoard event files and plot loss over steps with average loss.
Reads only the event file with the latest timestep (does not merge multiple files).
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tempfile
import shutil

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    USE_TB = True
except ImportError:
    USE_TB = False
    try:
        import tensorflow as tf
        USE_TF = True
    except ImportError:
        USE_TF = False
        raise ImportError("Please install tensorboard: pip install tensorboard")


def get_file_timestamp(event_file):
    """
    Get timestamp for an event file, trying filename first, then mtime.
    
    Returns:
        timestamp (float), timestamp_source (str)
    """
    # Try to extract timestamp from filename
    # Format: events.out.tfevents.TIMESTAMP.hostname.pid.0
    parts = event_file.name.split('.')
    if len(parts) >= 3:
        try:
            # The timestamp is typically the 3rd part (index 2)
            timestamp = int(parts[2])
            return timestamp, "filename"
        except (ValueError, IndexError):
            pass
    
    # Fallback to file modification time
    mtime = event_file.stat().st_mtime
    return mtime, "mtime"


def find_event_file_by_step(event_dir, target_step):
    """
    Find the event file that contains a specific step number.
    
    Args:
        event_dir: Directory containing event files
        target_step: Target step number to find
    
    Returns:
        Path to the selected event file, step info string
    """
    event_path = Path(event_dir)
    event_files = list(event_path.glob('events.out.tfevents.*'))
    
    if len(event_files) == 0:
        raise ValueError(f"No event files found in directory: {event_dir}")
    
    print(f"Found {len(event_files)} event file(s) in directory: {event_dir}")
    print(f"\nFinding event file containing step {target_step}...")
    
    matching_files = []
    
    for event_file in event_files:
        try:
            # Read this file to check step range
            temp_dir = tempfile.mkdtemp()
            temp_file = Path(temp_dir) / event_file.name
            shutil.copy2(event_file, temp_file)
            
            temp_ea = EventAccumulator(temp_dir, size_guidance={'scalars': 0})
            temp_ea.Reload()
            
            scalar_tags = temp_ea.Tags().get('scalars', [])
            if len(scalar_tags) == 0:
                shutil.rmtree(temp_dir)
                continue
            
            loss_tag = None
            for tag in scalar_tags:
                if 'loss' in tag.lower():
                    loss_tag = tag
                    break
            if loss_tag is None and len(scalar_tags) > 0:
                loss_tag = scalar_tags[0]
            
            if loss_tag:
                scalars = temp_ea.Scalars(loss_tag)
                if len(scalars) > 0:
                    steps = [s.step for s in scalars]
                    min_step = min(steps)
                    max_step = max(steps)
                    
                    if min_step <= target_step <= max_step:
                        matching_files.append((event_file, min_step, max_step, 0))
                        print(f"  {event_file.name}: step range {min_step} - {max_step} ✓ CONTAINS {target_step}")
                    else:
                        # Calculate distance to target
                        if max_step < target_step:
                            dist = target_step - max_step
                        else:
                            dist = min_step - target_step
                        print(f"  {event_file.name}: step range {min_step} - {max_step}, distance = {dist}")
            
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"  Warning: Could not read {event_file.name}: {e}")
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            continue
    
    if len(matching_files) == 0:
        raise ValueError(f"No event file found containing step {target_step}")
    
    # If multiple files contain the step, use the one with the latest timestamp
    if len(matching_files) > 1:
        print(f"\nFound {len(matching_files)} files containing step {target_step}, selecting latest...")
        # Get timestamps and select latest
        best_file = None
        best_timestamp = -1
        for event_file, min_step, max_step, _ in matching_files:
            timestamp, _ = get_file_timestamp(event_file)
            if timestamp > best_timestamp:
                best_timestamp = timestamp
                best_file = (event_file, min_step, max_step)
        selected_file, min_step, max_step = best_file
    else:
        selected_file, min_step, max_step, _ = matching_files[0]
    
    step_info = f"Step range: {min_step} - {max_step} (contains step {target_step})"
    return selected_file, step_info


def find_event_file_by_timestamp(event_dir, target_timestamp=None):
    """
    Find the event file with the latest timestamp, or closest to target timestamp.
    Uses timestamp from filename (events.out.tfevents.TIMESTAMP.*) or file modification time.
    
    Args:
        event_dir: Directory containing event files
        target_timestamp: Optional target timestamp (Unix timestamp). If provided, finds file closest to this.
    
    Returns:
        Path to the selected event file, timestamp string
    """
    event_path = Path(event_dir)
    event_files = list(event_path.glob('events.out.tfevents.*'))
    
    if len(event_files) == 0:
        raise ValueError(f"No event files found in directory: {event_dir}")
    
    print(f"Found {len(event_files)} event file(s) in directory: {event_dir}")
    
    # Collect all files with their timestamps
    file_timestamps = []
    for event_file in event_files:
        timestamp, source = get_file_timestamp(event_file)
        file_timestamps.append((event_file, timestamp, source))
    
    # Sort by timestamp for display
    file_timestamps.sort(key=lambda x: x[1])
    
    if target_timestamp is not None:
        print(f"\nFinding event file closest to target timestamp: {target_timestamp}")
        # Find file with timestamp closest to target
        closest_file = None
        min_diff = float('inf')
        
        for event_file, timestamp, source in file_timestamps:
            diff = abs(timestamp - target_timestamp)
            print(f"  {event_file.name}: timestamp = {timestamp} ({source}), diff = {diff:.1f}")
            if diff < min_diff:
                min_diff = diff
                closest_file = (event_file, timestamp, source)
        
        if closest_file is None:
            raise ValueError("Could not find any event file")
        
        selected_file, selected_timestamp, selected_source = closest_file
        print(f"\nSelected event file closest to target: {selected_file.name}")
        print(f"  Timestamp: {selected_timestamp} ({selected_source})")
        print(f"  Difference from target: {min_diff:.1f} seconds")
    else:
        print("\nFinding event file with latest timestamp...")
        # Find file with latest timestamp
        latest_file = None
        latest_timestamp = -1
        latest_source = None
        
        for event_file, timestamp, source in file_timestamps:
            print(f"  {event_file.name}: timestamp = {timestamp} ({source})")
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_source = source
                latest_file = event_file
        
        if latest_file is None:
            raise ValueError("Could not find any event file")
        
        selected_file = latest_file
        selected_timestamp = latest_timestamp
        selected_source = latest_source
        print(f"\nSelected event file with latest timestamp: {selected_file.name}")
        print(f"  Timestamp: {selected_timestamp} ({selected_source})")
    
    # Show timestamp in readable format
    import datetime
    if selected_timestamp > 1e10:  # Likely a Unix timestamp
        dt = datetime.datetime.fromtimestamp(selected_timestamp)
        timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        timestamp_str = str(selected_timestamp)
    
    return selected_file, timestamp_str


def parse_event_files(event_dir, target_timestamp=None, target_step=None):
    """
    Parse TensorBoard event files and extract step, loss, and learning rate.
    If directory is provided, uses only the file with the latest timestamp, closest to target timestamp, or containing target step.
    
    Args:
        event_dir: Path to directory containing event files or a single event file
        target_timestamp: Optional target timestamp (Unix timestamp) to find closest file
        target_step: Optional target step number to find file containing this step
    
    Returns:
        steps: array of step numbers
        losses: array of loss values
        lrs: array of learning rates (if available)
    """
    event_path = Path(event_dir)
    
    # If it's a directory, find the file based on criteria
    if event_path.is_dir():
        if target_step is not None:
            # Find file containing the target step
            latest_file, info_str = find_event_file_by_step(event_path, target_step)
            print(f"\nSelected event file: {latest_file.name}")
            print(f"  {info_str}")
        else:
            # Find file by timestamp
            latest_file, timestamp_str = find_event_file_by_timestamp(event_path, target_timestamp=target_timestamp)
        
        print(f"  File size: {latest_file.stat().st_size / (1024 * 1024):.2f} MB")
        
        # Create a temporary directory with only the selected file
        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / latest_file.name
        shutil.copy2(latest_file, temp_file)
        log_dir = temp_dir
        cleanup_temp = True
        
    else:
        # Single file
        if not event_path.exists():
            raise FileNotFoundError(f"Event file not found: {event_dir}")
        log_dir = str(event_path.parent)
        cleanup_temp = False
        print(f"Reading event file: {event_dir}")
    
    # Use EventAccumulator to read the event file
    if not USE_TB:
        raise ImportError("TensorBoard not available. Please install: tensorboard")
    
    try:
        # Increase size limit to read all events
        size_guidance = {
            'scalars': 0,  # 0 means load all scalars
        }
        
        print(f"\nLoading event file...")
        ea = EventAccumulator(log_dir, size_guidance=size_guidance)
        ea.Reload()
        
        # Get available scalar tags
        scalar_tags = ea.Tags()['scalars']
        print(f"\nAvailable scalar tags: {scalar_tags}")
        
        # Try to find loss and learning rate tags
        loss_tag = None
        lr_tag = None
        
        # Common loss tag names
        for tag in scalar_tags:
            if 'loss' in tag.lower() and loss_tag is None:
                loss_tag = tag
            if ('lr' in tag.lower() or 'learning_rate' in tag.lower()) and lr_tag is None:
                lr_tag = tag
        
        if loss_tag is None:
            # If no loss tag found, try to use the first scalar
            if len(scalar_tags) > 0:
                loss_tag = scalar_tags[0]
                print(f"Warning: No 'loss' tag found. Using first available tag: {loss_tag}")
            else:
                raise ValueError("No scalar data found in event files!")
        
        print(f"\nUsing loss tag: {loss_tag}")
        if lr_tag:
            print(f"Using learning rate tag: {lr_tag}")
        
        # Extract loss data
        loss_scalars = ea.Scalars(loss_tag)
        print(f"Extracted {len(loss_scalars)} scalar entries from event file")
        
        # Convert to arrays
        steps = np.array([s.step for s in loss_scalars])
        losses = np.array([s.value for s in loss_scalars])
        
        # Handle duplicates if any (should be minimal with single file)
        if len(np.unique(steps)) < len(steps):
            print(f"Detected {len(steps) - len(np.unique(steps))} duplicate steps")
            # Create a mapping: step -> value, keeping last occurrence
            step_to_data = {}
            for step, loss in zip(steps, losses):
                step_to_data[step] = loss
            
            # Sort by step and extract unique values
            unique_steps = sorted(step_to_data.keys())
            steps = np.array(unique_steps)
            losses = np.array([step_to_data[s] for s in unique_steps])
            print(f"After deduplication: {len(steps)} unique steps")
        
        # Extract learning rate data if available
        lrs = None
        if lr_tag:
            try:
                lr_scalars = ea.Scalars(lr_tag)
                # Match LR to steps (might have different step counts)
                lr_steps = np.array([s.step for s in lr_scalars])
                lr_values = np.array([s.value for s in lr_scalars])
                
                # Handle duplicates in LR data too
                if len(np.unique(lr_steps)) < len(lr_steps):
                    lr_step_to_value = {}
                    for step, val in zip(lr_steps, lr_values):
                        lr_step_to_value[step] = val
                    lr_steps = np.array(sorted(lr_step_to_value.keys()))
                    lr_values = np.array([lr_step_to_value[s] for s in lr_steps])
                
                # Interpolate LR to match loss steps
                if len(lr_steps) > 0:
                    lrs = np.interp(steps, lr_steps, lr_values)
            except KeyError:
                print(f"Warning: Learning rate tag '{lr_tag}' not found. Skipping LR plot.")
        
        print(f"\nFinal extracted data points: {len(steps)}")
        print(f"Step range: {steps[0]} - {steps[-1]}")
        
        # Clean up temporary directory if created
        if cleanup_temp:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory")
        
        return steps, losses, lrs
        
    except Exception as e:
        # Clean up temp dir on error
        if cleanup_temp:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        raise


def calculate_average_loss(losses, window_size=100):
    """
    Calculate rolling average of loss over a window.
    
    Args:
        losses: array of loss values
        window_size: number of steps to average over
    
    Returns:
        average_losses: array of averaged loss values
    """
    if len(losses) < window_size:
        # If we have fewer points than window size, just return mean
        return np.full_like(losses, np.mean(losses))
    
    # Use convolution for rolling average
    kernel = np.ones(window_size) / window_size
    averaged = np.convolve(losses, kernel, mode='valid')
    
    # Pad the beginning with the first averaged value
    padding = window_size - 1
    padded = np.concatenate([np.full(padding, averaged[0]), averaged])
    
    return padded


def plot_training_curves(steps, losses, lrs, avg_losses, output_path, window_size=100):
    """
    Create plots for loss, average loss, and learning rate.
    """
    # Create subplots - 2 if we have LR, 1 if not
    if lrs is not None and len(lrs) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        ax2 = None
    
    # Filter out NaN values for plotting
    valid_mask = ~np.isnan(losses)
    valid_steps = steps[valid_mask]
    valid_losses = losses[valid_mask]
    valid_avg_losses = avg_losses[valid_mask]
    
    # Plot 1: Loss and Average Loss
    if len(valid_losses) > 0:
        ax1.plot(valid_steps, valid_losses, alpha=0.3, color='blue', label='Loss (raw)', linewidth=0.5)
        ax1.plot(valid_steps, valid_avg_losses, color='red', label=f'Average Loss (window={window_size})', linewidth=2)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Over Steps', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale often better for loss visualization
    
    # Filter out NaN values for statistics
    valid_mask = ~np.isnan(losses)
    valid_steps = steps[valid_mask]
    valid_losses = losses[valid_mask]
    
    if len(valid_losses) == 0:
        print("Warning: All loss values are NaN!")
        return
    
    # Add statistics text
    min_loss_idx = np.argmin(valid_losses)
    min_loss_step = valid_steps[min_loss_idx]
    min_loss_val = valid_losses[min_loss_idx]
    final_avg_loss = avg_losses[-1] if not np.isnan(avg_losses[-1]) else np.nanmean(avg_losses)
    
    stats_text = f'Min Loss: {min_loss_val:.6f} at step {min_loss_step}\n'
    stats_text += f'Final Avg Loss: {final_avg_loss:.6f}\n'
    stats_text += f'Total Steps: {len(steps)}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Learning Rate (if available)
    if ax2 is not None and lrs is not None and len(lrs) > 0:
        ax2.plot(steps, lrs, color='green', linewidth=1.5)
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Steps: {len(steps)}")
    print(f"Valid Steps (non-NaN): {len(valid_losses)}")
    print(f"Step Range: {steps[0]} - {steps[-1]}")
    print(f"\nLoss Statistics:")
    if len(valid_losses) > 0:
        print(f"  Initial Loss: {valid_losses[0]:.6f}")
        print(f"  Final Loss: {valid_losses[-1]:.6f}")
        print(f"  Minimum Loss: {min_loss_val:.6f} (at step {min_loss_step})")
        print(f"  Mean Loss: {np.nanmean(losses):.6f}")
        print(f"  Median Loss: {np.nanmedian(losses):.6f}")
        print(f"  Final Average Loss (window={window_size}): {final_avg_loss:.6f}")
    else:
        print("  Warning: All loss values are NaN!")
    if lrs is not None and len(lrs) > 0:
        print(f"\nLearning Rate Statistics:")
        print(f"  Initial LR: {lrs[0]:.6e}")
        print(f"  Final LR: {lrs[-1]:.6e}")
        print(f"  Mean LR: {np.mean(lrs):.6e}")
    else:
        print("\nLearning Rate: Not available in event files")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Parse TensorBoard event files and plot loss curves')
    parser.add_argument('--event_dir', type=str, 
                       default='/home/anirudh/STDiffProject/STDiff_ckpts/kitti_range_64x512/logs/train_stdiff',
                       help='Path to directory containing TensorBoard event files (or single event file)')
    parser.add_argument('--output', type=str,
                       default='/home/anirudh/STDiffProject/training_loss_plot.png',
                       help='Output path for the plot PNG file')
    parser.add_argument('--window_size', type=int, default=100,
                       help='Window size for rolling average (default: 100)')
    parser.add_argument('--timestamp', type=float, default=None,
                       help='Target timestamp (Unix timestamp) to select closest event file. If not provided, uses latest file.')
    parser.add_argument('--step', type=int, default=None,
                       help='Target step number to find event file containing this step. Takes precedence over --timestamp.')
    
    args = parser.parse_args()
    
    # Parse event files
    try:
        steps, losses, lrs = parse_event_files(args.event_dir, target_timestamp=args.timestamp, target_step=args.step)
    except Exception as e:
        print(f"Error parsing event files: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if len(steps) == 0:
        print("Error: No training steps found in event files!")
        return
    
    # Calculate average loss
    print(f"\nCalculating rolling average with window size: {args.window_size}")
    avg_losses = calculate_average_loss(losses, window_size=args.window_size)
    
    # Create plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_training_curves(steps, losses, lrs, avg_losses, output_path, args.window_size)


if __name__ == '__main__':
    main()
