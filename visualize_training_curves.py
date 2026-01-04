"""
Script to visualize training curves from TensorBoard logs.
Reads TensorBoard event files and plots training metrics.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: tensorboard is not installed. Install it with: pip install tensorboard")
    exit(1)


def load_tensorboard_logs(log_dir):
    """
    Load TensorBoard logs from a directory.
    
    Args:
        log_dir: Path to directory containing TensorBoard event files
    
    Returns:
        EventAccumulator object with loaded events
    """
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        raise ValueError(f"Log directory not found: {log_dir}")
    
    # Create EventAccumulator
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    
    return ea


def extract_scalars(ea, scalar_name):
    """
    Extract scalar values from EventAccumulator.
    
    Args:
        ea: EventAccumulator object
        scalar_name: Name of the scalar to extract (e.g., 'train/loss')
    
    Returns:
        Tuple of (steps, values) arrays
    """
    if scalar_name not in ea.Tags()['scalars']:
        return None, None
    
    scalar_events = ea.Scalars(scalar_name)
    steps = [e.step for e in scalar_events]
    values = [e.value for e in scalar_events]
    
    return np.array(steps), np.array(values)


def plot_training_curves(log_dir, output_path=None, metrics=None, figsize=(12, 8)):
    """
    Plot training curves from TensorBoard logs.
    
    Args:
        log_dir: Path to directory containing TensorBoard event files
        output_path: Path to save the plot (if None, displays interactively)
        metrics: List of metric names to plot (if None, plots all available)
        figsize: Figure size tuple
    """
    # Load logs
    print(f"Loading TensorBoard logs from: {log_dir}")
    ea = load_tensorboard_logs(log_dir)
    
    # Get available scalars
    available_scalars = ea.Tags()['scalars']
    print(f"Available metrics: {available_scalars}")
    
    if len(available_scalars) == 0:
        print("No scalar metrics found in logs!")
        return
    
    # Filter metrics if specified
    if metrics is None:
        metrics_to_plot = available_scalars
    else:
        metrics_to_plot = [m for m in metrics if m in available_scalars]
        if len(metrics_to_plot) == 0:
            print(f"None of the specified metrics found. Available: {available_scalars}")
            return
    
    # Create subplots
    n_metrics = len(metrics_to_plot)
    if n_metrics == 1:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]
    elif n_metrics <= 2:
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
    elif n_metrics <= 4:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
    else:
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
    
    # Plot each metric
    for idx, metric_name in enumerate(metrics_to_plot):
        steps, values = extract_scalars(ea, metric_name)
        
        if steps is None or len(steps) == 0:
            print(f"Warning: No data for metric: {metric_name}")
            continue
        
        ax = axes[idx]
        ax.plot(steps, values, linewidth=2, label=metric_name)
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics
        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)
        final_val = values[-1]
        
        stats_text = f'Final: {final_val:.4f}\nMean: {mean_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training curves from TensorBoard logs"
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Path to directory containing TensorBoard event files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the plot (if not specified, displays interactively)"
    )
    parser.add_argument(
        "-m", "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics to plot (e.g., 'train/loss' 'train/lr'). If not specified, plots all available metrics"
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 8],
        help="Figure size (width height), default: 12 8"
    )
    
    args = parser.parse_args()
    
    plot_training_curves(
        args.log_dir,
        args.output,
        args.metrics,
        tuple(args.figsize)
    )


if __name__ == "__main__":
    main()

