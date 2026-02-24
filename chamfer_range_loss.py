#!/usr/bin/env python3
"""
Chamfer distance between ground truth and predicted range images.

Given a folder of test results (e.g. test_results_nomask_checkpoint-319200_DDPM_50steps),
each Preds_*.pt contains:
  - g_Vp: 5D (batch, time_frames, channel, h, w) e.g. (1, 3, 1, 64, 512)
  - g_Preds: 6D (batch, time_frames, num_predictions, channel, h, w) e.g. (1, 3, 3, 1, 64, 512)
            or 5D (batch, time_frames, channel, h, w). From 6D we always take first prediction.
  - g_Vp_mask: (batch, time_frames, h, w) - bool, True = valid LiDAR return (optional).
  - g_Preds_mask: (batch, sample_num, time_frames, 1, h, w) - float 0/1, predicted validity (optional).
  - global_min, global_max for denormalization.

The resulting range image (64, 512) has values in [global_min, global_max] in meters (lidar range).

This script:
  1. Loads range images and masks from .pt files
  2. Denormalizes and applies (x+1)/2 for GT
  3. Projects to 3D using spherical projection — GT uses g_Vp_mask, Pred uses g_Preds_mask (only valid pixels)
  4. Computes Chamfer distance between corresponding GT and Pred point clouds (per viewpoint)
"""

import os
import glob
import argparse
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from utils import projection


def chamfer_distance_custom(p1, p2):
    """
    Custom Chamfer Distance (symmetric, mean of min squared L2 distances).
    Same as in chamfer_compare.py; no PyTorch3D required.

    Args:
        p1: tensor of shape (batch, N, 3)
        p2: tensor of shape (batch, M, 3)
    Returns:
        loss: scalar
    """
    p1_exp = p1.unsqueeze(2)   # (batch, N, 1, 3)
    p2_exp = p2.unsqueeze(1)   # (batch, 1, M, 3)
    dist = torch.sum((p1_exp - p2_exp) ** 2, dim=-1)   # (batch, N, M)
    min_dist_p1_to_p2, _ = torch.min(dist, dim=2)   # (batch, N)
    min_dist_p2_to_p1, _ = torch.min(dist, dim=1)   # (batch, M)
    loss = torch.mean(min_dist_p1_to_p2) + torch.mean(min_dist_p2_to_p1)
    return loss


# Same as chamfer_loss.py / range projection
HEIGHT = 64
WIDTH = 512
FOV_UP = 3.0
FOV_DOWN = -25.0
MAX_POINTS_CHAMFER = 40000

# Default range for denormalization when not in .pt (e.g. test_results_nomask).
# After denormalization, the 64x512 range image values are in [DEFAULT_GMIN, DEFAULT_GMAX] meters (lidar range).
DEFAULT_GMIN = 1.149953
DEFAULT_GMAX = 81.352325


def get_projection_layer():
    cfg = {
        "DATA_CONFIG": {
            "FOV_UP": FOV_UP,
            "FOV_DOWN": FOV_DOWN,
            "WIDTH": WIDTH,
            "HEIGHT": HEIGHT,
        }
    }
    return projection(cfg)


def load_pred_file(path, device="cpu"):
    data = torch.load(path, map_location=device)
    return data


def _range_str(arr):
    """Return min/max string for an array."""
    return f"min={np.nanmin(arr):.6f} max={np.nanmax(arr):.6f}"


def extract_range_frames(data, global_min=None, global_max=None, verbose=False):
    """
    Extract and denormalize GT and Pred range images (one per time frame).
    When g_Vp_mask is available, also extract per-frame validity masks.
    Axis semantics:
      - g_Vp 5D: (batch, time_frames, channel, h, w)
      - g_Preds 6D: (batch, time_frames, num_predictions, channel, h, w) -> take first prediction
      - g_Preds 5D: (batch, time_frames, channel, h, w)
      - g_Vp_mask: (batch, time_frames, h, w) - True = valid
    Returns:
        gt_ranges: list of arrays (H, W), one per time frame
        pred_ranges: list of arrays (H, W), one per time frame
        gt_masks: list of (H, W) bool arrays for GT validity, or None
        pred_masks: list of (H, W) bool arrays for Pred validity, or None
    """
    gt_ranges = []
    pred_ranges = []
    gt_masks = None
    pred_masks = None

    if not isinstance(data, dict):
        return gt_ranges, pred_ranges, gt_masks, pred_masks

    g_vp = data.get("g_Vp")
    g_preds = data.get("g_Preds")
    g_vp_mask = data.get("g_Vp_mask")
    g_preds_mask = data.get("g_Preds_mask")
    gmin = data.get("global_min")
    gmax = data.get("global_max")

    if gmin is not None and isinstance(gmin, (torch.Tensor, np.ndarray)):
        gmin = float(gmin.item() if hasattr(gmin, "item") else gmin)
    if gmax is not None and isinstance(gmax, (torch.Tensor, np.ndarray)):
        gmax = float(gmax.item() if hasattr(gmax, "item") else gmax)

    if global_min is not None:
        gmin = global_min
    if global_max is not None:
        gmax = global_max
    # Use default range when not in file and not passed in
    if gmin is None:
        gmin = DEFAULT_GMIN
    if gmax is None:
        gmax = DEFAULT_GMAX

    if g_vp is None or g_preds is None:
        return gt_ranges, pred_ranges, gt_masks, pred_masks

    g_vp = g_vp.detach().cpu().numpy() if isinstance(g_vp, torch.Tensor) else np.asarray(g_vp)
    g_preds = g_preds.detach().cpu().numpy() if isinstance(g_preds, torch.Tensor) else np.asarray(g_preds)
    if g_vp_mask is not None:
        g_vp_mask = g_vp_mask.detach().cpu().numpy() if isinstance(g_vp_mask, torch.Tensor) else np.asarray(g_vp_mask)
        g_vp_mask = g_vp_mask.astype(bool)  # True = valid
    if g_preds_mask is not None:
        g_preds_mask = g_preds_mask.detach().cpu().numpy() if isinstance(g_preds_mask, torch.Tensor) else np.asarray(g_preds_mask)
        g_preds_mask = (g_preds_mask > 0.5).astype(bool)  # Threshold to bool: True = valid

    if verbose:
        print("\n" + "=" * 60)
        print("EXTRACT_RANGE_FRAMES — raw tensors (after .numpy())")
        print("=" * 60)
        print(f"  g_Vp    shape = {g_vp.shape}   {_range_str(g_vp)}")
        print(f"  g_Preds shape = {g_preds.shape}   {_range_str(g_preds)}")
        print(f"  global_min = {gmin}, global_max = {gmax} (used for denormalization)")

    # 5D: (batch, time_frames, channel, h, w)
    # 6D layout A: (batch, time_frames, num_predictions, channel, h, w) -> pred = [0, t, 0, 0]
    # 6D layout B: (batch, sample_num, time_frames, channel, h, w) from test_stdiff when sample_num=1 -> pred = [0, 0, t, 0]
    n_time = g_vp.shape[1]

    # Detect 6D layout: time at axis 1 (layout A) or axis 2 (layout B)
    # Layout A: (batch, time, num_predictions, C, H, W) e.g. (1, 3, 3, 1, 64, 512)
    # Layout B: (batch, sample_num, time, C, H, W) e.g. (1, 1, 3, 1, 64, 512) when sample_num=1
    if g_preds.ndim == 6:
        if g_preds.shape[1] == n_time:
            pred_layout = "batch, time, num_predictions, C, H, W"  # layout A
        elif g_preds.shape[2] == n_time:
            pred_layout = "batch, sample_num, time, C, H, W"  # layout B
        else:
            pred_layout = "batch, time, num_predictions, C, H, W"  # fallback
    else:
        pred_layout = None

    if g_vp_mask is not None:
        gt_masks = []
    if g_preds_mask is not None:
        pred_masks = []

    for t in range(n_time):
        # --- GT: slice to one frame ---
        gt = g_vp[0, t, 0]  # (h, w) = (64, 512)
        if verbose and t == 0:
            print("\n--- Ground Truth (GT), time frame 0 ---")
            print(f"  Step 1  Slice: gt = g_vp[0, t, 0]")
            print(f"          shape = {gt.shape}   {_range_str(gt)}   (raw model/disk value range, typically [-1, 1])")

        # Step 2: (x+1)/2  -> map [-1,1] to [0,1]
        gt = ((gt + 1.0) / 2.0).astype(np.float32)
        if verbose and t == 0:
            print(f"  Step 2  Transform: (x+1)/2  (normalized to [0, 1])")
            print(f"          shape = {gt.shape}   {_range_str(gt)}")

        # Step 3: denormalize to meters
        if gmin is not None and gmax is not None:
            gt = gt * (gmax - gmin) + gmin
        if verbose and t == 0:
            print(f"  Step 3  Denormalize: x * (gmax - gmin) + gmin  -> range in meters [gmin, gmax]")
            print(f"          shape = {gt.shape}   {_range_str(gt)}")

        # --- Pred: slice then denormalize only (no (x+1)/2) ---
        if g_preds.ndim == 6:
            if pred_layout == "batch, sample_num, time, C, H, W":
                pred = g_preds[0, 0, t, 0]  # (batch, sample_num, time, C, H, W) -> (h, w)
            else:
                pred = g_preds[0, t, 0, 0]  # (batch, time, num_predictions, C, H, W) -> (h, w)
        else:
            pred = g_preds[0, t, 0]  # 5D: (h, w)

        if verbose and t == 0:
            print("\n--- Prediction (Pred), time frame 0, first prediction ---")
            print(f"  Step 1  Slice: pred = g_preds[0, 0, t, 0] or [0, t, 0, 0] (6D) or g_preds[0, t, 0] (5D)")
            print(f"          shape = {pred.shape}   {_range_str(pred)}")
            print(f"          (no (x+1)/2 applied to Pred — only to GT)")

        pred = pred.astype(np.float32)
        if gmin is not None and gmax is not None:
            pred = pred * (gmax - gmin) + gmin
        if verbose and t == 0:
            print(f"  Step 2  Denormalize: x * (gmax - gmin) + gmin  -> range in meters [gmin, gmax]")
            print(f"          shape = {pred.shape}   {_range_str(pred)}")

        # Step 4: replace NaN/inf
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Step 5: apply masks and invalid-pixel handling
        # g_Vp_mask: (batch, time_frames, h, w) -> GT validity
        # g_Preds_mask: (batch, sample_num, time_frames, 1, h, w) -> Pred validity
        gt_mask_t = g_vp_mask[0, t] if g_vp_mask is not None else None
        pred_mask_t = None
        if g_preds_mask is not None:
            # g_Preds_mask: 6D (N, sample_num, Tp, 1, H, W) or 5D (N, Tp, 1, H, W)
            if g_preds_mask.ndim == 6:
                if g_preds_mask.shape[2] == n_time:
                    pred_mask_t = g_preds_mask[0, 0, t, 0]  # layout B: (batch, sample_num, time, C, H, W)
                else:
                    pred_mask_t = g_preds_mask[0, t, 0, 0]  # layout A
            else:
                pred_mask_t = g_preds_mask[0, t, 0] if g_preds_mask.ndim == 5 else g_preds_mask[0, t]
            pred_masks.append(pred_mask_t)
        if gt_mask_t is not None:
            gt_masks.append(gt_mask_t)

        if gt_mask_t is not None:
            gt = np.where(gt_mask_t, np.clip(gt, 1e-6, None).astype(np.float32), 0.0).astype(np.float32)
        else:
            gt = np.clip(gt, 1e-6, None).astype(np.float32)
        # Pred: use g_Preds_mask when available; else fall back to g_Vp_mask for fair comparison
        pm = pred_mask_t if pred_mask_t is not None else gt_mask_t
        if pm is not None:
            pred = np.where(pm, np.clip(pred, 1e-6, None).astype(np.float32), 0.0).astype(np.float32)
        else:
            pred = np.clip(pred, 1e-6, None).astype(np.float32)

        if verbose and t == 0:
            print("\n--- Both GT and Pred ---")
            if gt_mask_t is not None or pred_mask_t is not None:
                n_gt = int(np.sum(gt_mask_t)) if gt_mask_t is not None else gt.size
                n_pred = int(np.sum(pred_mask_t)) if pred_mask_t is not None else pred.size
                print(f"  Step 5  GT: g_Vp_mask {n_gt} valid | Pred: g_Preds_mask {n_pred} valid")
            else:
                print(f"  Step 5  No masks in file; projecting all pixels (clip invalid to 1e-6)")
            print(f"          GT   shape = {gt.shape}   {_range_str(gt)}")
            print(f"          Pred shape = {pred.shape}   {_range_str(pred)}")
            print("=" * 60 + "\n")

        gt_ranges.append(gt)
        pred_ranges.append(pred)

    return gt_ranges, pred_ranges, gt_masks, pred_masks


def range_to_pointcloud(proj_layer, range_img, device="cpu"):
    """
    Convert a single range image (H, W) to 3D point cloud (N, 3).
    Invalid pixels (<= 0) are excluded.
    """
    if isinstance(range_img, np.ndarray):
        range_t = torch.from_numpy(range_img.astype(np.float32))
    else:
        range_t = range_img.float()

    # Projection layer uses internal tensors on CPU; keep on same device as range_t for consistency
    points = proj_layer.get_valid_points_from_range_view(range_t)
    if points.numel() == 0:
        return torch.zeros(0, 3, device=device, dtype=torch.float32)
    points = points.to(device)
    return points


def downsample_point_cloud(pc, num_points=MAX_POINTS_CHAMFER):
    if pc.shape[0] <= num_points:
        return pc
    idx = torch.randperm(pc.shape[0], device=pc.device)[:num_points]
    return pc[idx]


def chamfer_two_pointclouds(p1, p2, device, downsample_to=MAX_POINTS_CHAMFER):
    """
    Downsample then compute Chamfer using custom function (no PyTorch3D needed).
    p1, p2: (N, 3) tensors on device. Returns scalar chamfer distance (symmetric).
    Lower downsample_to to reduce GPU memory.
    """
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        return float("nan")

    p1 = downsample_point_cloud(p1, downsample_to)
    p2 = downsample_point_cloud(p2, downsample_to)

    p1 = p1.unsqueeze(0).to(device)   # (1, N, 3)
    p2 = p2.unsqueeze(0).to(device)

    loss = chamfer_distance_custom(p1, p2)
    return loss.item()


def process_one_file(pred_path, proj_layer, device, sample_id=None, max_points=MAX_POINTS_CHAMFER, debug=False):
    """
    Load one Preds_*.pt, extract 3 GT and 3 Pred range images, project to 3D, compute chamfer per VP.
    When g_Vp_mask is present, only valid pixels are projected. Returns list of 3 chamfer distances.
    """
    data = load_pred_file(pred_path, device)
    gt_ranges, pred_ranges, gt_masks, pred_masks = extract_range_frames(data, verbose=debug)

    if len(gt_ranges) == 0 or len(pred_ranges) == 0:
        if debug:
            print("[debug] No gt or pred ranges: check g_Vp / g_Preds in .pt")
        return [], sample_id

    chamfers = []
    for vp_idx in range(len(gt_ranges)):
        pc_gt = range_to_pointcloud(proj_layer, gt_ranges[vp_idx], device)
        pc_pred = range_to_pointcloud(proj_layer, pred_ranges[vp_idx], device)
        if debug:
            gt_r, pred_r = gt_ranges[vp_idx], pred_ranges[vp_idx]
            print(f"[debug] Frame{vp_idx} gt range: shape={gt_r.shape} min={gt_r.min():.4f} max={gt_r.max():.4f} | "
                  f"pc_gt={pc_gt.shape[0]} pts, pc_pred={pc_pred.shape[0]} pts")

        try:
            cd = chamfer_two_pointclouds(pc_gt, pc_pred, device, downsample_to=max_points)
        except Exception as e:
            if debug:
                print(f"[debug] Frame{vp_idx} chamfer error: {e}")
            cd = float("nan")
        chamfers.append(cd)

    if sample_id is None:
        sample_id = os.path.basename(pred_path)
    return chamfers, sample_id


def main():
    parser = argparse.ArgumentParser(
        description="Chamfer distance between GT and predicted range images (after projecting to 3D)"
    )
    parser.add_argument(
        "folder",
        type=str,
        default= "./STDiff_ckpts/kitti_range_64x512_mask/test_results_wmask_36150_DDPM_50steps_tds_all",#"test_results_nomask_checkpoint-332000_DDPM_50steps",#"test_results_nomask_checkpoint-319200_DDPM_50steps",'test_results_checkpoint-70400_50steps'
        nargs="?",
        help="Folder containing Preds_*.pt files",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of Preds files to evaluate (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for chamfer computation",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=MAX_POINTS_CHAMFER,
        help=f"Max points per cloud to reduce GPU memory (default {MAX_POINTS_CHAMFER}). Lower if OOM.",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Print step-by-step transformations (shapes, ranges), then shapes and point counts for first file.")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return

    pred_files = sorted(glob.glob(os.path.join(folder, "Preds_*.pt")))
    if not pred_files:
        print(f"No Preds_*.pt files in {folder}")
        return

    if args.max_samples is not None:
        pred_files = pred_files[: args.max_samples]

    device = torch.device(args.device)
    proj_layer = get_projection_layer()

    max_points = args.max_points
    # Check if first file has masks (for user feedback)
    try:
        first_data = torch.load(pred_files[0], map_location="cpu", weights_only=True)
    except TypeError:
        first_data = torch.load(pred_files[0], map_location="cpu")
    has_gt_mask = "g_Vp_mask" in first_data
    has_pred_mask = "g_Preds_mask" in first_data

    print(f"Using device: {device}")
    print(f"Evaluating {len(pred_files)} files from {folder}")
    print(f"Chamfer: custom (mean min-squared distance), no PyTorch3D required")
    print(f"Max points per cloud: {max_points} (lower if GPU OOM)")
    print(f"Projection: {HEIGHT}x{WIDTH}, FOV [{FOV_DOWN}, {FOV_UP}] deg")
    if has_gt_mask and has_pred_mask:
        print(f"Masks: GT=g_Vp_mask, Pred=g_Preds_mask — only valid pixels projected")
    elif has_gt_mask:
        print(f"Masks: g_Vp_mask only (GT+Pred use same) — only valid pixels projected")
    elif has_pred_mask:
        print(f"Masks: g_Preds_mask only — Pred masked")
    else:
        print(f"Masks: not in .pt — projecting all pixels")
    print()

    # Accumulate chamfer per VP across all files
    all_vp0 = []
    all_vp1 = []
    all_vp2 = []

    for path in tqdm(pred_files, desc="Chamfer eval", unit="file"):
        debug_this = args.debug and (path == pred_files[0])
        chamfers, sid = process_one_file(path, proj_layer, device, max_points=max_points, debug=debug_this)
        if len(chamfers) == 0:
            continue
        if len(chamfers) >= 1:
            all_vp0.append(chamfers[0])
        if len(chamfers) >= 2:
            all_vp1.append(chamfers[1])
        if len(chamfers) >= 3:
            all_vp2.append(chamfers[2])

    def mean_valid(arr):
        arr = [x for x in arr if not (isinstance(x, float) and np.isnan(x))]
        return np.mean(arr) if arr else float("nan")

    print("=" * 50)
    print("Chamfer distance (range image -> 3D point cloud)")
    print("=" * 50)
    print(f"{'Viewpoint':<12} | {'Mean Chamfer (m)':<18} | Count")
    print("-" * 50)
    if all_vp0:
        print(f"{'VP 0':<12} | {mean_valid(all_vp0):<18.6f} | {len(all_vp0)}")
    if all_vp1:
        print(f"{'VP 1':<12} | {mean_valid(all_vp1):<18.6f} | {len(all_vp1)}")
    if all_vp2:
        print(f"{'VP 2':<12} | {mean_valid(all_vp2):<18.6f} | {len(all_vp2)}")
    print("=" * 50)

    if all_vp0 and all_vp1 and all_vp2:
        overall = mean_valid(all_vp0 + all_vp1 + all_vp2)
        print(f"Overall (all VPs): {overall:.6f}")
    print()


if __name__ == "__main__":
    main()