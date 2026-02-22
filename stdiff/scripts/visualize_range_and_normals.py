#!/usr/bin/env python3
"""
Visualize ground-truth range images and their normal maps.
Saves 2-3 samples as range (top) + normal map (bottom) stacked vertically.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project paths so both stdiff and losses are importable
SCRIPT_DIR = Path(__file__).resolve().parent
STDIFF_ROOT = SCRIPT_DIR.parent  # stdiff/
PROJECT_ROOT = STDIFF_ROOT.parent  # STDiffProject/
for p in [str(PROJECT_ROOT), str(STDIFF_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def load_projection_and_normals():
    """Import normal loss utilities."""
    from losses.normal_loss import (
        get_projection_layer,
        get_projection_config,
        range_to_normals,
    )
    return get_projection_layer, get_projection_config, range_to_normals


def load_global_minmax(dataset_dir):
    """Load global min/max from cache if available."""
    try:
        from stdiff.utils.dataset import load_global_minmax_from_cache
        # Match train config: train_folder_ids, num_observed, num_predict
        gmin, gmax = load_global_minmax_from_cache(
            dataset_dir, [0, 1, 2, 3, 4, 5, 6, 7, 8], 3, 3, None
        )
        return gmin, gmax
    except Exception as e:
        print(f"Could not load cache: {e}")
        return None, None


def range_to_rgb_turbo_r(arr):
    """Convert range [0,1] to RGB using turbo_r colormap."""
    arr = np.clip(arr, 0, 1).astype(np.float32)
    cmap = plt.get_cmap("turbo_r")
    rgba = cmap(arr)
    return (rgba[..., :3] * 255).astype(np.uint8)


def normals_to_rgb(normals):
    """Convert normals (B,H,W,3) in [-1,1] to RGB uint8. nx,ny,nz -> R,G,B."""
    # normals in [-1,1] -> (n+1)/2 -> [0,1]
    rgb = (normals + 1) / 2
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    return (rgb * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize range + normal maps")
    parser.add_argument("--data_dir", type=str, default="/DATA2/common/kitti/processed_data_64x512",
                        help="KITTI range dataset directory")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: stdiff/vis_range_normals)")
    parser.add_argument("--num_images", type=int, default=3, help="Number of images (2-3)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else STDIFF_ROOT / "vis_range_normals"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_images = min(max(args.num_images, 2), 3)

    # Find .npy files
    seq_folders = sorted([f for f in data_dir.iterdir() if f.is_dir() and f.name.isdigit()])
    if not seq_folders:
        print(f"No sequence folders in {data_dir}")
        return
    range_dir = seq_folders[0] / "processed" / "range"
    if not range_dir.exists():
        print(f"Range dir not found: {range_dir}")
        return
    npy_files = sorted(range_dir.glob("*.npy"))[:num_images]

    if not npy_files:
        print(f"No .npy files in {range_dir}")
        return

    # Load global min/max
    gmin, gmax = load_global_minmax(str(data_dir))
    if gmin is None or gmax is None:
        # Compute from loaded images
        all_vals = []
        for p in npy_files:
            r = np.load(str(p))
            if r.size > 0:
                valid = r[r > 0]
                if len(valid) > 0:
                    all_vals.extend(valid.astype(np.float32).tolist())
        gmin = float(np.min(all_vals)) if all_vals else 0.0
        gmax = float(np.max(all_vals)) if all_vals else 85.0
    print(f"Using global range: [{gmin:.4f}, {gmax:.4f}]")

    # Load range images (raw meters)
    range_meters_list = []
    valid_masks = []
    for p in npy_files:
        r = np.load(str(p))
        if r.ndim > 2:
            r = r.squeeze()
        valid = r > 0
        valid_masks.append(valid)
        range_meters_list.append(r)

    H, W = range_meters_list[0].shape

    # Normalize to [0,1] for projection (range_to_normals expects [0,1] and uses gmin/gmax to denorm)
    range_01_list = []
    for r, vm in zip(range_meters_list, valid_masks):
        r_norm = np.zeros_like(r, dtype=np.float32)
        if vm.any() and gmax > gmin:
            r_norm[vm] = (r[vm] - gmin) / (gmax - gmin)
        range_01_list.append(r_norm)

    # Stack to batch tensor (N, 1, H, W)
    range_01 = np.stack([r[np.newaxis, :, :] for r in range_01_list], axis=0)
    range_01_t = torch.from_numpy(range_01).float()

    # Compute normals
    get_proj, get_cfg, range_to_normals_fn = load_projection_and_normals()
    proj_cfg = get_cfg(height=H, width=W)
    proj_layer = get_proj(proj_cfg)
    normals = range_to_normals_fn(proj_layer, range_01_t, height=H, width=W,
                                  global_min=gmin, global_max=gmax)
    # normals: (N, 3, H, W)
    normals_np = normals.permute(0, 2, 3, 1).numpy()  # (N, H, W, 3)

    # Save each sample: range on top, normal below
    for i in range(len(npy_files)):
        # Range: turbo_r colormap
        r_01 = range_01_list[i]
        r_rgb = range_to_rgb_turbo_r(r_01)
        range_pil = Image.fromarray(r_rgb, mode="RGB")

        # Normal map: (H, W, 3) -> RGB
        n_vis = normals_to_rgb(normals_np[i])
        normal_pil = Image.fromarray(n_vis, mode="RGB")

        # Stack vertically: range on top, normal below
        w, h_r = range_pil.size
        _, h_n = normal_pil.size
        combined = Image.new("RGB", (w, h_r + h_n))
        combined.paste(range_pil, (0, 0))
        combined.paste(normal_pil, (0, h_r))

        out_path = out_dir / f"range_normal_{i}.png"
        combined.save(str(out_path))
        print(f"Saved {out_path}")

    print(f"Done. Saved {len(npy_files)} images to {out_dir}")


if __name__ == "__main__":
    main()
