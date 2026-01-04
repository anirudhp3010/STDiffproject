# Ground Truth Normalization and Scaling Process

## Overview
This document explains how range image ground truth is normalized and scaled through the pipeline.

## Step-by-Step Process

### 1. Raw Data Loading (dataset.py, lines 1005-1042)
- **Input**: `.npy` files with raw range values (typically 0-85 meters)
- **Per-Image Min-Max Normalization**:
  ```python
  valid_pixels = range_img[valid_mask]  # Only valid pixels (> 0)
  min_val, max_val = valid_pixels.min(), valid_pixels.max()
  range_img_norm[valid_mask] = (range_img[valid_mask] - min_val) / (max_val - min_val)
  ```
  - **Result**: Each image normalized to [0, 1] using its own min/max
  - **Important**: This is PER-IMAGE normalization, not global!
  - Invalid pixels (≤ 0) remain as 0.0

### 2. PIL Image Conversion (line 1038-1041)
- Convert normalized [0, 1] float to uint8 [0, 255] for PIL
- `range_img_uint8 = (range_img * 255).astype(np.uint8)`

### 3. Transform Pipeline (dataset.py, line 57-58)
For KITTI_RANGE:
```python
test_transform = transforms.Compose([
    VidResize(resize_size),      # Resize to model input size
    VidToTensor(),               # PIL [0,255] → Tensor [0,1] (divides by 255)
    norm_transform               # [0,1] → [-1,1] (x * 2 - 1)
])
```

### 4. VidToTensor (dataset.py, lines 828-835)
- Converts PIL Image to PyTorch Tensor
- `transforms.ToTensor()` automatically divides by 255
- **Result**: Tensor in [0, 1] range

### 5. norm_transform (dataset.py, line 37)
```python
self.norm_transform = lambda x: x * 2. - 1.
```
- **Result**: [0, 1] → [-1, 1]
- This is the range used during training/inference

### 6. Test Script Denormalization (test_stdiff.py, lines 199-200)
```python
Vo = (Vo / 2 + 0.5).clamp(0, 1)  # [-1, 1] → [0, 1]
Vp = (Vp / 2 + 0.5).clamp(0, 1)  # [-1, 1] → [0, 1]
```
- **Result**: Ground truth saved in [0, 1] range
- This is what's stored in `Preds_*.pt` files

## Key Points

1. **Per-Image Normalization**: The original normalization is done per-image using each image's min/max, not a global 0-85 scale.

2. **Cannot Directly Reverse**: To get back to original meters, you would need:
   - The original min/max values for each image
   - OR load the original `.npy` files directly

3. **Current Saved Format**: Ground truth in `Preds_*.pt` is in [0, 1] range (after denormalization from [-1, 1])

4. **Approximation**: If you multiply by 85, you're assuming:
   - All images were normalized from a 0-85 range
   - The min was always 0 and max was always 85
   - This is an approximation and may not be accurate for all images

## For Evaluation

If you want to evaluate in original scale (meters):
- **Option 1**: Load original `.npy` files and compare directly
- **Option 2**: Use the approximation: multiply [0, 1] by 85 (or max range)
- **Option 3**: Save original min/max values during dataset loading for exact reconstruction

## Current Implementation

The `eval_range_metrics.py` script uses Option 2 (approximation):
- Scales [0, 1] → [0, 85] by multiplying by `original_scale_max` (default 85.0)
- This gives approximate meter values but may not be exact due to per-image normalization

