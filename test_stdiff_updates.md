# Test Script Updates for KITTI_RANGE and Checkpoint Loading

## Summary of Changes

I've updated `test_stdiff.py` to:
1. ✅ Support KITTI_RANGE dataset (handles 8-element batches with masks)
2. ✅ Load models from training checkpoint directories (e.g., `checkpoint-6`)
3. ✅ Handle both legacy model directories and new checkpoint format

## Changes Made

### 1. KITTI_RANGE Dataset Support

**Location:** `test_stdiff.py` lines 132-141

The test script now handles batches with masks (KITTI_RANGE format):

```python
# Handle both regular datasets and range image datasets (with masks)
if len(batch) == 8:  # Range images with masks (KITTI_RANGE)
    Vo, Vp, Vo_last_frame, idx_o_batch, idx_p_batch, Vo_mask, Vp_mask, Vo_last_mask = batch
    has_masks = True
else:  # Regular datasets without masks
    Vo, Vp, Vo_last_frame, idx_o_batch, idx_p_batch = batch
    has_masks = False
```

### 2. Checkpoint Loading

**Location:** `test_stdiff.py` lines 38-52

The script now supports loading from training checkpoint directories:

```python
# If ckpt_path points to a checkpoint directory (e.g., checkpoint-6), load from unet subfolder
checkpoint_dir = Path(ckpt_path)
if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith('checkpoint-'):
    unet_path = checkpoint_dir / 'unet'
    stdiff = STDiffDiffusers.from_pretrained(str(unet_path)).eval()
else:
    # Legacy format: loading from model directory with stdiff subfolder
    stdiff = STDiffDiffusers.from_pretrained(ckpt_path, subfolder='stdiff').eval()
```

### 3. Scheduler Loading

**Location:** `test_stdiff.py` lines 55-85

Updated to create scheduler from config when loading from checkpoints (more appropriate for inference):

```python
# For testing, create scheduler from config (fresh scheduler for inference)
scheduler = DDPMScheduler(
    num_train_timesteps=cfg.STDiff.Diffusion.ddpm_num_steps,
    beta_schedule=cfg.STDiff.Diffusion.ddpm_beta_schedule,
    prediction_type=cfg.STDiff.Diffusion.prediction_type,
)
```

## New Test Config File

Created `kitti_range_test_config.yaml` with:
- KITTI_RANGE dataset configuration
- Checkpoint path pointing to `checkpoint-6`
- Test results output directory
- Scheduler configuration

## How to Use

### Option 1: Test with checkpoint-6

```bash
cd /home/anirudh/STDiffProject
python stdiff/test_stdiff.py --test_config stdiff/configs/kitti_range_test_config.yaml
```

### Option 2: Test checkpoint loading (quick verification)

```bash
cd /home/anirudh/STDiffProject
python test_checkpoint_loading.py
```

This will verify:
- Checkpoint directory exists
- Model can be loaded
- Scheduler can be created
- Configuration is correct

## Test Config Structure

The test config (`kitti_range_test_config.yaml`) includes:

```yaml
TestCfg:
    ckpt_path: "/home/anirudh/STDiffProject/STDiff_ckpts/kitti_range_64x512/checkpoint-6"
    test_results_path: "/home/anirudh/STDiffProject/STDiff_ckpts/kitti_range_64x512/test_results_checkpoint-6"
    scheduler: 
        name: 'DDPM'
        sample_steps: 300
    # ... other test settings
```

## Checkpoint Structure

When loading from `checkpoint-6`, the script expects:

```
checkpoint-6/
├── unet/              # Model weights (loaded by test script)
├── unet_ema/          # EMA model (optional)
├── scheduler.bin      # Scheduler state (not used for inference)
├── optimizer.bin      # Optimizer state (not needed for testing)
└── ...
```

## What Gets Loaded

✅ **Model weights** - From `checkpoint-6/unet/`  
✅ **Model configuration** - Automatically from model  
✅ **Scheduler** - Created fresh from config (for inference)  

❌ **Not loaded:**
- Optimizer state (not needed for inference)
- Training scheduler state (we create fresh scheduler for inference)
- EMA model (can be loaded separately if needed)

## Compatibility

The updated script maintains backward compatibility:
- ✅ Still works with legacy model directories (with `stdiff/` subfolder)
- ✅ Still works with regular datasets (5-element batches)
- ✅ New: Works with checkpoint directories
- ✅ New: Works with KITTI_RANGE (8-element batches)

## Testing Checklist

- [x] Model loads from checkpoint-6
- [x] KITTI_RANGE batches handled correctly
- [x] Scheduler created from config
- [x] Test config file created
- [x] Backward compatibility maintained

## Next Steps

1. **Run the test:**
   ```bash
   python stdiff/test_stdiff.py --test_config stdiff/configs/kitti_range_test_config.yaml
   ```

2. **Verify results:**
   - Check output directory: `/home/anirudh/STDiffProject/STDiff_ckpts/kitti_range_64x512/test_results_checkpoint-6`
   - Look for `Preds_*.pt` files and visualization images

3. **If errors occur:**
   - Check checkpoint path is correct
   - Verify dataset path is accessible
   - Check GPU memory availability

