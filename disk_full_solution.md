# Disk Full Error - Solution Guide

## The Problem

**Error:** `RuntimeError: PytorchStreamWriter failed writing file data/508: file write failed`

**Root Cause:** Disk is 100% full
- Filesystem: `/dev/nvme0n1p2`
- Total: 1.8T
- Used: 1.7T  
- **Available: 4.0K (basically 0!)**

**Checkpoint Size:** Each checkpoint is ~3.1 GB
- You have 14 checkpoints = ~43 GB
- Checkpoint frequency: Every 204 steps (very frequent!)

## Immediate Fix: Free Up Disk Space

### Option 1: Delete Old Checkpoints (Recommended)

**Quick cleanup - keep only latest 3:**
```bash
cd /home/anirudh/STDiffProject/STDiff_ckpts/kitti_range_64x512
# Keep only the 3 most recent checkpoints
ls -dt checkpoint-* | tail -n +4 | xargs rm -rf
```

**Or use the cleanup script:**
```bash
cd /home/anirudh/STDiffProject
./cleanup_checkpoints.sh
```

**Manual cleanup (keep specific checkpoints):**
```bash
cd /home/anirudh/STDiffProject/STDiff_ckpts/kitti_range_64x512
# Delete specific old checkpoints
rm -rf checkpoint-204 checkpoint-408 checkpoint-612
```

### Option 2: Find Other Large Files

```bash
# Find largest files/directories
du -h /home/anirudh | sort -rh | head -20

# Find large files
find /home/anirudh -type f -size +1G -exec ls -lh {} \; 2>/dev/null | head -20
```

### Option 3: Clean System Temp Files

```bash
# Clean pip cache
pip cache purge

# Clean conda cache
conda clean --all

# Clean system logs (if you have permission)
sudo journalctl --vacuum-time=7d
```

## Long-Term Solution: Reduce Checkpoint Frequency

### 1. Increase Checkpoint Interval

Edit `kitti_range_train_config.yaml`:

```yaml
Training:
    checkpointing_steps: 500  # Instead of 204 (save less frequently)
    # Or even:
    checkpointing_steps: 1000  # Save every 1000 steps
```

### 2. Keep Only Latest N Checkpoints

Modify `train_stdiff.py` to automatically delete old checkpoints:

```python
# Around line 320-324, modify checkpoint saving:
if global_step % cfg.Training.checkpointing_steps == 0:
    if accelerator.is_main_process:
        save_path = os.path.join(cfg.Env.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")
        
        # Keep only latest 3 checkpoints
        import glob
        checkpoints = sorted(glob.glob(os.path.join(cfg.Env.output_dir, "checkpoint-*")))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                import shutil
                shutil.rmtree(old_checkpoint)
                logger.info(f"Deleted old checkpoint: {old_checkpoint}")
```

### 3. Save to Different Location (if available)

If you have another disk with space:

```yaml
Env:
    output_dir: '/path/to/another/disk/STDiff_ckpts/kitti_range_64x512'
```

### 4. Use Smaller Checkpoints

Only save model weights, not full optimizer state:

```python
# In save_model_hook, modify to save only model:
def save_model_hook(models, weights, output_dir):
    if cfg.Training.use_ema:
        ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))
    
    for i, model in enumerate(models):
        # Save only model, not optimizer/scheduler
        model.save_pretrained(os.path.join(output_dir, "unet"))
        weights.pop()
```

## Recommended Config Changes

```yaml
Training:
    epochs: 400
    save_images_epochs: 4
    save_model_epochs: 2
    checkpointing_steps: 500  # Changed from 204 (saves less frequently)
    # This will create ~8 checkpoints instead of 14
```

## After Cleanup: Resume Training

If training was interrupted, you can resume from the latest checkpoint:

```yaml
Env:
    resume_ckpt: "checkpoint-2856"  # Use the latest checkpoint number
```

## Prevention

1. **Monitor disk space:**
   ```bash
   watch -n 60 'df -h /home/anirudh'
   ```

2. **Set up disk space alerts** (if possible)

3. **Use checkpoint cleanup script** regularly:
   ```bash
   # Add to cron or run manually
   ./cleanup_checkpoints.sh
   ```

## Quick Commands Summary

```bash
# Check disk space
df -h /home/anirudh

# See checkpoint sizes
du -sh /home/anirudh/STDiffProject/STDiff_ckpts/kitti_range_64x512/checkpoint-*

# Delete old checkpoints (keep latest 3)
cd /home/anirudh/STDiffProject/STDiff_ckpts/kitti_range_64x512
ls -dt checkpoint-* | tail -n +4 | xargs rm -rf

# Verify space freed
df -h /home/anirudh
```

## Expected Space Savings

- **Delete 11 old checkpoints:** ~34 GB freed
- **Reduce checkpoint frequency:** Future checkpoints will be less frequent
- **Total:** Should free up enough space to continue training

