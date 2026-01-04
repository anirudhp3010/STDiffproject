from PIL import Image
import numpy as np

gif_path = '/home/anirudh/STDiffProject/STDiff_ckpts/kitti_range_64x512/test_results_checkpoint-51200_seq8_old/test_examples_1_traj0/None_clip_0.gif'
gif = Image.open(gif_path)

frames = []
try:
    while True:
        frames.append(np.array(gif.copy()))
        gif.seek(gif.tell() + 1)
except EOFError:
    pass

print(f'Total frames: {len(frames)}')
print(f'Frame shape: {frames[0].shape}')

# Determine frame dimensions
if len(frames[0].shape) == 2:  # Grayscale: (H, W)
    H, W = frames[0].shape
    is_color = False
elif len(frames[0].shape) == 3:  # Color: (H, W, C)
    H, W, C = frames[0].shape
    is_color = True
else:
    raise ValueError(f"Unexpected frame shape: {frames[0].shape}")

# Calculate last third of columns
last_third_start = int(W * 2 / 3)
print(f'Last third of columns: columns {last_third_start} to {W-1} (out of {W} total columns)')

print(f'\nFrame statistics (full frame):')
for i, frame in enumerate(frames):
    print(f'  Frame {i}: min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}, std={frame.std():.2f}')

print(f'\nLast third of columns statistics (all rows, columns {last_third_start}:{W-1}):')
for i, frame in enumerate(frames):
    if is_color:
        # For color images, extract last third columns for all channels
        last_third_region = frame[:, last_third_start:, :]
    else:
        # For grayscale, extract last third columns
        last_third_region = frame[:, last_third_start:]
    
    region_min = last_third_region.min()
    region_max = last_third_region.max()
    region_mean = last_third_region.mean()
    region_std = last_third_region.std()
    
    print(f'  Frame {i}: min={region_min}, max={region_max}, mean={region_mean:.2f}, std={region_std:.2f}')
    
# Check if any frame is significantly different (white)
print(f'\nChecking for white frames (mean > 200):')
for i, frame in enumerate(frames):
    if frame.mean() > 200:
        print(f'  Frame {i} appears WHITE: mean={frame.mean():.2f}, max={frame.max()}')

