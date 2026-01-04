import sys, torch
from pathlib import Path
sys.path.insert(0, 'stdiff/utils')
sys.path.insert(0, 'stdiff')
from dataset import KITTIRangeImageDataset, VidResize, VidToTensor
from torchvision import transforms
from models import STDiffPipeline, STDiffDiffusers
from diffusers import DDPMScheduler

CKPT_DIR = Path('/csehome/pydah/STDiffProject/STDiff_ckpts/kitti_range_64/checkpoint-17748')
DATA_DIR = Path('/scratch/pydah/kitti/processed_data')

print("Loading model...")
stdiff = STDiffDiffusers.from_pretrained(str(CKPT_DIR), subfolder='unet').eval()

# Create scheduler with training config parameters
print("Creating scheduler...")
scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule='linear',
    prediction_type='epsilon'
)
pipe = STDiffPipeline(stdiff, scheduler)

print("Loading data...")
transform = transforms.Compose([VidResize((64, 2048)), VidToTensor()])
ds = KITTIRangeImageDataset(DATA_DIR, [8,9,10], transform, train=True, val=False, num_observed_frames=5, num_predict_frames=5)
batch = ds()[0]
Vo = batch[0].unsqueeze(0)  # CPU: (1, To, C, H, W)
Vo_last_frame = Vo[:, -1:, ...]  # (1, 1, C, H, W) - keep time dimension
To = Vo.shape[1]  # number of observed frames
Tp = 5  # number of predicted frames
idx_o = torch.linspace(0, To-1, To)  # 1D tensor
idx_p = torch.linspace(To, To+Tp-1, Tp)  # 1D tensor

print(f"Input shape: Vo={Vo.shape}, Vo_last_frame={Vo_last_frame.shape}")
print(f"idx_o={idx_o}, idx_p={idx_p}")

print("Running inference...")
with torch.no_grad():
    out = pipe(Vo, Vo_last_frame, idx_o, idx_p, num_inference_steps=50, output_type='numpy')
    print(f"✓ Inference complete!")
    # Output is ImagePipelineOutput with 'images' attribute: (N*Tp, H, W, C) for numpy
    pred_images = out.images
    print(f"  Prediction shape: {pred_images.shape}")
    print(f"  Value range: [{pred_images.min():.4f}, {pred_images.max():.4f}]")
    print(f"  Mean: {pred_images.mean():.4f}, Std: {pred_images.std():.4f}")
    print(f"  Number of predicted frames: {pred_images.shape[0]}")
