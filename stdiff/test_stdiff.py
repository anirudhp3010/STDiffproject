
from utils import LitDataModule

from utils import get_lightning_module_dataloader
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import torch
import torch.nn as nn

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, omegaconf
from einops import rearrange

from utils import visualize_batch_clips, eval_metrics, LitDataModule
from pathlib import Path
import argparse
from models import STDiffPipeline, STDiffDiffusers
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, PNDMScheduler, DDIMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--test_config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    return args.test_config

def main(cfg : DictConfig) -> None:
    accelerator = Accelerator()
    device = accelerator.device
    ckpt_path = cfg.TestCfg.ckpt_path
    r_save_path = cfg.TestCfg.test_results_path
    if not Path(r_save_path).exists():
        Path(r_save_path).mkdir(parents=True, exist_ok=True) 

    # Handle checkpoint loading - support both checkpoint directories and model directories
    # If ckpt_path points to a checkpoint directory (e.g., checkpoint-6), load from unet subfolder
    # Otherwise, load from stdiff subfolder (legacy format)
    checkpoint_dir = Path(ckpt_path)
    if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith('checkpoint-'):
        # Loading from a training checkpoint directory
        unet_path = checkpoint_dir / 'unet'
        if unet_path.exists():
            print(f"Loading model from checkpoint directory: {ckpt_path}/unet")
            stdiff = STDiffDiffusers.from_pretrained(str(unet_path)).eval()
        else:
            raise FileNotFoundError(f"Model not found in checkpoint directory: {unet_path}")
    else:
        # Legacy format: loading from model directory with stdiff subfolder
        stdiff = STDiffDiffusers.from_pretrained(ckpt_path, subfolder='stdiff').eval()
    #Print the number of parameters
    num_params = sum(p.numel() for p in stdiff.parameters() if p.requires_grad)
    print('Number of parameters is: ', num_params)
    
    # Enable gradient checkpointing if configured
    if cfg.TestCfg.gradient_checkpointing:
        # Enable gradient checkpointing on the UNet blocks
        def enable_gc_recursive(module):
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
            for child in module.children():
                enable_gc_recursive(child)
        
        enable_gc_recursive(stdiff.diffusion_unet)
        print("Gradient checkpointing enabled for inference")
    
    # Enable xformers memory efficient attention if configured
    if cfg.TestCfg.use_xformers:
        try:
            stdiff.enable_xformers_memory_efficient_attention()
            print("XFormers memory efficient attention enabled")
        except Exception as e:
            print(f"Warning: Failed to enable xformers: {e}. Continuing without xformers.")

    #init scheduler - handle both checkpoint directories and model directories
    checkpoint_dir = Path(ckpt_path)
    if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith('checkpoint-'):
        # Loading from a training checkpoint directory
        # Checkpoints save scheduler.bin directly in checkpoint dir, or we create from config
        scheduler_bin = checkpoint_dir / 'scheduler.bin'
        scheduler_dir = checkpoint_dir / 'scheduler'
        
        # For testing, we typically want a fresh scheduler with inference settings
        # Create scheduler from config (this is more appropriate for inference)
        print(f"Creating scheduler from config for inference")
        if cfg.TestCfg.scheduler.name == 'DDPM':
            scheduler = DDPMScheduler(
                num_train_timesteps=cfg.STDiff.Diffusion.ddpm_num_steps,
                beta_schedule=cfg.STDiff.Diffusion.ddpm_beta_schedule,
                prediction_type=cfg.STDiff.Diffusion.prediction_type,
            )
        elif cfg.TestCfg.scheduler.name == 'DPMMS':
            # First load base DDPM scheduler config
            base_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.STDiff.Diffusion.ddpm_num_steps,
                beta_schedule=cfg.STDiff.Diffusion.ddpm_beta_schedule,
                prediction_type=cfg.STDiff.Diffusion.prediction_type,
            )
            scheduler = DPMSolverMultistepScheduler.from_config(base_scheduler.config, solver_order=3)
        else:
            raise NotImplementedError("Scheduler is not supported")
    else:
        # Legacy format: loading from model directory
        if cfg.TestCfg.scheduler.name == 'DDPM':
            scheduler = DDPMScheduler.from_pretrained(ckpt_path, subfolder = 'scheduler')
        elif cfg.TestCfg.scheduler.name == 'DPMMS':
            scheduler = DPMSolverMultistepScheduler.from_pretrained(ckpt_path, subfolder="scheduler", solver_order=3)
        else:
            raise NotImplementedError("Scheduler is not supported")

    stdiff_pipeline = STDiffPipeline(stdiff, scheduler).to(device)
    if not accelerator.is_main_process:
        stdiff_pipeline.disable_pgbar()
    
    _, _, test_loader, pl_datamodule = get_lightning_module_dataloader(cfg)
    stdiff_pipeline, test_loader = accelerator.prepare(stdiff_pipeline, test_loader)
    
    # Get global min/max for KITTI_RANGE (for evaluation)
    global_min = None
    global_max = None
    if cfg.Dataset.name == 'KITTI_RANGE':
        global_min = getattr(pl_datamodule, 'range_image_global_min', None)
        global_max = getattr(pl_datamodule, 'range_image_global_max', None)
        if global_min is not None and global_max is not None:
            print(f"Using global min/max for evaluation: [{global_min:.6f}, {global_max:.6f}]")

    To = cfg.Dataset.test_num_observed_frames
    assert To == cfg.Dataset.num_observed_frames, 'invalid configuration'
    Tp = cfg.Dataset.test_num_predict_frames
    idx_o = torch.linspace(0, To-1 , To).to(device)
    if cfg.TestCfg.fps == 1:
        idx_p = torch.linspace(To, cfg.Dataset.num_predict_frames+To-1, cfg.Dataset.num_predict_frames).to(device)
    elif cfg.TestCfg.fps == 2:
        idx_p = torch.linspace(To, cfg.Dataset.num_predict_frames+To-1, 2*cfg.Dataset.num_predict_frames-1).to(device)
    #steps = cfg.TestCfg.fps*(cfg.Dataset.num_predict_frames-1) + 1
    #idx_p = torch.linspace(To, cfg.Dataset.num_predict_frames+To-1, steps).to(device)

    autoreg_iter = cfg.Dataset.test_num_predict_frames // cfg.Dataset.num_predict_frames
    autoreg_rem = cfg.Dataset.test_num_predict_frames % cfg.Dataset.num_predict_frames
    if autoreg_rem > 0:
        autoreg_iter = autoreg_iter + 1

    if accelerator.is_main_process:
        print('idx_o', idx_o)
        print('idx_p', idx_p)
        test_config = {'cfg': cfg, 'idx_o': idx_o.to('cpu'), 'idx_p': idx_p.to('cpu')}
        torch.save(test_config, f = Path(r_save_path).joinpath('TestConfig.pt'))
    
    def get_resume_batch_idx(r_save_path):
        save_path = Path(r_save_path)
        saved_preds = sorted(list(save_path.glob('Preds_*')))
        saved_batches = sorted([int(str(p.name).split('_')[1].split('.')[0]) for p in saved_preds])
        try:
            return saved_batches[-1]
        except IndexError:
            return -1
    resume_batch_idx = get_resume_batch_idx(r_save_path)
    print('number of test batches: ', len(test_loader))
    print('resume batch index: ', resume_batch_idx)

    #Predict and save the predictions to disk for evaluation
    with torch.no_grad():
        progress_bar = tqdm(total=len(test_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Testing...") 
        for idx, batch in enumerate(test_loader):
            if idx > resume_batch_idx: #resume test
                # Handle both regular datasets and range image datasets (with masks)
                if len(batch) == 8:  # Range images with masks (KITTI_RANGE)
                    Vo, Vp, Vo_last_frame, idx_o_batch, idx_p_batch, Vo_mask, Vp_mask, Vo_last_mask = batch
                    has_masks = True
                else:  # Regular datasets without masks
                    Vo, Vp, Vo_last_frame, idx_o_batch, idx_p_batch = batch
                    has_masks = False
                    Vo_mask = None
                    Vp_mask = None
                    Vo_last_mask = None

                # Check if mask prediction is enabled (from config)
                predict_mask = cfg.TestCfg.get('predict_mask', False) and has_masks
                
                preds = []
                preds_mask = [] if predict_mask else None
                if cfg.TestCfg.random_predict.first_pred_sample_num >= 2:
                    if predict_mask:
                        filter_first_out = stdiff_pipeline.filter_best_first_pred(
                            cfg.TestCfg.random_predict.first_pred_sample_num, Vo.clone(), 
                            Vo_last_frame, Vp[:, 0:1, ...], idx_o, idx_p, 
                            num_inference_steps = cfg.TestCfg.scheduler.sample_steps,
                            fix_init_noise=cfg.TestCfg.random_predict.fix_init_noise,
                            bs = cfg.TestCfg.random_predict.first_pred_parralle_bs,
                            predict_mask=True,
                            Vp_first_mask=Vp_mask[:, 0:1, ...] if has_masks else None
                        )
                    else:
                        filter_first_out = stdiff_pipeline.filter_best_first_pred(
                            cfg.TestCfg.random_predict.first_pred_sample_num, Vo.clone(), 
                            Vo_last_frame, Vp[:, 0:1, ...], idx_o, idx_p, 
                            num_inference_steps = cfg.TestCfg.scheduler.sample_steps,
                            fix_init_noise=cfg.TestCfg.random_predict.fix_init_noise,
                            bs = cfg.TestCfg.random_predict.first_pred_parralle_bs
                        )
                
                for i in range(cfg.TestCfg.random_predict.sample_num):
                    pred_clip = []
                    pred_clip_mask = [] if predict_mask else None
                    Vo_input = Vo.clone()
                    # Reset Vo_last_frame for each new trajectory (sample)
                    Vo_last_frame_iter = Vo_last_frame.clone()
                    for j in range(autoreg_iter):
                        if j == 0 and cfg.TestCfg.random_predict.first_pred_sample_num >= 2:
                            if predict_mask:
                                # filter_first_out has 6 values when predict_mask=True: 
                                # (best_m_first, best_first_preds, best_first_masks, idx_p, image_shape, generator)
                                best_m_first, best_first_preds, best_first_masks, idx_p, image_shape, gen = filter_first_out
                                temp_pred, temp_mask = stdiff_pipeline.pred_remainig_frames(
                                    best_m_first, best_first_preds, idx_p, image_shape, gen,
                                    fix_init_noise=cfg.TestCfg.random_predict.fix_init_noise,
                                    output_type="pil", to_cpu=False, 
                                    num_inference_steps=cfg.TestCfg.scheduler.sample_steps,
                                    predict_mask=True, best_first_masks=best_first_masks,
                                    Vo=Vo_input, idx_o=idx_o)
                            else:
                                # filter_first_out has 5 values when predict_mask=False:
                                # (best_m_first, best_first_preds, idx_p, image_shape, generator)
                                best_m_first, best_first_preds, idx_p, image_shape, gen = filter_first_out
                                temp_pred = stdiff_pipeline.pred_remainig_frames(
                                    best_m_first, best_first_preds, idx_p, image_shape, gen,
                                    fix_init_noise=cfg.TestCfg.random_predict.fix_init_noise,
                                    output_type="pil", to_cpu=False, 
                                    num_inference_steps=cfg.TestCfg.scheduler.sample_steps,
                                    Vo=Vo_input, idx_o=idx_o)
                        else:
                            if predict_mask:
                                temp_pred, temp_mask = stdiff_pipeline(
                                    Vo_input, Vo_last_frame_iter, idx_o, idx_p, 
                                    num_inference_steps = cfg.TestCfg.scheduler.sample_steps,
                                    to_cpu=False, fix_init_noise=cfg.TestCfg.random_predict.fix_init_noise,
                                    predict_mask=True, Vp_mask=Vp_mask
                                ) # Returns tuple: (image, mask)
                            else:
                                temp_pred = stdiff_pipeline(
                                    Vo_input, Vo_last_frame_iter, idx_o, idx_p, 
                                    num_inference_steps = cfg.TestCfg.scheduler.sample_steps,
                                    to_cpu=False, fix_init_noise=cfg.TestCfg.random_predict.fix_init_noise
                                ) #Torch Tensor (N, Tp, C, H, W), range [0, 1]
                        
                        pred_clip.append(temp_pred)
                        if predict_mask:
                            pred_clip_mask.append(temp_mask)
                        
                        # Convert from [0, 1] to [-1, 1] for next autoregressive iteration
                        temp_pred_clamped = temp_pred.clamp(0, 1)
                        Vo_input = temp_pred_clamped[:, -To:, ...]*2. - 1.
                        Vo_last_frame_iter = temp_pred_clamped[:, -1:, ...]*2. - 1.
                        # Clamp to [-1, 1] to ensure valid range for next iteration
                        Vo_input = Vo_input.clamp(-1, 1)
                        Vo_last_frame_iter = Vo_last_frame_iter.clamp(-1, 1)

                    pred_clip = torch.cat(pred_clip, dim = 1)
                    if autoreg_rem > 0:
                        pred_clip = pred_clip[:, 0:(autoreg_rem - cfg.Dataset.num_predict_frames), ...]
                    preds.append(pred_clip)
                    if predict_mask:
                        pred_clip_mask = torch.cat(pred_clip_mask, dim = 1)
                        if autoreg_rem > 0:
                            pred_clip_mask = pred_clip_mask[:, 0:(autoreg_rem - cfg.Dataset.num_predict_frames), ...]
                        preds_mask.append(pred_clip_mask)
                    
                preds = torch.stack(preds, 0) #(sample_num, N, Tp, C, H, W)
                preds = preds.permute(1, 0, 2, 3, 4, 5).contiguous() #(N, sample_num, num_predict_frames, C, H, W)
                preds = preds.clamp(0, 1)
                
                if predict_mask:
                    preds_mask = torch.stack(preds_mask, 0) #(sample_num, N, Tp, 1, H, W)
                    preds_mask = preds_mask.permute(1, 0, 2, 3, 4, 5).contiguous() #(N, sample_num, num_predict_frames, 1, H, W)
                    # preds_mask are raw values in [-1, 1], keep as is (no clamping)
                
                # Denormalize Vo and Vp for visualization
                Vo_vis = (Vo / 2 + 0.5).clamp(0, 1)
                Vp_vis = (Vp / 2 + 0.5).clamp(0, 1)

                g_preds = accelerator.gather(preds)
                g_Vo = accelerator.gather(Vo)
                g_Vp = accelerator.gather(Vp)
                
                # Gather masks if available (for KITTI_RANGE)
                if has_masks:
                    g_Vp_mask = accelerator.gather(Vp_mask)
                    # Vo_mask is from the original batch, need to gather it
                    g_Vo_mask = accelerator.gather(Vo_mask) if Vo_mask is not None else None
                else:
                    g_Vp_mask = None
                    g_Vo_mask = None
                
                # Gather predicted masks if predict_mask is enabled
                if predict_mask:
                    g_preds_mask = accelerator.gather(preds_mask)
                else:
                    g_preds_mask = None

                if accelerator.is_main_process:
                    dump_obj = {'Vo': g_Vo.detach().cpu(), 'g_Vp': g_Vp.detach().cpu(), 'g_Preds': g_preds.detach().cpu()}
                    # Add mask for ground truth predictions (Vp_mask) if available
                    if g_Vp_mask is not None:
                        dump_obj['g_Vp_mask'] = g_Vp_mask.detach().cpu()
                    # Add predicted masks if available
                    if g_preds_mask is not None:
                        dump_obj['g_Preds_mask'] = g_preds_mask.detach().cpu()
                    # Add global min/max for KITTI_RANGE (for evaluation)
                    if cfg.Dataset.name == 'KITTI_RANGE' and global_min is not None and global_max is not None:
                        dump_obj['global_min'] = global_min
                        dump_obj['global_max'] = global_max
                    torch.save(dump_obj, f=Path(r_save_path).joinpath(f'Preds_{idx}.pt'))
                    progress_bar.update(1)
                    for i  in range(min(cfg.TestCfg.random_predict.sample_num, 4)):
                        # Prepare masks for visualization if available
                        pred_masks_vis = None
                        gt_future_masks_vis = None
                        gt_past_masks_vis = None
                        
                        if predict_mask and g_preds_mask is not None:
                            # preds_mask: (N, sample_num, Tp, 1, H, W) - raw values in [-1, 1]
                            pred_masks_vis = g_preds_mask[:, i, ...]  # (N, Tp, 1, H, W)
                            # Normalize to [0, 1] for visualization (will be converted to binary in visualize function)
                            pred_masks_vis = (pred_masks_vis + 1.0) / 2.0
                        
                        if has_masks and g_Vp_mask is not None:
                            # Vp_mask: (N, Tp, H, W) - binary [0, 1]
                            # Add channel dimension for visualization
                            gt_future_masks_vis = g_Vp_mask.unsqueeze(2)  # (N, Tp, 1, H, W)
                        
                        if has_masks and g_Vo_mask is not None:
                            # Vo_mask: (N, To, H, W) - binary [0, 1]
                            # Add channel dimension
                            gt_past_masks_vis = g_Vo_mask.unsqueeze(2)  # (N, To, 1, H, W)
                        
                        visualize_batch_clips(
                            Vo_vis, Vp_vis, preds[:, i, ...], 
                            file_dir=Path(r_save_path).joinpath(f'test_examples_{idx}_traj{i}'),
                            pred_masks_batch=pred_masks_vis,
                            gt_future_masks_batch=gt_future_masks_vis,
                            gt_past_masks_batch=gt_past_masks_vis
                        )

                    del g_Vo
                    del g_Vp
                    del g_preds
    print("Inference finished")
    print("Start evaluation metrics")
if __name__ == '__main__':
    config_path = Path(parse_args())
    initialize(version_base=None, config_path=str(config_path.parent))
    cfg = compose(config_name=str(config_path.name))

    main(cfg)
    eval_metrics(cfg)