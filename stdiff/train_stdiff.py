import glob
import inspect
import logging
import math
import os
import shutil
import sys
from pathlib import Path

# Set memory management to reduce fragmentation BEFORE importing torch
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

import accelerate
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from packaging import version
from tqdm.auto import tqdm
import argparse
import gc

import diffusers
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available


import hydra
from hydra import compose, initialize
from hydra import initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from utils import get_lightning_module_dataloader
from models import STDiffDiffusers, STDiffPipeline

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--train_config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    return args.train_config

def main(cfg : DictConfig) -> None:
    logging_dir = os.path.join(cfg.Env.output_dir, 'logs')

    accelerator_project_config = ProjectConfiguration(total_limit=cfg.Training.epochs // cfg.Training.save_model_epochs)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.Training.gradient_accumulation_steps,
        mixed_precision=cfg.Training.mixed_precision,
        log_with=cfg.Env.logger,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if cfg.Env.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if cfg.Training.use_ema:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if cfg.Training.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), STDiffDiffusers)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = STDiffDiffusers.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.Env.output_dir is not None:
            os.makedirs(cfg.Env.output_dir, exist_ok=True)

    # Initialize the model
    model = STDiffDiffusers(cfg.STDiff.Diffusion.unet_config, cfg.STDiff.DiffNet)
    num_p_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num params of stdiff: {num_p_model/1e6} M')

    if cfg.Env.stdiff_init_ckpt is not None:
        model = STDiffDiffusers.from_pretrained(cfg.Env.stdiff_init_ckpt, subfolder='unet')
        print('Init from a checkpoint')
    
    # Enable gradient checkpointing if configured
    if cfg.Training.gradient_checkpointing:
        # Enable gradient checkpointing on the UNet blocks
        def enable_gc_recursive(module):
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
            for child in module.children():
                enable_gc_recursive(child)
        
        enable_gc_recursive(model.diffusion_unet)
        logger.info("Gradient checkpointing enabled")
    
    # Enable xformers memory efficient attention if configured
    if cfg.Training.use_xformers:
        try:
            model.enable_xformers_memory_efficient_attention()
            logger.info("XFormers memory efficient attention enabled")
        except Exception as e:
            logger.warning(f"Failed to enable xformers: {e}. Continuing without xformers.")

    # Create EMA for the model.
    if cfg.Training.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=cfg.Training.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=cfg.Training.ema_inv_gamma,
            power=cfg.Training.ema_power,
            model_cls=STDiffDiffusers,
            model_config=model.config,
        )

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.STDiff.Diffusion.ddpm_num_steps,
            beta_schedule=cfg.STDiff.Diffusion.ddpm_beta_schedule,
            prediction_type=cfg.STDiff.Diffusion.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=cfg.STDiff.Diffusion.ddpm_num_steps, beta_schedule=cfg.STDiff.Diffusion.ddpm_beta_schedule)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.Training.learning_rate,
        betas=cfg.Training.adam_betas,
        weight_decay=cfg.Training.adam_weight_decay,
        eps=cfg.Training.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # Preprocessing the datasets and DataLoaders creation.
    train_dataloader, val_dataloader, test_dataloader, _ = get_lightning_module_dataloader(cfg, stage="fit")

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        cfg.Training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.Training.lr_warmup_steps * cfg.Training.gradient_accumulation_steps,
        num_training_steps=len(train_dataloader) * cfg.Training.epochs,
        num_cycles=cfg.Training.num_cycles,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if cfg.Training.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = cfg.Dataset.batch_size * accelerator.num_processes * cfg.Training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.Training.gradient_accumulation_steps)
    max_train_steps = cfg.Training.epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {cfg.Training.epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.Dataset.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.Training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.Env.resume_ckpt is None:
        accelerator.print(
            f"Starting a new training run."
        )
        cfg.Env.resume_ckpt = None
    else:
        accelerator.print(f"Resuming from checkpoint {cfg.Env.resume_ckpt}")
        accelerator.load_state(os.path.join(cfg.Env.output_dir, cfg.Env.resume_ckpt))
        global_step = int(cfg.Env.resume_ckpt.split("-")[1])

        resume_global_step = global_step * cfg.Training.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * cfg.Training.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, cfg.Training.epochs):
        model.train()
        # Configure tqdm for immediate output flushing
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process, 
                           file=sys.stdout, mininterval=0.1)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Handle both regular datasets and range image datasets (with masks)
                if len(batch) == 8:  # Range images with masks
                    Vo, Vp, Vo_last_frame, idx_o, idx_p, Vo_mask, Vp_mask, Vo_last_mask = batch
                    has_masks = True
                else:  # Regular datasets without masks
                    Vo, Vp, Vo_last_frame, idx_o, idx_p = batch
                    has_masks = False
                    Vo_mask = None
                    Vp_mask = None
            
                clean_images = Vp.flatten(0, 1)
                # Flatten masks if they exist
                if has_masks:
                    valid_mask = Vp_mask.flatten(0, 1)  # (N*Tp, H, W)
                    # Keep binary masks as [0, 1]: valid=1, invalid=0
                    valid_mask_norm = valid_mask.float()  # (N*Tp, H, W)
                else:
                    valid_mask = None
                    valid_mask_norm = None

                # Check if mask prediction is enabled (from config)
                predict_mask = cfg.Training.get('predict_mask', False) and has_masks

                # Skip steps until we reach the resumed step
                if cfg.Env.resume_ckpt and epoch == first_epoch and step < resume_step:
                    if step % cfg.Training.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                # Sample noise that we'll add to the images
                # Base noise + small spatially constant (per-channel) noise: (..., 1, 1) broadcasts over H, W
                if not cfg.STDiff.DiffNet.autoregressive:
                    N, Tp, C, H, W = Vp.shape
                    noise = (
                        torch.randn(N, C, H, W).unsqueeze(1).repeat(1, Tp, 1, 1, 1).flatten(0, 1).to(clean_images.device)
                        + 0.1 * torch.randn(N * Tp, C, 1, 1, device=clean_images.device)
                    )
                    Vo_last_frame = None

                else:
                    bsz, C, H, W = clean_images.shape
                    noise = (
                        torch.randn(clean_images.shape).to(clean_images.device)
                        + 0.1 * torch.randn(bsz, C, 1, 1, device=clean_images.device)
                    )
                bsz = clean_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                # Add noise to masks if predict_mask is enabled
                noisy_mask = None
                mask_noise = None
                if predict_mask:
                    # Sample noise for masks
                    mask_noise = torch.randn(valid_mask_norm.shape).to(valid_mask_norm.device)
                    # Add noise to normalized masks (in [-1, 1] range)
                    noisy_mask = noise_scheduler.add_noise(valid_mask_norm, mask_noise, timesteps)
                    # Add channel dimension: (N*Tp, H, W) -> (N*Tp, 1, H, W)
                    noisy_mask = noisy_mask.unsqueeze(1)
                
                # For non-autoregressive mode, we need to concatenate Vo frames with noisy_images
                # This matches what the pipeline does during inference
                if not cfg.STDiff.DiffNet.autoregressive:
                    # Reshape Vo (all observed frames) to concatenate along channel dimension
                    # Vo: (N, To, C, H, W) -> Vo_reshaped: (N, To*C, H, W)
                    Vo_reshaped = Vo.view(Vo.shape[0], Vo.shape[1] * Vo.shape[2], Vo.shape[3], Vo.shape[4])  # (N, To*C, H, W)
                    # Repeat Vo_reshaped for all Tp frames to match batch dimension
                    # Vo_reshaped: (N, To*C, H, W) -> (N*Tp, To*C, H, W)
                    Vo_expanded = Vo_reshaped.unsqueeze(1).repeat(1, Vp.shape[1], 1, 1, 1).flatten(0, 1)
                    # Concatenate noisy_images with Vo_expanded
                    # If predict_mask, we need to handle mask separately or concatenate it too
                    if predict_mask and noisy_mask is not None:
                        # Concatenate image and mask first, then add Vo frames
                        # noisy_images: (N*Tp, C, H, W), noisy_mask: (N*Tp, 1, H, W)
                        noisy_images_with_mask = torch.cat([noisy_images, noisy_mask], dim=1)  # (N*Tp, C+1, H, W) = (N*Tp, 2, H, W)
                        noisy_images = torch.cat([noisy_images_with_mask, Vo_expanded.clamp(-1, 1)], dim=1)  # (N*Tp, 2+To*C, H, W) = (N*Tp, 5, H, W)
                        # For non-autoregressive, we don't pass noisy_mask separately since it's already in noisy_images
                        noisy_mask = None
                    else:
                        noisy_images = torch.cat([noisy_images, Vo_expanded.clamp(-1, 1)], dim=1)  # (N*Tp, C+To*C, H, W) = (N*Tp, 4, H, W)
                
                # Predict the noise residual
                model_output = model(Vo, idx_o, idx_p, noisy_images, timesteps, Vp, Vo_last_frame, 
                                     noisy_mask=noisy_mask, clean_mask=valid_mask_norm.unsqueeze(1) if predict_mask else None,
                                     predict_mask=predict_mask)

                # Use image channel only for image loss (channel 0 when predict_mask, else full sample)
                image_output_for_loss = model_output.sample[:, 0:1, ...] if predict_mask else model_output.sample
                if cfg.STDiff.Diffusion.prediction_type == "epsilon":
                    loss_per_pixel = F.l1_loss(image_output_for_loss, noise, reduction="none")  # (N*Tp, C, H, W)
                elif cfg.STDiff.Diffusion.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss_per_pixel = snr_weights * F.l1_loss(
                        image_output_for_loss, clean_images, reduction="none"
                    )  # use SNR weighting from distillation paper
                else:
                    raise ValueError(f"Unsupported prediction type: {cfg.STDiff.Diffusion.prediction_type}")
                
                # Compute image loss
                image_loss = loss_per_pixel.mean()
                
                # Compute mask loss if predict_mask is enabled (model keeps full 2-channel sample)
                mask_loss = None
                if predict_mask:
                    mask_output = model_output.sample[:, 1:2, ...]  # (N*Tp, 1, H, W) - mask logits
                    mask_target = valid_mask_norm.unsqueeze(1)  # (N*Tp, 1, H, W) - binary [0, 1]
                    mask_loss = F.binary_cross_entropy_with_logits(
                        mask_output, mask_target.float(), reduction="mean"
                    )
                
                # Combine losses
                mask_weight = cfg.Training.get('mask_loss_weight', 1.0)
                if mask_loss is not None:
                    loss = image_loss + mask_weight * mask_loss
                else:
                    loss = image_loss
                
                # Check for NaN or Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Step {global_step}: NaN/Inf loss detected (loss={loss.item()}), skipping backward pass")
                    optimizer.zero_grad()
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Check for NaN gradients before clipping
                    has_nan_grad = False
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan_grad = True
                                logger.warning(f"Step {global_step}: NaN/Inf gradients detected, skipping optimizer step")
                                break
                    
                    if not has_nan_grad:
                        max_grad_norm = cfg.Training.get('max_grad_norm', 1.0)
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                    else:
                        # Skip optimizer step but still zero gradients
                        pass
                    
                    optimizer.zero_grad()
                
                # Garbage collect and clear CUDA cache periodically to reduce fragmentation
                if step % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if cfg.Training.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % cfg.Training.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(cfg.Env.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
                        # Automatically clean up old checkpoints
                        max_checkpoints = cfg.Training.get('max_checkpoints_to_keep', 3)
                        if max_checkpoints > 0:
                            # Find all checkpoint directories
                            checkpoint_pattern = os.path.join(cfg.Env.output_dir, "checkpoint-*")
                            checkpoints = sorted(glob.glob(checkpoint_pattern), key=os.path.getmtime)
                            
                            # Keep only the latest N checkpoints
                            if len(checkpoints) > max_checkpoints:
                                checkpoints_to_delete = checkpoints[:-max_checkpoints]
                                for old_checkpoint in checkpoints_to_delete:
                                    try:
                                        shutil.rmtree(old_checkpoint)
                                        logger.info(f"Deleted old checkpoint: {os.path.basename(old_checkpoint)}")
                                    except Exception as e:
                                        logger.warning(f"Failed to delete checkpoint {old_checkpoint}: {e}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if cfg.Training.get('predict_mask', False):
                if mask_loss is not None:
                    logs["mask_loss"] = round(mask_loss.detach().item(), 4)
                    logs["image_loss"] = round(image_loss.detach().item(), 4)
                else:
                    logs["mask_loss"] = 0.0
            if cfg.Training.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % cfg.Training.save_images_epochs == 0 or epoch == cfg.Training.epochs - 1:
                unet = accelerator.unwrap_model(model)

                if cfg.Training.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = STDiffPipeline(
                    stdiff=unet,
                    scheduler=noise_scheduler,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                batch = next(iter(train_dataloader))
                # Handle both regular datasets (5 values) and range images with masks (8 values)
                if len(batch) == 8:
                    Vo, _, Vo_last_frame, idx_o, idx_p, _, Vp_mask, _ = batch
                    has_masks = True
                else:
                    Vo, _, Vo_last_frame, idx_o, idx_p = batch
                    has_masks = False
                    Vp_mask = None
                
                # Check if mask prediction is enabled
                predict_mask = cfg.Training.get('predict_mask', False) and has_masks
                
                if predict_mask:
                    images_output, masks = pipeline(
                        Vo,
                        Vo_last_frame,
                        idx_o,
                        idx_p,
                        generator=generator,
                        num_inference_steps=cfg.STDiff.Diffusion.ddpm_num_inference_steps,
                        output_type="numpy",
                        predict_mask=True,
                        Vp_mask=Vp_mask
                    )
                    # Extract images from ImagePipelineOutput
                    images = images_output.images
                    # masks are raw values, convert to binary for visualization if needed
                    # For now, just use images
                else:
                    images = pipeline(
                        Vo,
                        Vo_last_frame,
                        idx_o,
                        idx_p,
                        generator=generator,
                        num_inference_steps=cfg.STDiff.Diffusion.ddpm_num_inference_steps,
                        output_type="numpy"
                    ).images

                if cfg.Training.use_ema:
                    ema_model.restore(unet.parameters())

                # denormalize the images and save to tensorboard
                images_processed = (images * 255).round().astype("uint8")

                if cfg.Env.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif cfg.Env.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )

            if epoch % cfg.Training.save_model_epochs == 0 or epoch == cfg.Training.epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if cfg.Training.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = STDiffPipeline(
                    stdiff=unet,
                    scheduler=noise_scheduler
                )

                pipeline.save_pretrained(cfg.Env.output_dir)

                if cfg.Training.use_ema:
                    ema_model.restore(unet.parameters())

    accelerator.end_training()

if __name__ == '__main__':
    config_path = Path(parse_args())
    initialize_config_dir(version_base=None, config_dir=str('/home/anirudh/STDiffProject/stdiff/configs'))
    cfg = compose(config_name=str(config_path.name))

    main(cfg)