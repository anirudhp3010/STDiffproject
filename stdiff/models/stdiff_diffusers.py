import torch
import torchvision.transforms as transforms
from types import SimpleNamespace

from diffusers import ConfigMixin, ModelMixin, register_to_config, UNet2DMotionCond
from .diff_unet import DiffModel
from omegaconf import OmegaConf
from typing import Optional, Tuple


def _unet_forward_with_bottleneck(unet, sample: torch.Tensor, timestep, m_feat=None, class_labels=None):
    """Run UNet forward step-by-step and return (output, bottleneck). Works with any UNet (no return_bottleneck API)."""
    # Use inner module if wrapped (e.g. by DDP/Accelerate)
    unet = getattr(unet, "module", unet)
    # 0. center input if necessary
    if getattr(unet.config, "center_input_sample", False):
        sample = 2 * sample - 1.0

    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
    elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

    t_emb = unet.time_proj(timesteps)
    t_emb = t_emb.to(dtype=unet.dtype)
    emb = unet.time_embedding(t_emb)

    if getattr(unet, "class_embedding", None) is not None and class_labels is not None:
        if getattr(unet.config, "class_embed_type", None) == "timestep":
            class_labels = unet.time_proj(class_labels)
        class_emb = unet.class_embedding(class_labels).to(dtype=unet.dtype)
        emb = emb + class_emb

    # 2. pre-process
    skip_sample = sample
    sample = unet.conv_in(sample)

    # 3. down
    down_block_res_samples = (sample,)
    for downsample_block in unet.down_blocks:
        if hasattr(downsample_block, "skip_conv"):
            sample, res_samples, skip_sample = downsample_block(
                hidden_states=sample, temb=emb, m_feat=m_feat, skip_sample=skip_sample
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb, m_feat=m_feat)
        down_block_res_samples += res_samples

    # 4. mid -> capture bottleneck
    sample = unet.mid_block(sample, emb, m_feat=m_feat)
    bottleneck = sample

    # 5. up
    skip_sample = None
    for upsample_block in unet.up_blocks:
        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
        if hasattr(upsample_block, "skip_conv"):
            sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample, m_feat=m_feat)
        else:
            sample = upsample_block(sample, res_samples, emb, m_feat=m_feat)

    # 6. post-process
    sample = unet.conv_norm_out(sample)
    sample = unet.conv_act(sample)
    sample = unet.conv_out(sample)
    if skip_sample is not None:
        sample += skip_sample
    if getattr(unet.config, "time_embedding_type", None) == "fourier":
        timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
        sample = sample / timesteps

    out = SimpleNamespace(sample=sample)
    return out, bottleneck

class STDiffDiffusers(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, unet_cfg, tde_cfg, repa_config: Optional[dict] = None):
        super().__init__()
        try:
            self.autoreg = tde_cfg.autoregressive
            self.super_res_training = tde_cfg.super_res_training
            self.tde_model = DiffModel(tde_cfg.Int, tde_cfg.MotionEncoder, tde_cfg.DiffUnet)
        except AttributeError:
            tde_cfg = OmegaConf.structured(tde_cfg)
            self.autoreg = tde_cfg.autoregressive
            self.super_res_training =  tde_cfg.super_res_training
            self.tde_model = DiffModel(tde_cfg.Int, tde_cfg.MotionEncoder, tde_cfg.DiffUnet)
        self.diffusion_unet = UNet2DMotionCond(**unet_cfg)

        # Optional REPA projector (modular). Saved/loaded with the model via state_dict (save_pretrained includes it).
        self.repa_projector = None
        if repa_config is not None and repa_config.get("enabled", False):
            from .projector import REPAProjector
            self.repa_projector = REPAProjector(
                hidden_channels=unet_cfg["block_out_channels"][-1],
                z_dim=repa_config.get("z_dim", 768),
                projector_dim=repa_config.get("projector_dim", 2048),
                target_grid=tuple(repa_config["target_grid"]) if repa_config.get("target_grid") else None,
                scale_mode=repa_config.get("scale_mode", "bicubic"),
            )

    def forward(self, Vo, idx_o, idx_p, noisy_Vp, timestep, clean_Vp = None, Vo_last_frame=None, 
                noisy_mask=None, clean_mask=None, predict_mask=False, return_projection=False):
        #vo: (N, To, C, Ho, Wo), idx_o: (To, ), idx_p: (Tp, ), noisy_Vp: (N*Tp, C, Hp, Wp)
        m_context = self.tde_model.context_encode(Vo, idx_o) #(N, C, H, W)
            
        #use ode/sde to predict the future motion features
        m_future = self.tde_model.future_predict(m_context, torch.cat([idx_o[-1:], idx_p])) #(Tp, N, C, H, W)

        if self.autoreg:
            assert clean_Vp is not None and Vo_last_frame is not None, "input clean Vp and last frame of observation for autoregressive prediction."
            #for the superresolution model, prev_frames have a lower resolution (Ho, Wo)

            N, To, C, Ho, Wo = Vo.shape
            N, Tp, C, Hp, Wp = clean_Vp.shape
            if self.super_res_training:
                if Ho < Hp or Wo < Wp:
                    down_sample= transforms.Resize((Ho, Wo), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
                    up_sample = transforms.Resize((Hp, Wp), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

                    clean_Vp = up_sample(down_sample(clean_Vp.flatten(0, 1)))
                    clean_Vp = clean_Vp.reshape(N, Tp, C, Hp, Wp)
                    Vo_last_frame = up_sample(Vo[:, -1, ...]).reshape(N, 1, C, Hp, Wp)
            prev_frames = torch.cat([Vo_last_frame, clean_Vp[:, 0:-1, ...]], dim = 1)
            noisy_Vp = torch.cat([noisy_Vp, prev_frames.flatten(0, 1)], dim = 1)

        # Handle mask prediction: concatenate mask with image if predict_mask is True
        # Note: In non-autoregressive mode, the mask may already be concatenated in noisy_Vp
        # (done in training code). Check if mask needs to be added by checking channel count.
        if predict_mask and noisy_mask is not None:
            # For autoregressive mode, we need to add the mask
            # For non-autoregressive mode, check if mask is already in noisy_Vp
            if self.autoreg:
                # Autoregressive: concatenate mask
                noisy_Vp = torch.cat([noisy_Vp, noisy_mask], dim=1)
            else:
                # Non-autoregressive: mask should already be in noisy_Vp from training code
                # But if it's not (e.g., during inference), add it
                # Check expected channels: out_channels (2) + To*C
                # If noisy_Vp has fewer channels than expected, add mask
                expected_channels_with_mask = 2 + Vo.shape[1] * Vo.shape[2]  # out_channels + To*C
                if noisy_Vp.shape[1] < expected_channels_with_mask:
                    noisy_Vp = torch.cat([noisy_Vp, noisy_mask], dim=1)

        # Run UNet; when REPA is enabled, capture bottleneck (works with or without CustomDiffusers)
        m_feat = m_future.permute(1, 0, 2, 3, 4).flatten(0, 1)
        use_bottleneck = return_projection and self.repa_projector is not None
        if use_bottleneck:
            try:
                result = self.diffusion_unet(noisy_Vp, timestep, m_feat=m_feat, return_bottleneck=True)
                out, bottleneck = result
            except TypeError:
                # UNet has no return_bottleneck; run step-by-step to capture mid_block output
                out, bottleneck = _unet_forward_with_bottleneck(
                    self.diffusion_unet, noisy_Vp, timestep, m_feat=m_feat
                )
            zs_tilde = [self.repa_projector(bottleneck)]
            # Return a plain object so projection_output is always visible (BaseOutput/dict can hide custom attrs after DDP/wrapping)
            out = SimpleNamespace(sample=out.sample, projection_output=zs_tilde)
        else:
            out = self.diffusion_unet(noisy_Vp, timestep, m_feat=m_feat)

        # One-time debug: confirm projection_output is set when REPA was requested
        if return_projection and not getattr(self, "_logged_repa_forward", False):
            self._logged_repa_forward = True
            import logging
            logging.getLogger(__name__).info(
                "REPA forward: use_bottleneck=%s repa_projector=%s out_has_proj=%s",
                use_bottleneck, self.repa_projector is not None,
                getattr(out, "projection_output", None) is not None,
            )

        if return_projection and self.repa_projector is None:
            if not getattr(self, "_logged_no_projector", False):
                self._logged_no_projector = True
                import logging
                logging.getLogger(__name__).warning("REPA: return_projection=True but repa_projector is None (repa_config not enabled or model loaded without it?).")

        # Split output if predict_mask: out_channels should be 2 (1 image + 1 mask)
        # Keep full out.sample (2 channels) so training can extract both; set image_sample/mask_sample for pipeline
        if predict_mask:
            image_output = out.sample[:, 0:1, ...]  # (N*Tp, 1, H, W) for grayscale
            mask_output = out.sample[:, 1:2, ...]   # (N*Tp, 1, H, W)
            out.image_sample = image_output
            out.mask_sample = mask_output
            # Do NOT overwrite out.sample - keep full 2-channel for training (loss uses both channels)
        
        return out