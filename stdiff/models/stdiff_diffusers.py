import torch
import torchvision.transforms as transforms

from diffusers import ConfigMixin, ModelMixin, register_to_config, UNet2DMotionCond
from .diff_unet import DiffModel
from omegaconf import OmegaConf

class STDiffDiffusers(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, unet_cfg, tde_cfg):
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

    def forward(self, Vo, idx_o, idx_p, noisy_Vp, timestep, clean_Vp = None, Vo_last_frame=None, 
                noisy_mask=None, clean_mask=None, predict_mask=False):
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

        out = self.diffusion_unet(noisy_Vp, timestep, m_feat = m_future.permute(1, 0, 2, 3, 4).flatten(0, 1))
        
        # Split output if predict_mask: out_channels should be 2 (1 image + 1 mask)
        if predict_mask:
            # Split output: channels 0 for image, channel 1 for mask
            image_output = out.sample[:, 0:1, ...]  # (N*Tp, 1, H, W) for grayscale
            mask_output = out.sample[:, 1:2, ...]   # (N*Tp, 1, H, W)
            # Store both in the output object for backward compatibility
            out.image_sample = image_output
            out.mask_sample = mask_output
            # Set sample to image for backward compatibility with existing code
            out.sample = image_output
        
        return out