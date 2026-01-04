# UNet2DMotionCond Model Documentation

## Overview

`UNet2DMotionCond` is a 2D UNet model that extends the standard UNet architecture with **motion conditioning** capabilities. This model takes in a noisy sample, a timestep, and an optional motion feature tensor (predicted by ODE/SDE solvers) to generate denoised outputs.

### Key Features

- **Motion Conditioning**: Incorporates motion features (`m_feat`) throughout the network via SPADE-style normalization layers
- **Time Embedding**: Supports both positional and Fourier time embeddings
- **Class Conditioning**: Optional class label conditioning support
- **ConfigMixin Integration**: Inherits from `ConfigMixin` for easy configuration management

## Architecture

The model follows a standard U-Net architecture with:
- **Encoder (Down Blocks)**: Progressive downsampling with optional attention layers
- **Bottleneck (Mid Block)**: Central processing block
- **Decoder (Up Blocks)**: Progressive upsampling with skip connections
- **Motion Conditioning**: Applied at each ResNet block via `MotionCondModule`

## Configuration Parameters

The model uses `@register_to_config` decorator to automatically save/load configurations. All parameters are accessible via `self.config` after initialization.

### Input/Output Parameters

| Parameter | Type | Default | Description | Tensor Shape (if applicable) |
|-----------|------|---------|-------------|------------------------------|
| `sample_size` | `Optional[Union[int, Tuple[int, int]]]` | `None` | Height and width of input/output sample | - |
| `in_channels` | `int` | `3` | Number of channels in input image | Input: `(B, in_channels, H, W)` |
| `out_channels` | `int` | `3` | Number of channels in output | Output: `(B, out_channels, H, W)` |
| `center_input_sample` | `bool` | `False` | Whether to center input sample (scale to [-1, 1]) | - |

### Time Embedding Parameters

| Parameter | Type | Default | Description | Tensor Shape |
|-----------|------|---------|-------------|--------------|
| `time_embedding_type` | `str` | `"positional"` | Type: `"positional"` or `"fourier"` | - |
| `freq_shift` | `int` | `0` | Frequency shift for Fourier embedding | - |
| `flip_sin_to_cos` | `bool` | `True` | Flip sin to cos for Fourier embedding | - |

**Time Embedding Dimensions:**
- `time_embed_dim = block_out_channels[0] * 4` (e.g., 224 * 4 = 896)
- Positional: `timestep_input_dim = block_out_channels[0]` → `time_embed_dim`
- Fourier: `timestep_input_dim = 2 * block_out_channels[0]` → `time_embed_dim`

### Motion Conditioning Parameters

| Parameter | Type | Default | Description | Tensor Shape |
|-----------|------|---------|-------------|--------------|
| `m_channels` | `int` | `0` | Number of channels in motion feature tensor | `m_feat`: `(B, m_channels, H_m, W_m)` |

**Note**: When `m_channels=0`, motion conditioning is disabled. The motion features are interpolated to match spatial dimensions at each ResNet block.

### Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `down_block_types` | `Tuple[str]` | `("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")` | Types of downsampling blocks |
| `up_block_types` | `Tuple[str]` | `("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")` | Types of upsampling blocks |
| `block_out_channels` | `Tuple[int]` | `(224, 448, 672, 896)` | Output channels for each block level |
| `layers_per_block` | `int` | `2` | Number of ResNet layers per block |
| `mid_block_scale_factor` | `float` | `1` | Scale factor for mid block output |
| `downsample_padding` | `int` | `1` | Padding for downsample convolution |

### Attention Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attention_head_dim` | `Union[int, Tuple[int]]` | `8` | Attention head dimension (can be per-block) |
| `add_attention` | `bool` | `True` | Whether to add attention in mid block |

### Normalization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `norm_num_groups` | `int` | `32` | Number of groups for GroupNorm |
| `norm_eps` | `float` | `1e-5` | Epsilon for normalization layers |
| `resnet_time_scale_shift` | `str` | `"default"` | Time scale shift: `"default"` or `"scale_shift"` |

### Activation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `act_fn` | `str` | `"silu"` | Activation function (typically `"silu"` or `"swish"`) |

### Class Conditioning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_embed_type` | `Optional[str]` | `None` | Type: `None`, `"timestep"`, or `"identity"` |
| `num_class_embeds` | `Optional[int]` | `None` | Number of class embeddings (if `class_embed_type=None`) |

## Forward Pass: Tensor Flow

### Input Tensors

```python
sample: torch.FloatTensor  # Shape: (batch_size, in_channels, height, width)
timestep: Union[torch.Tensor, float, int]  # Shape: (batch_size,) after processing
m_feat: Optional[torch.FloatTensor]  # Shape: (batch_size, m_channels, H_m, W_m)
class_labels: Optional[torch.Tensor]  # Shape: (batch_size,)
```

### Step-by-Step Tensor Transformations

#### 1. Input Preprocessing
```python
# If center_input_sample=True:
sample = 2 * sample - 1.0  # Scale from [0, 1] to [-1, 1]
# Shape: (B, in_channels, H, W)
```

#### 2. Time Embedding Processing
```python
# Timestep processing
timesteps = timestep * torch.ones(sample.shape[0], ...)  # Shape: (B,)
t_emb = self.time_proj(timesteps)  # Shape: (B, block_out_channels[0]) or (B, 2*block_out_channels[0])
emb = self.time_embedding(t_emb)  # Shape: (B, time_embed_dim) = (B, block_out_channels[0]*4)
```

#### 3. Class Embedding (if enabled)
```python
if class_embedding is not None:
    class_emb = self.class_embedding(class_labels)  # Shape: (B, time_embed_dim)
    emb = emb + class_emb  # Shape: (B, time_embed_dim)
```

#### 4. Input Convolution
```python
skip_sample = sample  # Save for skip connection: (B, in_channels, H, W)
sample = self.conv_in(sample)  # Shape: (B, block_out_channels[0], H, W)
```

#### 5. Downsampling Blocks
```python
down_block_res_samples = (sample,)  # Initialize with input: (B, block_out_channels[0], H, W)

for downsample_block in self.down_blocks:
    # Each block processes and downsamples
    # Input: sample shape depends on previous block
    # Output: sample with reduced spatial size, increased channels
    # Example progression:
    # Block 0: (B, 224, H, W) → (B, 224, H/2, W/2)  [if not final]
    # Block 1: (B, 224, H/2, W/2) → (B, 448, H/4, W/4)
    # Block 2: (B, 448, H/4, W/4) → (B, 672, H/8, W/8)
    # Block 3: (B, 672, H/8, W/8) → (B, 896, H/8, W/8)  [final, no downsample]
    
    sample, res_samples = downsample_block(
        hidden_states=sample, 
        temb=emb,  # (B, time_embed_dim)
        m_feat=m_feat  # (B, m_channels, H_m, W_m) - interpolated inside block
    )
    # res_samples: tuple of intermediate activations for skip connections
    down_block_res_samples += res_samples
```

**Motion Conditioning in Down Blocks:**
- `m_feat` is passed to each ResNet block
- Inside `ResnetBlock2D.forward()`, if `m_channels != 0`:
  - `m_feat` is interpolated to match current spatial dimensions
  - Applied via `MotionCondModule` after first convolution
  - Shape: `m_feat` interpolated to match `hidden_states` spatial size

#### 6. Mid Block
```python
# Input: (B, block_out_channels[-1], H_min, W_min)
# For default config: (B, 896, H/8, W/8)
sample = self.mid_block(sample, emb, m_feat=m_feat)
# Output: (B, block_out_channels[-1], H_min, W_min)
```

#### 7. Upsampling Blocks
```python
# Process in reverse order of down blocks
# Example progression (reversed channels):
# Block 0: (B, 896, H/8, W/8) + skip → (B, 672, H/4, W/4)
# Block 1: (B, 672, H/4, W/4) + skip → (B, 448, H/2, W/2)
# Block 2: (B, 448, H/2, W/2) + skip → (B, 224, H, W)
# Block 3: (B, 224, H, W) + skip → (B, 224, H, W)  [final, no upsample]

for upsample_block in self.up_blocks:
    res_samples = down_block_res_samples[-len(upsample_block.resnets):]
    down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
    
    sample = upsample_block(
        sample,  # Current hidden states
        res_samples,  # Skip connections from encoder
        emb,  # Time embedding: (B, time_embed_dim)
        m_feat=m_feat  # Motion features: (B, m_channels, H_m, W_m)
    )
```

#### 8. Output Processing
```python
# Normalization and activation
sample = self.conv_norm_out(sample)  # GroupNorm: (B, block_out_channels[0], H, W)
sample = self.conv_act(sample)  # SiLU activation
sample = self.conv_out(sample)  # Final conv: (B, out_channels, H, W)

# Optional skip connection addition (if skip_sample exists)
if skip_sample is not None:
    sample += skip_sample  # Element-wise addition

# Fourier embedding normalization (if applicable)
if time_embedding_type == "fourier":
    timesteps = timesteps.reshape((B, 1, 1, 1))
    sample = sample / timesteps  # Normalize by timestep
```

### Final Output

```python
# Return UNet2DOutput
return UNet2DOutput(sample=sample)  # sample: (B, out_channels, H, W)
```

## Motion Conditioning Mechanism

### MotionCondModule Architecture

The motion conditioning is implemented via `MotionCondModule` (in `resnet.py`), inspired by SPADE normalization:

```python
class MotionCondModule(nn.Module):
    def __init__(self, motion_embed_dim, out_dim, ...):
        # Intermediate dimension
        nhidden = (motion_embed_dim + out_dim) // 2
        
        # Shared MLP
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(motion_embed_dim, nhidden, 3, 1, 1),
            nn.ReLU()
        )
        
        # Gamma and beta generators
        self.mlp_gamma = nn.Conv2d(nhidden, out_dim, 3, 1, 1)
        self.mlp_beta = nn.Conv2d(nhidden, out_dim, 3, 1, 1)
```

### Forward Process

1. **Normalize**: Apply parameter-free normalization (InstanceNorm2d)
2. **Interpolate**: Resize `m_feat` to match current feature map spatial dimensions
3. **Generate Parameters**: 
   - `gamma = mlp_gamma(mlp_shared(m_feat))`  # Shape: (B, out_dim, H, W)
   - `beta = mlp_beta(mlp_shared(m_feat))`   # Shape: (B, out_dim, H, W)
4. **Apply**: `output = normalized * (1 + gamma) + beta`

This allows motion features to modulate the feature maps at each ResNet block, providing fine-grained control over the denoising process.

## Configuration Files

The model inherits from `ConfigMixin`, which provides automatic configuration management. Configuration files are typically stored as JSON files.

### Configuration File Location

Configuration files follow the pattern:
- `config.<model_name>.json` (e.g., `config.unet_2d_motion_cond.json`)

### Saving Configuration

```python
from diffusers import UNet2DMotionCond

model = UNet2DMotionCond(
    in_channels=3,
    out_channels=3,
    m_channels=64,  # Enable motion conditioning
    block_out_channels=(224, 448, 672, 896),
    # ... other parameters
)

# Save configuration
model.save_config("path/to/config")
# Creates: path/to/config/config.unet_2d_motion_cond.json
```

### Loading from Configuration

```python
from diffusers import UNet2DMotionCond

# Load from config file
model = UNet2DMotionCond.from_config("path/to/config/config.unet_2d_motion_cond.json")

# Or from a dictionary
config_dict = {...}  # Configuration dictionary
model = UNet2DMotionCond.from_config(config_dict)
```

### Configuration Structure

A typical configuration JSON file would look like:

```json
{
  "sample_size": null,
  "in_channels": 3,
  "out_channels": 3,
  "center_input_sample": false,
  "time_embedding_type": "positional",
  "m_channels": 64,
  "freq_shift": 0,
  "flip_sin_to_cos": true,
  "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
  "up_block_types": ["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
  "block_out_channels": [224, 448, 672, 896],
  "layers_per_block": 2,
  "mid_block_scale_factor": 1,
  "downsample_padding": 1,
  "act_fn": "silu",
  "attention_head_dim": 8,
  "norm_num_groups": 32,
  "norm_eps": 1e-5,
  "resnet_time_scale_shift": "default",
  "add_attention": true,
  "class_embed_type": null,
  "num_class_embeds": null
}
```

## Related Files

### Core Implementation Files

- **`unet_2d_motion_cond.py`**: Main model definition
- **`unet_2d_blocks.py`**: Block definitions (down, up, mid blocks)
- **`resnet.py`**: ResNet blocks with `MotionCondModule` implementation
- **`embeddings.py`**: Time embedding implementations (`TimestepEmbedding`, `Timesteps`, `GaussianFourierProjection`)

### Configuration and Utilities

- **`configuration_utils.py`**: `ConfigMixin` base class for configuration management
- **`modeling_utils.py`**: `ModelMixin` base class for model utilities

### Usage Example

```python
import torch
from diffusers import UNet2DMotionCond

# Initialize model
model = UNet2DMotionCond(
    in_channels=3,
    out_channels=3,
    m_channels=64,  # Motion feature channels
    block_out_channels=(224, 448, 672, 896),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
)

# Forward pass
batch_size = 4
height, width = 64, 64
sample = torch.randn(batch_size, 3, height, width)
timestep = torch.randint(0, 1000, (batch_size,))
m_feat = torch.randn(batch_size, 64, height, width)  # Motion features

with torch.no_grad():
    output = model(sample, timestep, m_feat=m_feat)
    print(f"Output shape: {output.sample.shape}")  # (4, 3, 64, 64)
```

## Key Differences from Standard UNet2D

1. **Motion Conditioning**: Additional `m_feat` parameter throughout the network
2. **MotionCondModule**: SPADE-style normalization layers in ResNet blocks
3. **`m_channels` Parameter**: Controls whether motion conditioning is enabled
4. **Forward Signature**: `forward(sample, timestep, m_feat=None, ...)` vs standard `forward(sample, timestep, ...)`

## Notes

- When `m_channels=0`, motion conditioning is disabled and the model behaves like a standard UNet (with some overhead from conditional checks)
- Motion features are automatically interpolated to match spatial dimensions at each block
- The model supports both training and inference modes
- All tensor operations are compatible with mixed precision training (FP16/BF16)

