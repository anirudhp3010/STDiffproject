# Detailed Explanation of `diff_unet.py`

This document explains every component, parameter, and function in `diff_unet.py` step by step, with references to the configuration file.

---

## Overview

The `diff_unet.py` file implements the **Temporal Dynamics Encoder (TDE)** model, which is responsible for:
1. Encoding observed frames into motion features
2. Predicting future motion features using ODE/SDE integration

**Important:** This is NOT a diffusion model. It's a motion feature predictor that uses differential equation solvers.

---

## Configuration Reference

From `kitti_range_train_config.yaml`:

```yaml
STDiff:
    DiffNet:
        autoregressive: True
        super_res_training: False
        MotionEncoder:
            learn_diff_image: True
            image_size: [64, 512]  # Height x Width
            in_channels: 1          # Grayscale input
            model_channels: 64      # Base channel size
            n_downs: 2              # Number of downsampling layers
        DiffUnet:
            n_layers: 2             # Number of layers in ODE/SDE function network
            nonlinear: 'tanh'       # Nonlinear activation
        Int:
            sde: True               # Use SDE (True) or ODE (False)
            method: 'euler_heun'    # Integration method
            sde_options:
                noise_type: 'diagonal'
                sde_type: "stratonovich"
                dt: 0.1             # Time step size
                rtol: 1e-3           # Relative tolerance
                atol: 1e-3           # Absolute tolerance
                adaptive: False      # Adaptive step size
            ode_options:
                step_size: 0.1
                norm: null
```

---

## Class 1: `DiffModel` (Main Class)

### Initialization: `__init__(int_cfg, motion_encoder_cfg, diff_unet_cfg)`

**Purpose:** Creates the complete temporal dynamics encoder model.

**Parameters:**
- `int_cfg`: Integration configuration (ODE/SDE settings) → `cfg.STDiff.DiffNet.Int`
- `motion_encoder_cfg`: Motion encoder configuration → `cfg.STDiff.DiffNet.MotionEncoder`
- `diff_unet_cfg`: DiffUnet configuration → `cfg.STDiff.DiffNet.DiffUnet`

#### Step-by-Step Initialization:

**Step 1: Store Configurations (lines 19-22)**
```python
self.sde = int_cfg.sde  # True = use SDE, False = use ODE
self.int_cfg = int_cfg
self.motion_encoder_cfg = motion_encoder_cfg
self.diff_unet_cfg = diff_unet_cfg
```

**Step 2: Extract Motion Encoder Settings (lines 24-26)**
```python
self.learn_diff_image = motion_encoder_cfg.learn_diff_image  # True = learn from frame differences
n_downs = motion_encoder_cfg.n_downs  # 2 = number of downsampling layers
image_size = motion_encoder_cfg.image_size  # [64, 512] = [height, width]
```

**Step 3: Calculate Motion Feature Dimensions (lines 27-35)**
```python
# Handle non-square images (like 64x512 range images)
if isinstance(image_size, (list, tuple, ListConfig)):
    image_h, image_w = int(image_size[0]), int(image_size[1])  # 64, 512
    H = int(image_h/(2**n_downs))  # 64/(2^2) = 16 (height after downsampling)
    W = int(image_w/(2**n_downs))  # 512/(2^2) = 128 (width after downsampling)
else:
    H = W = int(image_size/(2**n_downs))  # For square images

# Calculate motion feature channels
# model_channels * (2^n_downs) = 64 * (2^2) = 64 * 4 = 256
motion_C = motion_encoder_cfg.model_channels * (2**n_downs)
self.motion_feature_size = (motion_C, H, W)  # (256, 16, 128)
```

**Result:** Motion features have shape `(256, 16, 128)` - 256 channels, 16x128 spatial size.

**Step 4: Create Components (lines 37-38)**
```python
self.motion_encoder = MotionEncoder(motion_encoder_cfg)  # Encodes frames → motion features
self.conv_gru_cell = ConvGRUCell(motion_C, motion_C, kernel_size=3, stride=1, padding=1)
# ConvGRU: 256 input channels, 256 hidden channels, 3x3 conv
```

**Step 5: Configure DiffUnet (lines 40-44)**
```python
with open_dict(self.diff_unet_cfg):
    self.diff_unet_cfg.image_size = H  # 16 (height of motion features)
    self.diff_unet_cfg.in_channels = motion_C  # 256
    self.diff_unet_cfg.hidden_channels = motion_C  # 256
```

**Step 6: Create DiffUnet (line 46)**
```python
self.diff_unet = DiffUnet(diff_unet_cfg, int_cfg, self.motion_feature_size)
# This is the ODE/SDE solver network
```

---

### Method 1: `context_encode(Vo, idx_o)`

**Purpose:** Encodes observed frames into a motion context feature.

**Inputs:**
- `Vo`: `(N, To, C, H, W)` - Observed frames
  - `N`: Batch size
  - `To`: Number of observed frames (3)
  - `C`: Channels (1 for grayscale)
  - `H, W`: Image dimensions (64, 512)
- `idx_o`: `(To,)` - Temporal indices of observed frames

**Output:** `(N, C, H, W)` - Motion context feature `(N, 256, 16, 128)`

#### Step-by-Step:

**Step 1: Compute Frame Differences or Encode Frames (lines 56-62)**
```python
if self.learn_diff_image:  # True in config
    assert Vo.shape[1] >= 2  # Need at least 2 frames
    # Compute frame-to-frame differences
    # Vo[:, 1:, ...] - Vo[:, 0:-1, ...] = (N, To-1, C, H, W)
    # Example: frames [0,1,2] → differences [(1-0), (2-1)] = [diff_01, diff_12]
    diff_images = Vo[:, 1:, ...] - Vo[:, 0:-1, ...]
    h = self.condition_enc(diff_images)  # Encode differences
else:
    h = self.condition_enc(Vo)  # Encode all frames directly
```

**Step 2: Initialize Motion Feature (lines 64-65)**
```python
m = torch.zeros(self.motion_feature_size, device=h.device)  # (256, 16, 128)
m = repeat(m, 'C H W -> N C H W', N=Vo.shape[0])  # (N, 256, 16, 128)
```

**Step 3: Process First Frame (line 67)**
```python
m = self.conv_gru_cell(h[:, 0, ...], m)
# Update motion feature with first frame's encoded features
```

**Step 4: Process Remaining Frames Recurrently (lines 70-74)**
```python
To = h.shape[1]  # Number of observed frames
for i in range(1, To):
    # Option 1 (commented): Use ODE integration
    # m = odeint(self.diff_unet, m, idx_o[i-1:i+1])[-1, ...]
    
    # Option 2 (used): Use ConvGRU
    m = self.conv_gru_cell(h[:, i, ...], m)
```

**Result:** `m` contains the motion context feature encoding all observed frames.

---

### Method 2: `future_predict(m_context, idx_p)`

**Purpose:** Predicts future motion features using ODE/SDE integration.

**Inputs:**
- `m_context`: `(N, C, H, W)` - Motion context from `context_encode()`
- `idx_p`: `(Tp+1,)` - Temporal indices for prediction
  - First element: Last observed frame index
  - Remaining: Future frame indices

**Output:** `(Tp, N, C, H, W)` - Future motion features for each predicted frame

#### Step-by-Step:

**Step 1: Choose Integration Method (lines 86-96)**
```python
if self.sde:  # True in config
    N, C, H, W = m_context.shape  # (N, 256, 16, 128)
    
    # Flatten spatial dimensions for SDE solver
    m_context_flat = m_context.flatten(1)  # (N, 256*16*128) = (N, 524288)
    
    # Solve SDE: dm/dt = f(m, t) + g(m, t) * dW
    m_future = sdeint(
        self.diff_unet,           # SDE function (has f() and g() methods)
        m_context_flat,           # Initial condition
        idx_p,                    # Time points to solve at
        method=self.int_cfg.method,  # 'euler_heun'
        dt=self.int_cfg.sde_options.dt,  # 0.1
        rtol=self.int_cfg.sde_options.rtol,  # 1e-3
        atol=self.int_cfg.sde_options.atol,  # 1e-3
        adaptive=self.int_cfg.sde_options.adaptive  # False
    )  # Output: (t, N, C*H*W) = (Tp+1, N, 524288)
    
    # Reshape back to spatial format
    m_future = rearrange(m_future, 't N (C H W) -> t N C H W', C=C, H=H, W=W)
    # Result: (Tp+1, N, 256, 16, 128)
else:
    # ODE version: dm/dt = f(m, t)
    m_future = odeint(
        self.diff_unet,           # ODE function (has forward() method)
        m_context,                # Initial condition
        idx_p,                    # Time points
        method=self.int_cfg.method,
        options=self.int_cfg.ode_options
    )  # Output: (Tp+1, N, 256, 16, 128)

# Return all except first (which is the context)
return m_future[1:, ...]  # (Tp, N, 256, 16, 128)
```

**What happens:** The ODE/SDE solver integrates the motion feature forward in time, predicting how motion evolves from the observed context to future frames.

---

### Method 3: `condition_enc(x)`

**Purpose:** Encodes input frames/differences using the motion encoder.

**Input:** `(N, T, C, H, W)` - Frames or frame differences
**Output:** `(N, T, C', H', W')` - Encoded features `(N, T, 256, 16, 128)`

```python
N, To, _, _, _ = x.shape
x = x.flatten(0, 1)  # (N*T, C, H, W) - Flatten batch and time
x = self.motion_encoder(x)  # (N*T, 256, 16, 128) - Encode each frame
return rearrange(x, '(N T) C H W -> N T C H W', N=N, T=To)  # Reshape back
```

---

## Class 2: `DiffUnet` (ODE/SDE Solver Network)

### Initialization: `__init__(diff_unet_cfg, int_cfg, motion_feature_size)`

**Purpose:** Creates the neural network that defines the ODE/SDE dynamics.

**Parameters:**
- `diff_unet_cfg`: `cfg.STDiff.DiffNet.DiffUnet`
- `int_cfg`: `cfg.STDiff.DiffNet.Int`
- `motion_feature_size`: `(256, 16, 128)` - Shape of motion features

#### Step-by-Step:

**Step 1: Extract Configuration (lines 116-120)**
```python
self.nonlienar = diff_unet_cfg.nonlinear  # 'tanh'
self.n_layers = diff_unet_cfg.n_layers  # 2
self.in_channels = diff_unet_cfg.in_channels  # 256
self.out_channels = self.in_channels  # 256 (same as input)
self.hidden_channels = diff_unet_cfg.hidden_channels  # 256
```

**Step 2: Create Drift Function (line 123)**
```python
self.diff_unet_f = OdeSdeFuncNet(
    in_channels=256,
    hidden_channels=256,
    out_channels=256,
    n_layers=2,
    nonlinear='tanh'
)
# This defines f(m, t) in the ODE/SDE: dm/dt = f(m, t) [+ g(m, t)*dW]
```

**Step 3: Create Diffusion Function (if SDE) (lines 125-129)**
```python
self.sde = int_cfg.sde  # True
if self.sde:
    self.diff_unet_g = OdeSdeFuncNet(...)  # Same structure as f
    # This defines g(m, t) in the SDE: dm/dt = f(m, t) + g(m, t) * dW
    self.noise_type = int_cfg.sde_options.noise_type  # 'diagonal'
    self.sde_type = int_cfg.sde_options.sde_type  # 'stratonovich'
```

**Note:** For ODE (when `sde=False`), only `diff_unet_f` is needed. For SDE, both `f` and `g` are needed.

---

### Method 1: `forward(t, x)` - For ODE

**Purpose:** Defines the ODE dynamics: `dm/dt = f(m, t)`

**Inputs:**
- `t`: Time (scalar, not used currently)
- `x`: `(N, C, H, W)` - Current motion feature state

**Output:** `(N, C, H, W)` - Rate of change `dm/dt`

```python
return self.diff_unet_f(x)  # Simply apply the drift network
```

---

### Method 2: `f(t, x)` - For SDE Drift

**Purpose:** Defines the drift term in SDE: `dm/dt = f(m, t) + g(m, t) * dW`

**Inputs:**
- `t`: Time (scalar, not used currently)
- `x`: `(N, C*H*W)` - Flattened motion feature

**Output:** `(N, C*H*W)` - Drift term (flattened)

```python
C, H, W = self.motion_feature_size  # 256, 16, 128
x = rearrange(x, 'N (C H W) -> N C H W', C=C, H=H, W=W)  # Reshape
x = self.diff_unet_f(x)  # Apply drift network
return x.flatten(1)  # Flatten back for SDE solver
```

---

### Method 3: `g(t, x)` - For SDE Diffusion

**Purpose:** Defines the diffusion term in SDE: `dm/dt = f(m, t) + g(m, t) * dW`

**Inputs:**
- `t`: Time (scalar, not used currently)
- `x`: `(N, C*H*W)` - Flattened motion feature

**Output:** `(N, C*H*W)` - Diffusion term (flattened)

```python
C, H, W = self.motion_feature_size
x = rearrange(x, 'N (C H W) -> N C H W', C=C, H=H, W=W)  # Reshape
x = self.diff_unet_g(x)  # Apply diffusion network
x = F.tanh(x)  # Bound the diffusion term
return x.flatten(1)  # Flatten back
```

**Why tanh?** The diffusion term `g(m, t)` is bounded to prevent numerical instability.

---

## Class 3: `OdeSdeFuncNet` (Neural Network for ODE/SDE)

### Initialization: `__init__(in_channels, hidden_channels, out_channels, n_layers, nonlinear)`

**Purpose:** Creates a CNN that defines the dynamics function.

**Parameters:**
- `in_channels`: 256 (motion feature channels)
- `hidden_channels`: 256
- `out_channels`: 256
- `n_layers`: 2 (from config)
- `nonlinear`: 'tanh' (from config)

#### Architecture (lines 167-175):

```python
layers = []
# Layer 1: Input conv
layers.append(nn.Conv2d(256, 256, 3, 1, 1))  # 3x3 conv, padding=1

# Hidden layers (n_layers = 2)
for i in range(2):
    layers.append(nn.Tanh())  # Activation
    layers.append(nn.Conv2d(256, 256, 3, 1, 1))  # 3x3 conv

# Final layers
layers.append(nn.Tanh())  # Activation
layers.append(zero_module(nn.Conv2d(256, 256, 3, 1, 1)))  # Output conv (zero-initialized)

self.net = nn.Sequential(*layers)
```

**Network Structure:**
```
Input (256, H, W)
  ↓
Conv2d(256→256, 3x3) + padding
  ↓
Tanh()
  ↓
Conv2d(256→256, 3x3) + padding
  ↓
Tanh()
  ↓
Conv2d(256→256, 3x3) + padding
  ↓
Tanh()
  ↓
ZeroConv2d(256→256, 3x3) + padding
  ↓
Output (256, H, W)
```

**Why zero-initialized output?** Ensures the network starts near zero, making training more stable.

---

## Class 4: `ConvGRUCell` (Convolutional GRU)

### Initialization: `__init__(in_channels, hidden_channels, kernel_size, stride, padding)`

**Purpose:** Recurrent cell for processing temporal sequences of motion features.

**Parameters:**
- `in_channels`: 256 (encoded frame features)
- `hidden_channels`: 256 (motion feature channels)
- `kernel_size`: 3
- `stride`: 1
- `padding`: 1

#### Architecture (lines 190-191):

```python
self.GateConv = nn.Conv2d(256+256, 2*256, 3, 1, 1)  # For update/reset gates
self.NewStateConv = nn.Conv2d(256+256, 256, 3, 1, 1)  # For candidate state
```

**Input:** `(256, H, W)` - Encoded frame features
**Hidden:** `(256, H, W)` - Motion feature state

---

### Forward: `forward(inputs, prev_h)`

**Purpose:** Updates motion feature state given new frame features.

**GRU Equations:**
```
u = σ(GateConv([inputs, prev_h])[:256])  # Update gate
r = σ(GateConv([inputs, prev_h])[256:])  # Reset gate
h_tilde = tanh(NewStateConv([inputs, r * prev_h]))  # Candidate state
new_h = (1 - u) * prev_h + u * h_tilde  # New state
```

**Step-by-Step (lines 199-203):**
```python
# Concatenate input and previous hidden state
concat = torch.cat((inputs, prev_h), dim=1)  # (512, H, W)

# Compute gates
gates = self.GateConv(concat)  # (512, H, W)
u, r = torch.split(gates, 256, dim=1)  # Split into update and reset gates
u, r = F.sigmoid(u), F.sigmoid(r)  # (256, H, W) each

# Compute candidate state
h_tilde = F.tanh(self.NewStateConv(torch.cat((inputs, r*prev_h), dim=1)))
# Uses reset gate to control how much of previous state to use

# Update state
new_h = (1 - u) * prev_h + u * h_tilde
# Update gate controls interpolation between old and new state
```

---

## Class 5: `MotionEncoder` (CNN Encoder)

### Initialization: `__init__(motion_encoder_cfg)`

**Purpose:** Encodes input frames into motion feature space.

**Parameters from config:**
- `in_channels`: 1 (grayscale)
- `model_channels`: 64 (base channels)
- `n_downs`: 2 (downsampling layers)

#### Architecture (lines 220-235):

```python
# Initial conv
Conv2d(1, 64, 5x5, padding=2) + ReLU
  ↓
# First downsampling
MaxPool2d(2)  # 64x512 → 32x256
Conv2d(64, 128, 5x5, padding=2) + ReLU
  ↓
# Second downsampling
MaxPool2d(2)  # 32x256 → 16x128
Conv2d(128, 256, 7x7, padding=3) + ReLU
  ↓
Output: (256, 16, 128)
```

**Step-by-Step:**
1. **Input:** `(1, 64, 512)` - Grayscale range image
2. **After first conv:** `(64, 64, 512)` - 64 channels, same size
3. **After first pool:** `(64, 32, 256)` - Downsampled by 2
4. **After second conv:** `(128, 32, 256)` - 128 channels
5. **After second pool:** `(128, 16, 128)` - Downsampled by 2
6. **After final conv:** `(256, 16, 128)` - 256 channels, final size

**Total downsampling:** 4x (2^2) in each dimension: 64→16, 512→128

---

## Complete Flow Example

### Training Step:

```
1. Input: Vo = (N, 3, 1, 64, 512) - 3 observed frames

2. context_encode(Vo, idx_o):
   a. Compute differences: (N, 2, 1, 64, 512)
   b. MotionEncoder: (N, 2, 256, 16, 128)
   c. ConvGRU processing: (N, 256, 16, 128) - motion context

3. future_predict(m_context, idx_p):
   a. SDE integration: (4, N, 256, 16, 128) - 3 future + 1 context
   b. Return: (3, N, 256, 16, 128) - future motion features

4. These motion features are used to condition the diffusion UNet
```

---

## Key Takeaways

1. **DiffUnet is NOT a diffusion model** - it's an ODE/SDE solver for motion features
2. **Motion features** are downsampled representations (256 channels, 16x128 spatial)
3. **SDE integration** predicts how motion evolves over time
4. **ConvGRU** processes temporal sequences of frames
5. **MotionEncoder** converts images to motion feature space
6. All parameters come from `kitti_range_train_config.yaml` under `STDiff.DiffNet`

---

## Parameter Summary Table

| Parameter | Config Path | Value | Meaning |
|-----------|-------------|-------|---------|
| `learn_diff_image` | `DiffNet.MotionEncoder.learn_diff_image` | True | Learn from frame differences |
| `image_size` | `DiffNet.MotionEncoder.image_size` | [64, 512] | Input image dimensions |
| `in_channels` | `DiffNet.MotionEncoder.in_channels` | 1 | Grayscale input |
| `model_channels` | `DiffNet.MotionEncoder.model_channels` | 64 | Base channel size |
| `n_downs` | `DiffNet.MotionEncoder.n_downs` | 2 | Downsampling layers |
| `n_layers` | `DiffNet.DiffUnet.n_layers` | 2 | ODE/SDE network layers |
| `nonlinear` | `DiffNet.DiffUnet.nonlinear` | 'tanh' | Activation function |
| `sde` | `DiffNet.Int.sde` | True | Use SDE (True) or ODE (False) |
| `method` | `DiffNet.Int.method` | 'euler_heun' | Integration method |
| `dt` | `DiffNet.Int.sde_options.dt` | 0.1 | Time step size |
| `rtol` | `DiffNet.Int.sde_options.rtol` | 1e-3 | Relative tolerance |
| `atol` | `DiffNet.Int.sde_options.atol` | 1e-3 | Absolute tolerance |
| `adaptive` | `DiffNet.Int.sde_options.adaptive` | False | Adaptive step size |
| `noise_type` | `DiffNet.Int.sde_options.noise_type` | 'diagonal' | SDE noise type |
| `sde_type` | `DiffNet.Int.sde_options.sde_type` | 'stratonovich' | SDE interpretation |

