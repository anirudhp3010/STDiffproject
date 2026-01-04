# Memory Usage Analysis for STDiff Training

## Memory-Intensive Components Identified

### 1. **Input Data Loading** (train_stdiff.py:236-253)
**Location:** `train_stdiff.py` lines 236-253

**Memory Usage:**
- `Vo`: (N=2, To=5, C=1, H=64, W=2048) = **2 × 5 × 1 × 64 × 2048 × 4 bytes = ~5.2 MB**
- `Vp`: (N=2, Tp=5, C=1, H=64, W=2048) = **2 × 5 × 1 × 64 × 2048 × 4 bytes = ~5.2 MB**
- `Vo_last_frame`: (N=2, C=1, H=64, W=2048) = **~1 MB**
- `Vo_mask`, `Vp_mask`: Same size as Vo/Vp = **~10.4 MB**
- **Total batch data: ~22 MB** (just input tensors)

### 2. **Flattened Images** (train_stdiff.py:248)
**Location:** `train_stdiff.py` line 248

```python
clean_images = Vp.flatten(0, 1)  # (N*Tp, C, H, W) = (10, 1, 64, 2048)
```
- Creates new tensor: **10 × 1 × 64 × 2048 × 4 bytes = ~5.2 MB**
- Original Vp still in memory = **duplicate memory**

### 3. **Noise Generation** (train_stdiff.py:268)
**Location:** `train_stdiff.py` line 268

```python
noise = torch.randn(clean_images.shape).to(clean_images.device)
```
- Noise tensor: **(10, 1, 64, 2048) = ~5.2 MB**

### 4. **Noisy Images** (train_stdiff.py:277)
**Location:** `train_stdiff.py` line 277

```python
noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
```
- Another tensor: **(10, 1, 64, 2048) = ~5.2 MB**
- **Now have: clean_images + noise + noisy_images = ~15.6 MB**

### 5. **Model Forward Pass** (stdiff_diffusers.py:23-49)
**Location:** `stdiff_diffusers.py` forward method

#### 5a. Context Encoding (line 25)
```python
m_context = self.tde_model.context_encode(Vo, idx_o)  # (N, C, H, W)
```
- Encodes all observed frames: **(2, C, H, W)** where C depends on MotionEncoder
- Memory: **~2 × 64 × 64 × 2048 × 4 bytes** (if same resolution) or downsampled

#### 5b. Future Motion Prediction (line 28)
```python
m_future = self.tde_model.future_predict(m_context, ...)  # (Tp, N, C, H, W)
```
- Motion features: **(5, 2, C, H, W)** = **5 × 2 × C × 64 × 2048 × 4 bytes**
- If C=64 (model_channels): **~33 MB**

#### 5c. Autoregressive Concatenation (line 44-45)
```python
prev_frames = torch.cat([Vo_last_frame, clean_Vp[:, 0:-1, ...]], dim=1)
noisy_Vp = torch.cat([noisy_Vp, prev_frames.flatten(0, 1)], dim=1)
```
- Concatenates along channel dimension: **(10, 2, 64, 2048)** = **~10.4 MB**
- **Original noisy_Vp still in memory = duplicate**

### 6. **UNet2DMotionCond Forward** (stdiff_diffusers.py:47)
**Location:** `stdiff_diffusers.py` line 47

```python
out = self.diffusion_unet(noisy_Vp, timestep, m_feat=m_future.permute(...).flatten(0, 1))
```

**UNet Architecture (from config):**
- Input: **(10, 2, 64, 2048)**
- Block channels: [128, 256, 256, 512, 512]
- 5 down blocks + 5 up blocks with attention

**Memory per layer:**
- **Down Block 1:** (10, 128, 32, 1024) = **~16.8 MB**
- **Down Block 2:** (10, 256, 16, 512) = **~8.4 MB** + **Attention: O(512×512) = ~1 MB**
- **Down Block 3:** (10, 256, 8, 256) = **~2.1 MB** + **Attention: O(256×256) = ~0.25 MB**
- **Down Block 4:** (10, 512, 4, 128) = **~1.0 MB** + **Attention: O(128×128) = ~0.06 MB**
- **Down Block 5:** (10, 512, 2, 64) = **~0.25 MB** + **Attention: O(64×64) = ~0.02 MB**

**Attention Memory (Quadratic):**
- Attention at resolution 1024×512: **1024 × 512 × 4 bytes × 2 (Q, K) = ~4 MB per head**
- With 128 head_dim and multiple heads: **~16-32 MB per attention layer**
- **4 attention layers in down path = ~64-128 MB**

**Skip Connections:**
- All intermediate features stored for upsampling: **~30-50 MB**

**Total UNet Memory: ~150-200 MB** (just activations)

### 7. **Model Output** (train_stdiff.py:279)
**Location:** `train_stdiff.py` line 279

```python
model_output = model(...).sample  # (10, 1, 64, 2048)
```
- Output tensor: **~5.2 MB**

### 8. **Loss Computation** (train_stdiff.py:282-303)
**Location:** `train_stdiff.py` lines 282-303

```python
loss_per_pixel = F.l1_loss(model_output, noise, reduction="none")  # (10, 1, 64, 2048)
valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(loss_per_pixel)  # (10, 1, 64, 2048)
```
- `loss_per_pixel`: **~5.2 MB**
- `valid_mask_expanded`: **~5.2 MB** (though expand_as doesn't copy, just view)
- **Total: ~5.2 MB**

### 9. **Gradient Computation** (train_stdiff.py:305)
**Location:** `train_stdiff.py` line 305

```python
accelerator.backward(loss)
```
- **Gradients for all model parameters: ~2-3× model size**
- Model parameters: ~50-100M parameters = **~200-400 MB** (fp32) or **~100-200 MB** (fp16)
- **Gradients: ~200-400 MB** (fp32) or **~100-200 MB** (fp16)

### 10. **EMA Model** (train_stdiff.py:140-149)
**Location:** `train_stdiff.py` lines 140-149

```python
ema_model = EMAModel(model.parameters(), ...)
```
- **Full copy of model: ~200-400 MB** (fp32) or **~100-200 MB** (fp16)

### 11. **Gradient Accumulation** (train_stdiff.py:237, config line 90)
**Location:** `train_stdiff.py` line 237, config gradient_accumulation_steps: 2

```python
with accelerator.accumulate(model):
```
- Stores gradients for 2 steps before updating
- **Effectively doubles gradient memory during accumulation**

## Total Memory Estimate

### Per Batch:
- Input data: **~22 MB**
- Intermediate tensors: **~30 MB**
- UNet activations: **~150-200 MB**
- Model parameters: **~200-400 MB** (fp32) or **~100-200 MB** (fp16)
- Gradients: **~200-400 MB** (fp32) or **~100-200 MB** (fp16)
- EMA model: **~200-400 MB** (fp32) or **~100-200 MB** (fp16)

### Total GPU Memory:
- **FP32: ~800-1000 MB per batch**
- **FP16: ~500-700 MB per batch** (with mixed precision)

### Peak Memory (with gradient accumulation):
- **FP32: ~1.2-1.5 GB**
- **FP16: ~700-900 MB**

## Memory Bottlenecks (Ranked)

1. **UNet Attention Layers** - Quadratic memory O(H×W×H×W) at 2048 width
2. **EMA Model** - Full model copy
3. **Gradient Storage** - 2-3× model size
4. **Multiple Frame Processing** - 10 frames per batch (5 observed + 5 predicted)
5. **Skip Connections** - All intermediate features stored
6. **Gradient Accumulation** - Stores gradients for 2 steps

## Recommendations

1. **Enable gradient checkpointing** in UNet
2. **Reduce attention head dimensions** from 128 to 64
3. **Use smaller batch size** (already at 2)
4. **Disable EMA** if not critical (saves ~200-400 MB)
5. **Reduce number of frames** (5→3)
6. **Use fp16 mixed precision** (already enabled)
7. **Reduce UNet channels** in config
8. **Use linear attention** instead of full attention for wide images

