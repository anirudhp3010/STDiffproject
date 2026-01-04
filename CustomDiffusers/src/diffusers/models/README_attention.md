# Attention Module - Line-by-Line Documentation

This document provides a comprehensive line-by-line explanation of `attention.py`, which implements various attention mechanisms and normalization layers used in diffusion models.

## Table of Contents

1. [Imports and Dependencies](#imports-and-dependencies)
2. [AttentionBlock Class](#attentionblock-class)
3. [BasicTransformerBlock Class](#basictransformerblock-class)
4. [FeedForward Class](#feedforward-class)
5. [Activation Functions](#activation-functions)
6. [Normalization Layers](#normalization-layers)

---

## Imports and Dependencies

### Lines 1-13: Copyright and License Header
```python
# Copyright 2023 The HuggingFace Team. All rights reserved.
# ... License text ...
```
**Explanation**: Standard Apache 2.0 license header for HuggingFace codebase.

### Line 14: Math Module
```python
import math
```
**Explanation**: Used for mathematical operations, particularly `sqrt()` for attention scaling.

### Line 15: Type Hints
```python
from typing import Callable, Optional
```
**Explanation**: 
- `Callable`: Type hint for function/callable objects
- `Optional`: Type hint for values that can be None

### Lines 17-19: PyTorch Imports
```python
import torch
import torch.nn.functional as F
from torch import nn
```
**Explanation**:
- `torch`: Core PyTorch library
- `F`: Functional API for operations like `gelu()`, `group_norm()`
- `nn`: Neural network module classes

### Lines 21-23: Custom Imports
```python
from ..utils.import_utils import is_xformers_available
from .attention_processor import Attention
from .embeddings import CombinedTimestepLabelEmbeddings
```
**Explanation**:
- `is_xformers_available()`: Checks if xformers library is installed
- `Attention`: Attention processor class (used in BasicTransformerBlock)
- `CombinedTimestepLabelEmbeddings`: Embedding for timestep + class labels

### Lines 26-30: XFormers Conditional Import
```python
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None
```
**Explanation**: 
- Conditionally imports xformers library if available
- `xformers.ops`: Contains memory-efficient attention operations
- If not available, sets `xformers = None` to avoid import errors

---

## AttentionBlock Class

### Lines 33-47: Class Definition and Docstring
```python
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other...
    """
```
**Explanation**: 
- Inherits from `nn.Module` (PyTorch base class for neural network modules)
- **Note**: This class is deprecated (see line 49)
- Implements self-attention for spatial features

### Line 49: Deprecation Warning
```python
# IMPORTANT;TODO(Patrick, William) - this class will be deprecated soon. Do not use it anymore
```
**Explanation**: Explicit warning that this class should not be used in new code.

### Lines 51-58: `__init__` Method Signature
```python
def __init__(
    self,
    channels: int,
    num_head_channels: Optional[int] = None,
    norm_num_groups: int = 32,
    rescale_output_factor: float = 1.0,
    eps: float = 1e-5,
):
```
**Explanation**:
- `channels`: Number of input/output channels
- `num_head_channels`: Channels per attention head (if None, uses single head)
- `norm_num_groups`: Groups for GroupNorm (default 32)
- `rescale_output_factor`: Factor to rescale output (default 1.0)
- `eps`: Epsilon for normalization (default 1e-5)

### Line 59: Super Initialization
```python
super().__init__()
```
**Explanation**: Calls parent `nn.Module.__init__()` to initialize the module.

### Lines 60-63: Store Parameters and Compute Number of Heads
```python
self.channels = channels
self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
self.num_head_size = num_head_channels
```
**Explanation**:
- **Line 60**: Store total channels
- **Line 62**: Calculate number of attention heads:
  - If `num_head_channels` is provided: `num_heads = channels / num_head_channels`
  - If None: single head attention (`num_heads = 1`)
- **Line 63**: Store head size for later use

### Line 64: Group Normalization Layer
```python
self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True)
```
**Explanation**:
- **GroupNorm**: Normalizes across channel groups (more stable than BatchNorm for small batches)
- `num_channels`: Total channels to normalize
- `num_groups`: Number of groups (e.g., 32 groups for 224 channels = 7 channels per group)
- `affine=True`: Learnable scale and shift parameters

### Lines 66-69: Query, Key, Value Linear Layers
```python
# define q,k,v as linear layers
self.query = nn.Linear(channels, channels)
self.key = nn.Linear(channels, channels)
self.value = nn.Linear(channels, channels)
```
**Explanation**:
- **Self-attention mechanism**: Each position computes attention over all positions
- `query`: Projects input to query space (what to look for)
- `key`: Projects input to key space (what is available)
- `value`: Projects input to value space (what to return)
- All map from `channels` → `channels` dimensions

### Line 71: Rescale Factor
```python
self.rescale_output_factor = rescale_output_factor
```
**Explanation**: Stores the rescaling factor for output normalization.

### Line 72: Output Projection
```python
self.proj_attn = nn.Linear(channels, channels, 1)
```
**Explanation**:
- Projects attention output back to original dimension
- `bias=1` (third argument) means bias is enabled (default)
- Maps `channels` → `channels`

### Lines 74-75: Memory Efficient Attention Flags
```python
self._use_memory_efficient_attention_xformers = False
self._attention_op = None
```
**Explanation**:
- Flags for xformers memory-efficient attention
- `_use_memory_efficient_attention_xformers`: Whether to use xformers
- `_attention_op`: Optional custom attention operation

### Lines 77-82: Reshape Heads to Batch Dimension
```python
def reshape_heads_to_batch_dim(self, tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = self.num_heads
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
    return tensor
```
**Explanation**:
- **Purpose**: Reshapes tensor for multi-head attention computation
- **Input**: `(batch_size, seq_len, dim)` - e.g., `(4, 256, 224)`
- **Line 78**: Extract dimensions
- **Line 80**: Reshape to `(batch_size, seq_len, num_heads, dim_per_head)`
  - Example: `(4, 256, 224)` → `(4, 256, 8, 28)` if `num_heads=8`
- **Line 81**: Permute and reshape to `(batch_size * num_heads, seq_len, dim_per_head)`
  - Permute: `(0, 2, 1, 3)` swaps head and sequence dimensions
  - Final: `(32, 256, 28)` - treats each head as separate batch item
- **Why**: Allows parallel computation of all heads

### Lines 84-89: Reshape Batch Dimension Back to Heads
```python
def reshape_batch_dim_to_heads(self, tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = self.num_heads
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
    return tensor
```
**Explanation**:
- **Purpose**: Reverse operation - combines heads back
- **Input**: `(batch_size * num_heads, seq_len, dim_per_head)` - e.g., `(32, 256, 28)`
- **Line 87**: Reshape to `(batch_size, num_heads, seq_len, dim_per_head)`
- **Line 88**: Permute and reshape to `(batch_size, seq_len, dim)`
  - Permute: `(0, 2, 1, 3)` swaps back
  - Final: `(4, 256, 224)` - concatenated heads

### Lines 91-119: Set Memory Efficient Attention
```python
def set_use_memory_efficient_attention_xformers(
    self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
):
```
**Explanation**: Method to enable/disable xformers memory-efficient attention.

#### Line 94: Check if Enabling
```python
if use_memory_efficient_attention_xformers:
```
**Explanation**: Only validate if enabling xformers.

#### Lines 95-102: Check XFormers Availability
```python
if not is_xformers_available():
    raise ModuleNotFoundError(
        (
            "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
            " xformers"
        ),
        name="xformers",
    )
```
**Explanation**: Raises error if xformers is not installed.

#### Lines 103-107: Check CUDA Availability
```python
elif not torch.cuda.is_available():
    raise ValueError(
        "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
        " only available for GPU "
    )
```
**Explanation**: XFormers memory-efficient attention requires CUDA/GPU.

#### Lines 109-117: Test XFormers
```python
else:
    try:
        # Make sure we can run the memory efficient attention
        _ = xformers.ops.memory_efficient_attention(
            torch.randn((1, 2, 40), device="cuda"),
            torch.randn((1, 2, 40), device="cuda"),
            torch.randn((1, 2, 40), device="cuda"),
        )
    except Exception as e:
        raise e
```
**Explanation**:
- Tests xformers with dummy tensors on CUDA
- If test fails, raises the exception
- Ensures xformers works before enabling

#### Lines 118-119: Set Flags
```python
self._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
self._attention_op = attention_op
```
**Explanation**: Store the settings for use in forward pass.

### Lines 121-174: Forward Pass

#### Line 122: Save Residual
```python
residual = hidden_states
```
**Explanation**: Save input for residual connection (skip connection).

#### Line 123: Extract Dimensions
```python
batch, channel, height, width = hidden_states.shape
```
**Explanation**: Extract spatial dimensions from 4D tensor `(B, C, H, W)`.

#### Lines 125-126: Normalize
```python
# norm
hidden_states = self.group_norm(hidden_states)
```
**Explanation**: Apply GroupNorm for stable training.

#### Line 128: Flatten Spatial Dimensions
```python
hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)
```
**Explanation**:
- **View**: Reshape `(B, C, H, W)` → `(B, C, H*W)`
- **Transpose**: Swap channel and spatial dims → `(B, H*W, C)`
- **Why**: Attention operates on sequence of tokens (spatial positions)

#### Lines 130-133: Project to Q, K, V
```python
# proj to q, k, v
query_proj = self.query(hidden_states)
key_proj = self.key(hidden_states)
value_proj = self.value(hidden_states)
```
**Explanation**:
- Project normalized features to query, key, value spaces
- All outputs: `(B, H*W, C)`

#### Line 135: Compute Attention Scale
```python
scale = 1 / math.sqrt(self.channels / self.num_heads)
```
**Explanation**:
- **Scaled dot-product attention**: Prevents large dot products
- Scale = `1 / sqrt(d_k)` where `d_k = channels / num_heads` (dim per head)
- Example: If `channels=224`, `num_heads=8`, then `scale = 1/sqrt(28) ≈ 0.189`

#### Lines 137-139: Reshape for Multi-Head Attention
```python
query_proj = self.reshape_heads_to_batch_dim(query_proj)
key_proj = self.reshape_heads_to_batch_dim(key_proj)
value_proj = self.reshape_heads_to_batch_dim(value_proj)
```
**Explanation**: Reshape to `(B*num_heads, H*W, C/num_heads)` for parallel head computation.

#### Lines 141-146: XFormers Memory Efficient Attention
```python
if self._use_memory_efficient_attention_xformers:
    # Memory efficient attention
    hidden_states = xformers.ops.memory_efficient_attention(
        query_proj, key_proj, value_proj, attn_bias=None, op=self._attention_op
    )
    hidden_states = hidden_states.to(query_proj.dtype)
```
**Explanation**:
- **XFormers**: Optimized attention implementation (faster, less memory)
- `attn_bias=None`: No attention bias/mask
- `op`: Optional custom operation
- **Line 146**: Ensure output dtype matches input

#### Lines 147-162: Standard Attention (if not using xformers)
```python
else:
    attention_scores = torch.baddbmm(
        torch.empty(
            query_proj.shape[0],
            query_proj.shape[1],
            key_proj.shape[1],
            dtype=query_proj.dtype,
            device=query_proj.device,
        ),
        query_proj,
        key_proj.transpose(-1, -2),
        beta=0,
        alpha=scale,
    )
    attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)
    hidden_states = torch.bmm(attention_probs, value_proj)
```
**Explanation**:
- **Line 148-160**: `torch.baddbmm` computes `Q @ K^T * scale`
  - Creates empty output tensor with correct shape
  - `beta=0`: Don't add to output tensor
  - `alpha=scale`: Multiply result by scale factor
  - Result: `(B*heads, H*W, H*W)` attention scores
- **Line 161**: Apply softmax to get attention probabilities
  - Convert to float for numerical stability
  - Softmax along last dimension (over keys)
  - Convert back to original dtype
- **Line 162**: `torch.bmm` computes `attention_probs @ V`
  - Batch matrix multiplication
  - Result: `(B*heads, H*W, C/heads)` weighted values

#### Line 165: Reshape Back
```python
hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
```
**Explanation**: Combine heads back: `(B*heads, H*W, C/heads)` → `(B, H*W, C)`.

#### Line 168: Output Projection
```python
hidden_states = self.proj_attn(hidden_states)
```
**Explanation**: Project attention output: `(B, H*W, C)` → `(B, H*W, C)`.

#### Line 170: Reshape to Spatial Format
```python
hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)
```
**Explanation**:
- **Transpose**: `(B, H*W, C)` → `(B, C, H*W)`
- **Reshape**: `(B, C, H*W)` → `(B, C, H, W)`
- Restores original spatial structure

#### Lines 172-174: Residual Connection and Rescale
```python
# res connect and rescale
hidden_states = (hidden_states + residual) / self.rescale_output_factor
return hidden_states
```
**Explanation**:
- **Residual connection**: Add original input (skip connection)
- **Rescale**: Divide by `rescale_output_factor` (typically 1.0)
- **Why residual**: Helps gradient flow and training stability

---

## BasicTransformerBlock Class

### Lines 177-192: Class Definition
```python
class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.
    ...
    """
```
**Explanation**: Standard transformer block with self-attention, cross-attention, and feed-forward layers.

### Lines 194-209: `__init__` Parameters
```python
def __init__(
    self,
    dim: int,
    num_attention_heads: int,
    attention_head_dim: int,
    dropout=0.0,
    cross_attention_dim: Optional[int] = None,
    activation_fn: str = "geglu",
    num_embeds_ada_norm: Optional[int] = None,
    attention_bias: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    norm_elementwise_affine: bool = True,
    norm_type: str = "layer_norm",
    final_dropout: bool = False,
):
```
**Explanation**:
- `dim`: Feature dimension
- `num_attention_heads`: Number of attention heads
- `attention_head_dim`: Dimension per head
- `dropout`: Dropout probability
- `cross_attention_dim`: Dimension for cross-attention (encoder features)
- `activation_fn`: Activation for feed-forward ("geglu", "gelu", etc.)
- `num_embeds_ada_norm`: Number of embeddings for adaptive normalization
- `attention_bias`: Whether attention layers have bias
- `only_cross_attention`: If True, first attention is cross-attention
- `upcast_attention`: Whether to upcast attention to float32
- `norm_elementwise_affine`: Whether LayerNorm has learnable parameters
- `norm_type`: "layer_norm", "ada_norm", or "ada_norm_zero"
- `final_dropout`: Whether to apply dropout after feed-forward

### Line 210: Super Initialization
```python
super().__init__()
```

### Line 211: Store Flag
```python
self.only_cross_attention = only_cross_attention
```

### Lines 213-214: Adaptive Normalization Flags
```python
self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
```
**Explanation**:
- **AdaNorm**: Adaptive normalization that uses timestep embeddings
- **AdaNormZero**: Variant that also returns modulation parameters

### Lines 216-220: Validate AdaNorm Parameters
```python
if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
    raise ValueError(
        f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
        f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
    )
```
**Explanation**: Ensures `num_embeds_ada_norm` is provided when using adaptive norms.

### Lines 222-231: Self-Attention Layer
```python
# 1. Self-Attn
self.attn1 = Attention(
    query_dim=dim,
    heads=num_attention_heads,
    dim_head=attention_head_dim,
    dropout=dropout,
    bias=attention_bias,
    cross_attention_dim=cross_attention_dim if only_cross_attention else None,
    upcast_attention=upcast_attention,
)
```
**Explanation**:
- Creates self-attention layer
- If `only_cross_attention=True`, this becomes cross-attention (uses `cross_attention_dim`)

### Line 233: Feed-Forward Layer
```python
self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
```

### Lines 235-247: Cross-Attention Layer
```python
# 2. Cross-Attn
if cross_attention_dim is not None:
    self.attn2 = Attention(
        query_dim=dim,
        cross_attention_dim=cross_attention_dim,
        heads=num_attention_heads,
        dim_head=attention_head_dim,
        dropout=dropout,
        bias=attention_bias,
        upcast_attention=upcast_attention,
    )  # is self-attn if encoder_hidden_states is none
else:
    self.attn2 = None
```
**Explanation**:
- Creates cross-attention if `cross_attention_dim` is provided
- Cross-attention attends to encoder features (e.g., text embeddings)
- If None, no cross-attention layer

### Lines 249-254: First Normalization Layer
```python
if self.use_ada_layer_norm:
    self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
elif self.use_ada_layer_norm_zero:
    self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
else:
    self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
```
**Explanation**: Creates normalization layer before self-attention (adaptive or standard).

### Lines 256-266: Second Normalization Layer
```python
if cross_attention_dim is not None:
    # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
    # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
    # the second cross attention block.
    self.norm2 = (
        AdaLayerNorm(dim, num_embeds_ada_norm)
        if self.use_ada_layer_norm
        else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
    )
else:
    self.norm2 = None
```
**Explanation**:
- Creates normalization before cross-attention
- **Note**: `AdaLayerNormZero` not used here (comment explains why)
- If no cross-attention, `norm2 = None`

### Lines 268-269: Third Normalization Layer
```python
# 3. Feed-forward
self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
```
**Explanation**: Normalization before feed-forward (always standard LayerNorm).

### Lines 271-280: Forward Method Signature
```python
def forward(
    self,
    hidden_states,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    timestep=None,
    cross_attention_kwargs=None,
    class_labels=None,
):
```
**Explanation**:
- `hidden_states`: Input features
- `attention_mask`: Mask for self-attention
- `encoder_hidden_states`: Features for cross-attention (e.g., text embeddings)
- `encoder_attention_mask`: Mask for cross-attention
- `timestep`: Timestep for adaptive normalization
- `cross_attention_kwargs`: Additional kwargs for cross-attention
- `class_labels`: Class labels for adaptive normalization

### Lines 281-288: First Normalization
```python
if self.use_ada_layer_norm:
    norm_hidden_states = self.norm1(hidden_states, timestep)
elif self.use_ada_layer_norm_zero:
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
        hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
    )
else:
    norm_hidden_states = self.norm1(hidden_states)
```
**Explanation**:
- **AdaNorm**: Returns normalized features
- **AdaNormZero**: Returns normalized features + modulation parameters
  - `gate_msa`: Gate for multi-head self-attention
  - `shift_mlp`, `scale_mlp`: Shift/scale for MLP
  - `gate_mlp`: Gate for MLP
- **Standard**: Just normalized features

### Lines 290-300: Self-Attention
```python
# 1. Self-Attention
cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
attn_output = self.attn1(
    norm_hidden_states,
    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
    attention_mask=attention_mask,
    **cross_attention_kwargs,
)
if self.use_ada_layer_norm_zero:
    attn_output = gate_msa.unsqueeze(1) * attn_output
hidden_states = attn_output + hidden_states
```
**Explanation**:
- **Line 291**: Initialize kwargs dict if None
- **Lines 292-297**: Apply self-attention
  - If `only_cross_attention=True`, uses `encoder_hidden_states` as cross-attention
- **Lines 298-299**: If AdaNormZero, gate the attention output
- **Line 300**: Residual connection

### Lines 302-316: Cross-Attention
```python
if self.attn2 is not None:
    norm_hidden_states = (
        self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
    )
    # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
    # prepare attention mask here

    # 2. Cross-Attention
    attn_output = self.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=encoder_attention_mask,
        **cross_attention_kwargs,
    )
    hidden_states = attn_output + hidden_states
```
**Explanation**:
- **Lines 303-305**: Normalize (adaptive or standard)
- **Lines 310-315**: Apply cross-attention to encoder features
- **Line 316**: Residual connection

### Lines 318-329: Feed-Forward
```python
# 3. Feed-forward
norm_hidden_states = self.norm3(hidden_states)

if self.use_ada_layer_norm_zero:
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

ff_output = self.ff(norm_hidden_states)

if self.use_ada_layer_norm_zero:
    ff_output = gate_mlp.unsqueeze(1) * ff_output

hidden_states = ff_output + hidden_states
```
**Explanation**:
- **Line 319**: Normalize before feed-forward
- **Lines 321-322**: If AdaNormZero, apply shift/scale modulation
- **Line 324**: Apply feed-forward network
- **Lines 326-327**: If AdaNormZero, gate the output
- **Line 329**: Residual connection

### Line 331: Return
```python
return hidden_states
```

---

## FeedForward Class

### Lines 334-345: Class Definition
```python
class FeedForward(nn.Module):
    r"""
    A feed-forward layer.
    ...
    """
```

### Lines 347-355: `__init__` Method
```python
def __init__(
    self,
    dim: int,
    dim_out: Optional[int] = None,
    mult: int = 4,
    dropout: float = 0.0,
    activation_fn: str = "geglu",
    final_dropout: bool = False,
):
    super().__init__()
    inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim
```
**Explanation**:
- `dim`: Input dimension
- `dim_out`: Output dimension (defaults to `dim`)
- `mult`: Multiplier for hidden dimension (default 4)
- `inner_dim`: Hidden dimension = `dim * mult`

### Lines 360-367: Select Activation Function
```python
if activation_fn == "gelu":
    act_fn = GELU(dim, inner_dim)
if activation_fn == "gelu-approximate":
    act_fn = GELU(dim, inner_dim, approximate="tanh")
elif activation_fn == "geglu":
    act_fn = GEGLU(dim, inner_dim)
elif activation_fn == "geglu-approximate":
    act_fn = ApproximateGELU(dim, inner_dim)
```
**Explanation**: Creates appropriate activation module.

### Lines 369-378: Build Network
```python
self.net = nn.ModuleList([])
# project in
self.net.append(act_fn)
# project dropout
self.net.append(nn.Dropout(dropout))
# project out
self.net.append(nn.Linear(inner_dim, dim_out))
# FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
if final_dropout:
    self.net.append(nn.Dropout(dropout))
```
**Explanation**:
- **ModuleList**: Sequential container for modules
- **Order**: Activation → Dropout → Linear projection → (optional) Dropout
- Activation projects `dim → inner_dim`
- Linear projects `inner_dim → dim_out`

### Lines 380-383: Forward Pass
```python
def forward(self, hidden_states):
    for module in self.net:
        hidden_states = module(hidden_states)
    return hidden_states
```
**Explanation**: Applies modules sequentially.

---

## Activation Functions

### GELU Class (Lines 386-405)

#### Lines 391-394: Initialization
```python
def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
    super().__init__()
    self.proj = nn.Linear(dim_in, dim_out)
    self.approximate = approximate
```
**Explanation**: Linear projection + GELU activation.

#### Lines 396-400: GELU Method
```python
def gelu(self, gate):
    if gate.device.type != "mps":
        return F.gelu(gate, approximate=self.approximate)
    # mps: gelu is not implemented for float16
    return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)
```
**Explanation**:
- **MPS**: Apple Metal Performance Shaders
- GELU not implemented for float16 on MPS, so convert to float32 and back

#### Lines 402-405: Forward
```python
def forward(self, hidden_states):
    hidden_states = self.proj(hidden_states)
    hidden_states = self.gelu(hidden_states)
    return hidden_states
```

### GEGLU Class (Lines 408-429)

#### Lines 417-419: Initialization
```python
def __init__(self, dim_in: int, dim_out: int):
    super().__init__()
    self.proj = nn.Linear(dim_in, dim_out * 2)
```
**Explanation**: Projects to `2 * dim_out` (for gating mechanism).

#### Lines 421-425: GELU Method
```python
def gelu(self, gate):
    if gate.device.type != "mps":
        return F.gelu(gate)
    # mps: gelu is not implemented for float16
    return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)
```

#### Lines 427-429: Forward
```python
def forward(self, hidden_states):
    hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
    return hidden_states * self.gelu(gate)
```
**Explanation**:
- **GEGLU**: Gated GELU Linear Unit
- Split projection into two halves
- One half is gated by GELU of the other: `output = x * GELU(gate)`

### ApproximateGELU Class (Lines 432-445)

#### Lines 439-441: Initialization
```python
def __init__(self, dim_in: int, dim_out: int):
    super().__init__()
    self.proj = nn.Linear(dim_in, dim_out)
```

#### Lines 443-445: Forward
```python
def forward(self, x):
    x = self.proj(x)
    return x * torch.sigmoid(1.702 * x)
```
**Explanation**: Approximate GELU using sigmoid: `x * sigmoid(1.702 * x)` (faster than exact GELU).

---

## Normalization Layers

### AdaLayerNorm Class (Lines 448-464)

#### Lines 453-458: Initialization
```python
def __init__(self, embedding_dim, num_embeddings):
    super().__init__()
    self.emb = nn.Embedding(num_embeddings, embedding_dim)
    self.silu = nn.SiLU()
    self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
    self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
```
**Explanation**:
- **Embedding**: Maps timestep to embedding
- **Linear**: Projects to scale and shift parameters
- **LayerNorm**: Without learnable parameters (`elementwise_affine=False`)

#### Lines 460-464: Forward
```python
def forward(self, x, timestep):
    emb = self.linear(self.silu(self.emb(timestep)))
    scale, shift = torch.chunk(emb, 2)
    x = self.norm(x) * (1 + scale) + shift
    return x
```
**Explanation**:
- **Line 461**: Embed timestep → SiLU → Linear → `(scale, shift)`
- **Line 462**: Split into scale and shift
- **Line 463**: Adaptive normalization: `norm(x) * (1 + scale) + shift`

### AdaLayerNormZero Class (Lines 467-485)

#### Lines 472-479: Initialization
```python
def __init__(self, embedding_dim, num_embeddings):
    super().__init__()
    self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
    self.silu = nn.SiLU()
    self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
    self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
```
**Explanation**:
- **CombinedTimestepLabelEmbeddings**: Combines timestep + class label embeddings
- **Linear**: Projects to 6 parameters (scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp)

#### Lines 481-485: Forward
```python
def forward(self, x, timestep, class_labels, hidden_dtype=None):
    emb = self.linear(self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
    x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
    return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
```
**Explanation**:
- **Line 482**: Generate 6 modulation parameters
- **Line 483**: Split into 6 chunks
- **Line 484**: Apply normalization with MSA parameters
- **Line 485**: Return normalized features + MLP parameters (for later use)

### AdaGroupNorm Class (Lines 488-520)

#### Lines 493-509: Initialization
```python
def __init__(
    self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
):
    super().__init__()
    self.num_groups = num_groups
    self.eps = eps
    self.act = None
    if act_fn == "swish":
        self.act = lambda x: F.silu(x)
    elif act_fn == "mish":
        self.act = nn.Mish()
    elif act_fn == "silu":
        self.act = nn.SiLU()
    elif act_fn == "gelu":
        self.act = nn.GELU()
    self.linear = nn.Linear(embedding_dim, out_dim * 2)
```
**Explanation**:
- Adaptive GroupNorm (for 2D/3D tensors, not sequences)
- Optional activation for embedding
- Linear projects to scale and shift

#### Lines 511-520: Forward
```python
def forward(self, x, emb):
    if self.act:
        emb = self.act(emb)
    emb = self.linear(emb)
    emb = emb[:, :, None, None]
    scale, shift = emb.chunk(2, dim=1)
    x = F.group_norm(x, self.num_groups, eps=self.eps)
    x = x * (1 + scale) + shift
    return x
```
**Explanation**:
- **Line 512**: Apply activation to embedding if provided
- **Line 513**: Project to scale and shift
- **Line 514**: Add spatial dimensions `[None, None]` for broadcasting
- **Line 515**: Split into scale and shift
- **Line 516**: Apply GroupNorm
- **Line 517**: Apply adaptive modulation
- **Usage**: For 2D/3D feature maps (e.g., in ResNet blocks)

---

## Summary

This module provides:

1. **AttentionBlock**: Deprecated self-attention block for spatial features
2. **BasicTransformerBlock**: Full transformer block with self/cross-attention and feed-forward
3. **FeedForward**: MLP with various activation functions
4. **Activation Functions**: GELU, GEGLU, ApproximateGELU
5. **Normalization Layers**: AdaLayerNorm, AdaLayerNormZero, AdaGroupNorm

All components support timestep-conditioned operations for diffusion models.

