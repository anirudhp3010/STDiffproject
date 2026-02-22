"""
Modular REPA projection loss: negative cosine similarity between
projected diffusion features and encoder (ScaLR) features.
Same formulation as U-REPA.
"""
import torch
import torch.nn.functional as F


def projection_loss(
    z: torch.Tensor,
    z_tilde: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Negative cosine similarity (alignment loss).
    Both inputs are L2-normalized; we minimize -cos_sim = -(z · z_tilde).

    Args:
        z: (B, N, C) or (B, C) - encoder features (ScaLR), normalized
        z_tilde: (B, N, C) or (B, C) - projected model features, will be normalized
        reduce: 'mean' or 'sum'

    Returns:
        Scalar loss (higher = worse alignment)
    """
    z_tilde = F.normalize(z_tilde, dim=-1)
    z = F.normalize(z, dim=-1)

    # Cosine similarity: (z * z_tilde).sum(dim=-1) in [-1, 1]
    # Loss = -cos_sim -> minimize to maximize alignment
    if z.dim() == 3:
        cos_sim = (z * z_tilde).sum(dim=-1)  # (B, N)
    else:
        cos_sim = (z * z_tilde).sum(dim=-1)  # (B,)

    loss = -cos_sim
    if reduce == "mean":
        return loss.mean()
    return loss.sum()


def compute_projection_loss(
    zs: list,
    zs_tilde: list,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Compute projection loss over multiple encoder/encoder_types (e.g. multi-scale).
    Each zs[i] is (B, N, C_i), each zs_tilde[i] is (B, N, C_i).

    Args:
        zs: List of encoder features, each (B, N, C)
        zs_tilde: List of projected model features, each (B, N, C)
        reduce: 'mean' or 'sum'

    Returns:
        Scalar loss
    """
    total = 0.0
    count = 0
    for z, z_tilde in zip(zs, zs_tilde):
        # Handle spatial alignment: if z is (B, N1, C) and z_tilde is (B, N2, C), interpolate
        if z.shape[1] != z_tilde.shape[1]:
            # Reshape z to (B, C, H, W) for interpolation
            # Assume N = H*W; we need consistent grid
            # Simple fallback: global average pool both and compare (B, C)
            z_pool = z.mean(dim=1)
            z_tilde_pool = z_tilde.mean(dim=1)
            total += projection_loss(z_pool, z_tilde_pool, reduce="mean")
        else:
            total += projection_loss(z, z_tilde, reduce="mean")
        count += 1
    return total / max(count, 1)
