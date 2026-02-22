"""
Modular normal map loss for range images.

Computes L1 loss between ground-truth and predicted surface normals.
Normals are derived from range images via differentiable spherical projection.
Supports masking by valid pixels only.
"""
import importlib.util
import torch
import torch.nn.functional as F
from pathlib import Path

# Default projection config (KITTI range: 64x512)
DEFAULT_PROJ_CONFIG = {
    "DATA_CONFIG": {
        "FOV_UP": 3.0,
        "FOV_DOWN": -25.0,
        "WIDTH": 512,
        "HEIGHT": 64,
    }
}


def get_projection_config(height=64, width=512, fov_up=3.0, fov_down=-25.0):
    """Build projection config for given dimensions."""
    return {
        "DATA_CONFIG": {
            "FOV_UP": fov_up,
            "FOV_DOWN": fov_down,
            "WIDTH": width,
            "HEIGHT": height,
        }
    }


def get_projection_layer(cfg=None):
    """Get projection layer for range-to-3D conversion."""
    # Load projection from project root utils.py (not stdiff.utils)
    _root_utils = Path(__file__).resolve().parent.parent.parent / "utils.py"
    _spec = importlib.util.spec_from_file_location("_utils_projection", _root_utils)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.projection(cfg or DEFAULT_PROJ_CONFIG)


def range_to_vertex_grid(proj_layer, range_img):
    """
    Convert range image to 3D vertex grid (differentiable).

    Args:
        proj_layer: projection instance with x_fac, y_fac, z_fac
        range_img: (B, 1, H, W) or (B, H, W), values in [0, 1] (use (x+1)/2 for [-1,1] input)

    Returns:
        vertex: (B, H, W, 3) 3D coordinates
    """
    if range_img.dim() == 4:
        range_img = range_img.squeeze(1)  # (B, H, W)
    B, H, W = range_img.shape
    device = range_img.device
    x_fac = proj_layer.x_fac.to(device).to(range_img.dtype)
    y_fac = proj_layer.y_fac.to(device).to(range_img.dtype)
    z_fac = proj_layer.z_fac.to(device).to(range_img.dtype)

    # Clamp to small positive to avoid zeros (invalid will be masked in loss)
    range_safe = range_img.clamp(min=1e-6)

    vertex = torch.zeros(B, H, W, 3, device=device, dtype=range_img.dtype)
    vertex[..., 0] = range_safe * x_fac
    vertex[..., 1] = range_safe * y_fac
    vertex[..., 2] = range_safe * z_fac

    return vertex


def vertex_to_normals(vertex, height, width):
    """
    Compute surface normals from vertex grid (differentiable).
    Normal at (y,x) = cross((v-p)/|v-p|, (u-p)/|u-p|) where u=vertex[y,x+1], v=vertex[y+1,x].

    Args:
        vertex: (B, H, W, 3) 3D positions
        height, width: int

    Returns:
        normals: (B, H, W, 3), unit vectors or zero for invalid
    """
    B, H, W, _ = vertex.shape
    device = vertex.device

    # Wrap x+1 for spherical projection
    x_next = (torch.arange(W, device=device) + 1) % W
    y_next = torch.clamp(torch.arange(H, device=device) + 1, max=H - 1)

    p = vertex  # (B, H, W, 3)
    u = vertex[:, :, x_next]  # (B, H, W, 3) - neighbor to the right
    v = vertex[:, y_next]  # (B, H, W, 3) - neighbor below

    diff_u = u - p
    diff_v = v - p

    # Cross product: w = cross(diff_v, diff_u) -> normal pointing "up"
    w = torch.cross(diff_v, diff_u, dim=-1)

    norm = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normals = w / norm

    # Invalid: where cross product is degenerate (e.g. collinear points)
    valid_n = (norm.squeeze(-1) > 1e-6).float().unsqueeze(-1)
    normals = normals * valid_n

    # Exclude last row (no v neighbor)
    row_mask = torch.ones(B, H, W, 1, device=device)
    row_mask[:, -1, :, :] = 0
    normals = normals * row_mask

    return normals


def range_to_normals(proj_layer, range_img, height=64, width=512, global_min=None, global_max=None):
    """
    Convert range image to normal map (differentiable).

    Args:
        proj_layer: projection instance
        range_img: (B, 1, H, W) or (B, H, W). If normalized [0, 1], pass global_min/global_max
                   to denormalize to meters before projection.
        height, width: image dimensions
        global_min, global_max: if provided, denormalize range: meters = range_img * (gmax - gmin) + gmin

    Returns:
        normals: (B, 3, H, W) for loss computation
    """
    if global_min is not None and global_max is not None and global_max > global_min:
        # Denormalize [0, 1] -> [global_min, global_max] meters for correct 3D geometry
        range_img = range_img * (global_max - global_min) + global_min
    vertex = range_to_vertex_grid(proj_layer, range_img)
    normals = vertex_to_normals(vertex, height, width)
    return normals.permute(0, 3, 1, 2)  # (B, 3, H, W)


def compute_normal_loss(
    pred_normals,
    gt_normals,
    valid_mask,
    loss_type="l1",
):
    """
    Compute normal loss only on valid pixels.

    Args:
        pred_normals: (B, 3, H, W) predicted normals (unit vectors)
        gt_normals: (B, 3, H, W) ground truth normals (unit vectors)
        valid_mask: (B, H, W) or (B, 1, H, W), 1=valid 0=invalid
        loss_type: "l1" or "cosine" (1 - cos_sim)

    Returns:
        scalar loss (mean over valid pixels)
    """
    if valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)  # (B, 1, H, W)

    # Erode mask: pixel valid only if self and neighbors (x+1, y+1) are valid
    # for normal computation
    B, _, H, W = valid_mask.shape
    device = valid_mask.device
    x_next = (torch.arange(W, device=device) + 1) % W
    y_next = torch.clamp(torch.arange(H, device=device) + 1, max=H - 1)

    m_self = valid_mask.squeeze(1)  # (B, H, W)
    m_u = m_self[:, :, x_next]
    m_v = m_self[:, y_next]
    eroded = (m_self * m_u * m_v).unsqueeze(1)  # (B, 1, H, W)

    # Exclude last row
    eroded[:, :, -1, :] = 0

    if loss_type == "l1":
        diff = (pred_normals - gt_normals).abs().sum(dim=1, keepdim=True)
        masked_diff = diff * eroded
        num_valid = eroded.sum().clamp(min=1)
        loss = masked_diff.sum() / num_valid
    elif loss_type == "cosine":
        cos_sim = (pred_normals * gt_normals).sum(dim=1, keepdim=True).clamp(-1, 1)
        loss_per_pixel = 1.0 - cos_sim  # [0, 2], 0 when aligned
        masked_loss = loss_per_pixel * eroded
        num_valid = eroded.sum().clamp(min=1)
        loss = masked_loss.sum() / num_valid
    else:
        raise ValueError(f"loss_type must be 'l1' or 'cosine', got {loss_type}")

    return loss
