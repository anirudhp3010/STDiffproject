"""
Modular REPA projector: maps diffusion model bottleneck features to encoder (ScaLR) feature space.
Same structure as U-REPA / R3DPA SiT projectors.
"""
import torch
import torch.nn as nn


def build_mlp(
    in_dim: int,
    projector_dim: int,
    z_dim: int,
) -> nn.Module:
    """Build projector MLP: in_dim -> projector_dim -> projector_dim -> z_dim."""
    return nn.Sequential(
        nn.Linear(in_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class REPAProjector(nn.Module):
    """
    Projects UNet bottleneck features (B, C, H, W) to encoder feature dim (B, N, z_dim).
    Supports spatial reshaping and optional interpolation to match encoder grid.
    """

    def __init__(
        self,
        hidden_channels: int,
        z_dim: int,
        projector_dim: int = 2048,
        target_grid: tuple = None,
        scale_mode: str = "bicubic",
    ):
        """
        Args:
            hidden_channels: UNet bottleneck channel dim (e.g. block_out_channels[-1])
            z_dim: Target encoder feature dim (e.g. 768 for ScaLR)
            projector_dim: MLP hidden dim
            target_grid: (H, W) to interpolate to for spatial alignment. If None, no interpolation.
            scale_mode: 'bicubic' or 'bilinear' for interpolation
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.z_dim = z_dim
        self.target_grid = target_grid
        self.scale_mode = scale_mode

        self.mlp = build_mlp(hidden_channels, projector_dim, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) bottleneck features
        Returns:
            (B, N, z_dim) projected features, N = H*W (or target_grid[0]*target_grid[1] if interpolating)

        Order matches R3DPA: project first (MLP), then interpolate if needed.
        """
        B, C, H, W = x.shape
        # 1. Reshape to (B, H*W, C) and project (MLP first, like R3DPA)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        z = self.mlp(x_flat)  # (B, H*W, z_dim)

        # 2. Optional: interpolate projected features to match encoder grid (like R3DPA UNet/SiT)
        if self.target_grid is not None and (H, W) != self.target_grid:
            H_t, W_t = self.target_grid
            # Reshape to (B, z_dim, H, W) for interpolation
            z_spatial = z.permute(0, 2, 1).reshape(B, self.z_dim, H, W)
            # BFloat16 not supported by bicubic/bilinear on some backends; use float32 for interpolate
            dtype_in = z_spatial.dtype
            z_spatial = torch.nn.functional.interpolate(
                z_spatial.float(),
                size=self.target_grid,
                mode=self.scale_mode,
                align_corners=False,
            ).to(dtype_in)
            z = z_spatial.permute(0, 2, 3, 1).reshape(B, H_t * W_t, self.z_dim)

        return z
