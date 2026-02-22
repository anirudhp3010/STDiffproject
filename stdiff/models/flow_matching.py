"""
Flow-matching / v-prediction components for STDiff.

Provides interpolant functions (linear, cosine) and utilities for training with
flow matching instead of DDPM. The model predicts velocity v = d_alpha_t * x_0 + d_sigma_t * noise.
"""
import math
from typing import Tuple

import torch


def get_interpolant(
    t: torch.Tensor,
    path_type: str = "linear",
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute flow-matching interpolant coefficients.

    Args:
        t: Time values in [0, 1]. Shape (batch_size,) or broadcastable.
        path_type: 'linear' or 'cosine'.
        device: Device for output tensors (inherits from t if None).

    Returns:
        alpha_t, sigma_t, d_alpha_t, d_sigma_t
        - x_t = alpha_t * x_0 + sigma_t * noise
        - v_target = d_alpha_t * x_0 + d_sigma_t * noise (velocity for ODE)
    """
    if device is None and torch.is_tensor(t):
        device = t.device
    elif device is None:
        device = torch.device("cpu")

    if path_type == "linear":
        # Linear path: x_t = (1-t)*x_0 + t*noise
        # alpha_t = 1 - t, sigma_t = t
        # d/dt: d_alpha_t = -1, d_sigma_t = 1
        alpha_t = 1.0 - t
        sigma_t = t
        d_alpha_t = torch.full_like(t, -1.0) if torch.is_tensor(t) else -1.0
        d_sigma_t = torch.full_like(t, 1.0) if torch.is_tensor(t) else 1.0

    elif path_type == "cosine":
        # Cosine path: alpha_t = cos(t*pi/2), sigma_t = sin(t*pi/2)
        # d_alpha_t = -pi/2 * sin(t*pi/2), d_sigma_t = pi/2 * cos(t*pi/2)
        half_pi = math.pi / 2
        angle = t * half_pi
        alpha_t = torch.cos(angle) if torch.is_tensor(t) else math.cos(float(t) * half_pi)
        sigma_t = torch.sin(angle) if torch.is_tensor(t) else math.sin(float(t) * half_pi)
        d_alpha_t = -half_pi * sigma_t
        d_sigma_t = half_pi * alpha_t

    else:
        raise ValueError(f"Unknown path_type: {path_type}. Use 'linear' or 'cosine'.")

    return alpha_t, sigma_t, d_alpha_t, d_sigma_t


def add_flow_noise(
    x_0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    path_type: str = "linear",
) -> torch.Tensor:
    """
    Add flow-matching noise: x_t = alpha_t * x_0 + sigma_t * noise.

    Args:
        x_0: Clean data, shape (N, C, H, W) or similar.
        noise: Gaussian noise, same shape as x_0.
        t: Time in [0, 1], shape (N,) or broadcastable.
        path_type: 'linear' or 'cosine'.

    Returns:
        Noisy input x_t.
    """
    alpha_t, sigma_t, _, _ = get_interpolant(t, path_type, x_0.device)
    # Expand for broadcasting: (N,) -> (N, 1, 1, 1) for (N, C, H, W)
    shape = (x_0.shape[0],) + (1,) * (x_0.dim() - 1)
    alpha_t = alpha_t.reshape(shape).to(x_0.dtype).to(x_0.device)
    sigma_t = sigma_t.reshape(shape).to(x_0.dtype).to(x_0.device)
    return alpha_t * x_0 + sigma_t * noise


def get_velocity_target(
    x_0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    path_type: str = "linear",
) -> torch.Tensor:
    """
    Compute v-prediction target: v = d_alpha_t * x_0 + d_sigma_t * noise.

    Args:
        x_0: Clean data.
        noise: Gaussian noise.
        t: Time in [0, 1].
        path_type: 'linear' or 'cosine'.

    Returns:
        Velocity target v (same shape as x_0).
    """
    _, _, d_alpha_t, d_sigma_t = get_interpolant(t, path_type, x_0.device)
    shape = (x_0.shape[0],) + (1,) * (x_0.dim() - 1)
    d_alpha_t = d_alpha_t.reshape(shape).to(x_0.dtype).to(x_0.device)
    d_sigma_t = d_sigma_t.reshape(shape).to(x_0.dtype).to(x_0.device)
    return d_alpha_t * x_0 + d_sigma_t * noise


def recover_x0_from_velocity(
    x_t: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    path_type: str = "linear",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Single-step recovery of x_0 from (x_t, t, v) using the flow-matching ODE.

    Linear path: x_0 = x_t - t * v
    Cosine path: x_0 = (x_t - sigma_t * v / d_sigma_t) / (alpha_t - sigma_t * d_alpha_t / d_sigma_t)

    Args:
        x_t: Noisy state at time t, shape (N, C, H, W).
        t: Time in [0, 1], shape (N,) or scalar.
        v: Predicted velocity, shape (N, C, H, W).
        path_type: 'linear' or 'cosine'.
        eps: Small value for numerical stability (cosine path near t=0).

    Returns:
        Recovered x_0 (same shape as x_t).
    """
    if path_type == "linear":
        shape = (x_t.shape[0],) + (1,) * (x_t.dim() - 1)
        t_expand = t.reshape(shape).to(x_t.dtype).to(x_t.device) if torch.is_tensor(t) else t
        return x_t - t_expand * v

    elif path_type == "cosine":
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = get_interpolant(t, path_type, x_t.device)
        shape = (x_t.shape[0],) + (1,) * (x_t.dim() - 1)
        alpha_t = alpha_t.reshape(shape).to(x_t.dtype).to(x_t.device)
        sigma_t = sigma_t.reshape(shape).to(x_t.dtype).to(x_t.device)
        d_alpha_t = d_alpha_t.reshape(shape).to(x_t.dtype).to(x_t.device)
        d_sigma_t = d_sigma_t.reshape(shape).to(x_t.dtype).to(x_t.device)

        denom = alpha_t - sigma_t * d_alpha_t / (d_sigma_t + eps)
        numer = x_t - sigma_t * v / (d_sigma_t + eps)
        return numer / denom.clamp(min=eps)

    else:
        raise ValueError(f"path_type must be 'linear' or 'cosine', got {path_type}")


class FlowMatchingNoiseAdder:
    """
    Replaces DDPMScheduler for flow-matching training.
    Provides add_noise() and get_velocity_target() with continuous t in [0, 1].
    """

    def __init__(self, path_type: str = "linear"):
        self.path_type = path_type

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """x_t = alpha_t * x_0 + sigma_t * noise."""
        return add_flow_noise(x_0, noise, t, self.path_type)

    def get_velocity_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """v = d_alpha_t * x_0 + d_sigma_t * noise."""
        return get_velocity_target(x_0, noise, t, self.path_type)
