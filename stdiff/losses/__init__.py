"""Modular loss functions for STDiff training."""

from .projection_loss import projection_loss, compute_projection_loss
from .normal_loss import (
    compute_normal_loss,
    get_projection_layer as get_normal_projection_layer,
    get_projection_config,
    range_to_normals,
)

__all__ = [
    "projection_loss",
    "compute_projection_loss",
    "compute_normal_loss",
    "get_normal_projection_layer",
    "get_projection_config",
    "range_to_normals",
]
