"""
Modular ScaLR feature loader for STDiff.
Loads precomputed ScaLR features from disk, compatible with R3DPA extraction format.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod


class BaseFeatureLoader(ABC):
    """Abstract interface for loading encoder features."""

    @abstractmethod
    def load(self, sequence: str, frame: str) -> torch.Tensor:
        """Load features for a single frame. Returns (N, C) or (H, W, C)."""
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimension of each feature vector."""
        pass

    @property
    @abstractmethod
    def grid_shape(self) -> Tuple[int, int]:
        """(H, W) of the feature grid."""
        pass


class ScaLRFeatureLoader(BaseFeatureLoader):
    """
    Loads ScaLR features from npz files.
    Expected structure: {data_root}/scalr_features/{sequence}/{frame}/feature_grid_{H}x{W}.npz
    Key: 'feats' -> (H*W, C) or (H, W, C)
    """

    def __init__(
        self,
        data_root: str,
        grid_size: Tuple[int, int] = (8, 64),
        subdir: Optional[str] = "scalr_features",
    ):
        """
        Args:
            data_root: Root path. If subdir is set, features live at data_root/subdir/{seq}/{frame}/.
                       If subdir is None/empty, data_root is the features root directly: data_root/{seq}/{frame}/
            grid_size: (H, W) of the feature grid
            subdir: Subdirectory under data_root for features. None/empty = data_root is features root.
        """
        self.data_root = Path(data_root)
        self.grid_size = grid_size
        self.features_dir = self.data_root / subdir if subdir else self.data_root
        self._feature_dim = None

    def _resolve_path(self, sequence: str, frame: str) -> Path:
        """Get path to feature npz for sequence/frame."""
        frame_name = str(frame).zfill(6) if isinstance(frame, (int, np.integer)) else frame
        seq_name = str(sequence).zfill(2) if isinstance(sequence, (int, np.integer)) else sequence
        return self.features_dir / seq_name / frame_name / f"feature_grid_{self.grid_size[0]}x{self.grid_size[1]}.npz"

    def load(self, sequence: str, frame: str) -> Optional[torch.Tensor]:
        """
        Load features for a single frame.
        Returns:
            (N, C) tensor, N = H*W, or None if file not found
        """
        path = self._resolve_path(sequence, frame).resolve()
        if not path.exists():
            return None
        data = np.load(path)
        feats = data["feats"]
        if feats.ndim == 3:
            feats = feats.reshape(-1, feats.shape[-1])
        self._feature_dim = feats.shape[-1]
        return torch.from_numpy(feats.astype(np.float32))

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            # Try to infer from first available file
            for p in self.features_dir.rglob("feature_grid_*.npz"):
                d = np.load(p)
                feats = d["feats"]
                self._feature_dim = feats.shape[-1]
                return self._feature_dim
        return self._feature_dim or 768  # ScaLR default fallback

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return self.grid_size


def load_scalr_features_for_clip(
    loader: ScaLRFeatureLoader,
    clip_files: List[Path],
) -> Optional[torch.Tensor]:
    """
    Load ScaLR features for a clip of frames.
    Assumes clip_files are paths like .../sequences/00/processed/range/000000.npy
    or .../00/processed/range/000000.npy - we extract sequence and frame from path.

    Returns:
        (T, N, C) tensor for T frames, or None if any frame missing
    """
    import logging
    feats_list = []
    for npy_path in clip_files:
        # Use absolute path so workers (different cwd) resolve correctly
        npy_path = Path(npy_path).resolve()
        # Path: {root}/{seq}/processed/range/{frame}.npy
        parts = npy_path.as_posix().replace("\\", "/").split("/")
        if "processed" in parts:
            idx = parts.index("processed")
            seq = parts[idx - 1]
            frame = parts[-1].replace(".npy", "")
        else:
            # Fallback: last two path components
            seq = parts[-3] if len(parts) >= 3 else "00"
            frame = parts[-1].replace(".npy", "")

        f = loader.load(seq, frame)
        if f is None:
            # One-time debug: log first failure so user can see resolved path
            try:
                load_scalr_features_for_clip._log_fail_once
            except AttributeError:
                load_scalr_features_for_clip._log_fail_once = True
                feat_path = loader._resolve_path(seq, frame)
                logging.getLogger(__name__).warning(
                    "ScaLR load failed once. npy_path=%s seq=%s frame=%s -> feat_path=%s exists=%s",
                    str(npy_path), seq, frame, str(feat_path), feat_path.exists(),
                )
            return None
        feats_list.append(f)

    return torch.stack(feats_list, dim=0)  # (T, N, C)
