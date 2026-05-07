"""Video transforms used by the dataloaders.

The transforms operate on a ``(T, H, W, C)`` ``uint8`` array (as returned
by :mod:`av` / :mod:`imageio`) and produce a ``(T, 3, H, W)`` float
tensor in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["VideoTransform", "build_default_transform"]


@dataclass(slots=True)
class VideoTransform:
    """Resize + center-crop + normalise a ``(T, H, W, C)`` uint8 array.

    Attributes:
        size: Output square spatial size (e.g., 256).
        normalize: If True, return ``[0, 1]``; otherwise return
            ``[-1, 1]`` (the SVD VAE expects ``[-1, 1]``).
    """

    size: int = 256
    normalize: bool = True

    def __call__(self, frames: np.ndarray) -> Tensor:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames must be (T, H, W, 3); got {frames.shape}")
        # to (T, 3, H, W) float
        t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float() / 255.0

        # Resize so the short side is `size`.
        _, _, h, w = t.shape
        scale = self.size / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=True)

        # Center crop to (size, size)
        top = (new_h - self.size) // 2
        left = (new_w - self.size) // 2
        t = t[:, :, top : top + self.size, left : left + self.size]

        if not self.normalize:
            t = t * 2.0 - 1.0
        return t


def build_default_transform(*, size: int = 256, normalize: bool = True) -> VideoTransform:
    """Factory for the standard 256-square ``[0, 1]`` transform."""
    return VideoTransform(size=size, normalize=normalize)
