"""Visualisation helpers for flow fields and video rendering.

We use a small subset of the standard Sintel-style HSV colour wheel for
flow visualisation; this avoids a dependency on OpenCV.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

__all__ = ["flow_to_rgb", "save_video"]


def _make_color_wheel() -> np.ndarray:
    """Build the standard Middlebury colour wheel (55 entries)."""
    # Counts per segment.
    rs, ys, gs, cs, bs, ms = 15, 6, 4, 11, 13, 6
    n = rs + ys + gs + cs + bs + ms
    wheel = np.zeros((n, 3), dtype=np.float32)
    col = 0
    # R -> Y
    wheel[col : col + rs, 0] = 255
    wheel[col : col + rs, 1] = np.floor(255 * np.arange(rs) / rs)
    col += rs
    # Y -> G
    wheel[col : col + ys, 0] = 255 - np.floor(255 * np.arange(ys) / ys)
    wheel[col : col + ys, 1] = 255
    col += ys
    # G -> C
    wheel[col : col + gs, 1] = 255
    wheel[col : col + gs, 2] = np.floor(255 * np.arange(gs) / gs)
    col += gs
    # C -> B
    wheel[col : col + cs, 1] = 255 - np.floor(255 * np.arange(cs) / cs)
    wheel[col : col + cs, 2] = 255
    col += cs
    # B -> M
    wheel[col : col + bs, 2] = 255
    wheel[col : col + bs, 0] = np.floor(255 * np.arange(bs) / bs)
    col += bs
    # M -> R
    wheel[col : col + ms, 2] = 255 - np.floor(255 * np.arange(ms) / ms)
    wheel[col : col + ms, 0] = 255
    return wheel


_WHEEL = _make_color_wheel()


def flow_to_rgb(flow: Tensor, *, max_norm: float | None = None) -> Tensor:
    """Encode a flow field as RGB using the Middlebury colour wheel.

    Args:
        flow: ``(B, 2, H, W)`` or ``(2, H, W)`` float tensor.
        max_norm: Optional explicit normalisation; if ``None`` the
            per-frame maximum magnitude is used.

    Returns:
        ``uint8`` tensor with the same leading shape, channel last
        ``(B, H, W, 3)`` or ``(H, W, 3)``.
    """
    squeeze = flow.dim() == 3
    if squeeze:
        flow = flow.unsqueeze(0)
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must be (B, 2, H, W); got {tuple(flow.shape)}")
    fx = flow[:, 0].cpu().float().numpy()
    fy = flow[:, 1].cpu().float().numpy()

    rad = np.sqrt(fx**2 + fy**2)
    ang = np.arctan2(-fy, -fx) / math.pi  # [-1, 1]
    fk = (ang + 1) / 2 * (_WHEEL.shape[0] - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % _WHEEL.shape[0]
    f = fk - k0

    out = np.zeros((*fx.shape, 3), dtype=np.float32)
    for i in range(3):
        col0 = _WHEEL[k0, i] / 255.0
        col1 = _WHEEL[k1, i] / 255.0
        col = (1 - f) * col0 + f * col1
        if max_norm is not None:
            scale = max_norm
        else:
            scale = float(np.maximum(rad.max(), 1e-6))
        out[..., i] = (1.0 - rad / scale * (1.0 - col)) * 255.0

    out = np.clip(out, 0, 255).astype(np.uint8)
    rgb = torch.from_numpy(out)
    return rgb[0] if squeeze else rgb


def save_video(
    frames: Tensor,
    path: str | Path,
    *,
    fps: int = 8,
) -> None:
    """Persist a video tensor to disk as an MP4.

    Args:
        frames: ``(T, 3, H, W)`` or ``(T, H, W, 3)`` tensor in ``[0, 1]``
            or ``[0, 255]``.
        path: Output filesystem path.
        fps: Frame rate of the resulting video.
    """
    import imageio.v3 as iio  # local import keeps imageio optional

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if frames.dim() != 4:
        raise ValueError(f"frames must be 4-D; got {tuple(frames.shape)}")
    if frames.shape[1] == 3:
        frames = frames.permute(0, 2, 3, 1)
    arr = frames.detach().cpu().float().numpy()
    if arr.max() <= 1.0 + 1e-3:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    iio.imwrite(out_path, arr, fps=fps, codec="libx264", quality=7)
