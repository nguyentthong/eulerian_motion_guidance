"""Differentiable bilinear warping operator ``W(·, ·)``.

The paper repeatedly invokes a warping operator ``W(x, f)`` that takes a
tensor ``x`` of shape ``(B, C, H, W)`` and a flow field ``f`` of shape
``(B, 2, H, W)`` (in pixel units) and returns the bilinearly resampled
tensor.  This module implements that operator together with two helpers
used by Eq. 8 and Eq. 10:

* :func:`flow_to_grid` converts an absolute flow in pixel coordinates to
  the normalised ``[-1, 1]`` grid expected by ``F.grid_sample``.
* :func:`sample_flow_at_flow` is the operator ``F(f_bwd, f_fwd)`` from
  Eq. 8 — it samples the backward flow at the locations indicated by the
  forward flow.  See ``DESIGN_NOTES.md`` §4 for the interpretation.

All functions are fully differentiable and shape-asserted at module
boundaries to fail loud rather than silent on dimension mismatches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "backward_warp",
    "flow_to_grid",
    "sample_flow_at_flow",
]


def flow_to_grid(flow: Tensor) -> Tensor:
    """Convert a pixel-space flow field to a normalised sampling grid.

    Given an absolute displacement field in pixel coordinates, return the
    grid expected by :func:`torch.nn.functional.grid_sample`, where every
    coordinate is mapped to ``[-1, 1]``.  Concretely, for output pixel
    ``(y, x)`` we want to read input pixel ``(y + dy, x + dx)``; that
    target pixel in normalised coordinates is

    .. math::

        \\tilde u = 2 (x + dx) / (W - 1) - 1, \\\\
        \\tilde v = 2 (y + dy) / (H - 1) - 1.

    Args:
        flow: ``(B, 2, H, W)`` tensor.  ``flow[:, 0]`` is the *x*
            (horizontal) displacement, ``flow[:, 1]`` is the *y*
            (vertical) displacement, both in pixels.

    Returns:
        ``(B, H, W, 2)`` sampling grid in normalised coordinates suitable
        for ``align_corners=True`` ``grid_sample``.
    """
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError(f"Expected flow of shape (B, 2, H, W); got {tuple(flow.shape)}")
    b, _, h, w = flow.shape
    device, dtype = flow.device, flow.dtype

    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    base = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
    base = base.unsqueeze(0).expand(b, -1, -1, -1)  # (B, H, W, 2)

    flow_perm = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)  channels last
    target = base + flow_perm

    # Normalise to [-1, 1].
    scale = torch.tensor([w - 1, h - 1], device=device, dtype=dtype).clamp_min(1.0)
    grid = 2.0 * target / scale - 1.0
    return grid


def backward_warp(
    feat: Tensor,
    flow: Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tensor:
    """Backward-warp ``feat`` by the (forward) ``flow``.

    Mathematically this is the operator ``W`` used in Eq. 1 and Eq. 10:
    ``W(feat, f)(x) = feat(x + f(x))``.  Because ``grid_sample`` does
    inverse sampling, supplying the forward flow here gives exactly the
    forward warp expected by the paper (i.e., transporting ``feat`` from
    its grid to a new grid using the displacement ``f``).

    Args:
        feat: Source tensor of shape ``(B, C, H, W)``.
        flow: Forward flow ``(B, 2, H, W)`` in pixels (``x`` then ``y``).
        mode: Sampling mode for ``grid_sample`` (``"bilinear"`` or
            ``"nearest"``).
        padding_mode: Out-of-bounds policy (``"zeros"``, ``"border"``,
            or ``"reflection"``).

    Returns:
        Warped feature tensor with the same shape as ``feat``.
    """
    if feat.dim() != 4:
        raise ValueError(f"feat must be (B, C, H, W); got {tuple(feat.shape)}")
    if flow.shape[0] != feat.shape[0] or flow.shape[2:] != feat.shape[2:]:
        raise ValueError(
            f"feat ({tuple(feat.shape)}) and flow ({tuple(flow.shape)}) must share "
            "batch and spatial dimensions"
        )
    grid = flow_to_grid(flow)
    return F.grid_sample(
        feat,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


def sample_flow_at_flow(flow_bwd: Tensor, flow_fwd: Tensor) -> Tensor:
    """Implement the ``F(f_bwd, f_fwd)`` operator from Eq. 8.

    "Backward flow vector sampled at the target location mapped by the
    forward flow" — i.e., we look up ``flow_bwd`` at the position
    ``x + flow_fwd(x)`` for every pixel ``x``.  This is the pillar of the
    forward–backward cycle check.

    Args:
        flow_bwd: Backward flow ``(B, 2, H, W)`` (i.e. ``f_{t+1→t}``).
        flow_fwd: Forward flow ``(B, 2, H, W)`` (i.e. ``f_{t→t+1}``).

    Returns:
        Resampled backward flow ``(B, 2, H, W)``.

    Notes:
        Author's interpretation.  See ``DESIGN_NOTES.md`` §4.  The
        sampling is bilinear with zero padding, matching standard
        forward–backward cycle implementations in the optical-flow
        literature.
    """
    if flow_bwd.shape != flow_fwd.shape:
        raise ValueError(
            f"flow_bwd {tuple(flow_bwd.shape)} and flow_fwd {tuple(flow_fwd.shape)} "
            "must have identical shape"
        )
    return backward_warp(flow_bwd, flow_fwd, mode="bilinear", padding_mode="zeros")
