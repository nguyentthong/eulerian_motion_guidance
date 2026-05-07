"""Eulerian motion-field representation (Definition 2 of the paper).

The Eulerian motion field models motion as the *adjacent-frame*
displacement ``f_{t→t+1}: Ω → R²``.  In code we represent a full clip as
a stack of ``T-1`` such flows together with their corresponding backward
flows ``f_{t+1→t}``.

Crucially, this representation is **not** anchored to the reference
frame ``I_0``.  Compare the Lagrangian formulation in Eq. 1 which uses
``u_{0→t}`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "EulerianFlowSequence",
    "adjacent_pair_indices",
    "rescale_flow",
]


def adjacent_pair_indices(num_frames: int) -> tuple[Tensor, Tensor]:
    """Return integer index tensors ``(src, dst)`` for adjacent pairs.

    Implements the index sets used by Eq. 11–12.  For a clip of length
    ``T`` we return two ``(T-1,)`` long tensors: the first lists frame
    indices ``0, 1, ..., T-2`` and the second lists ``1, 2, ..., T-1``.
    The forward set ``P_fwd`` is then ``zip(src, dst)``; the backward
    set ``P_bwd`` is its reverse.

    Args:
        num_frames: ``T``, must be ≥ 2.

    Returns:
        Pair of ``(T-1,)`` LongTensor index tensors ``(src, dst)``.
    """
    if num_frames < 2:
        raise ValueError(f"Need at least 2 frames to form adjacent pairs; got {num_frames}")
    src = torch.arange(num_frames - 1, dtype=torch.long)
    dst = torch.arange(1, num_frames, dtype=torch.long)
    return src, dst


def rescale_flow(flow: Tensor, target_size: tuple[int, int]) -> Tensor:
    """Resample a flow field to a new spatial resolution.

    Optical-flow magnitudes are expressed in *pixel* units; resizing the
    grid therefore requires scaling the flow vectors by the inverse
    aspect ratio.  This is necessary in the geometric loss because RAFT
    estimates flow at 256×256 pixel space whereas the SVD latent grid is
    32×32.

    Args:
        flow: ``(B, 2, H, W)`` source flow.
        target_size: ``(H', W')`` target spatial size.

    Returns:
        Resampled flow ``(B, 2, H', W')`` with magnitudes scaled to the
        new grid.
    """
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must be (B, 2, H, W); got {tuple(flow.shape)}")
    h_src, w_src = flow.shape[-2:]
    h_dst, w_dst = target_size
    if (h_src, w_src) == (h_dst, w_dst):
        return flow
    resampled = F.interpolate(flow, size=target_size, mode="bilinear", align_corners=True)
    sx = w_dst / max(w_src, 1)
    sy = h_dst / max(h_src, 1)
    scale = flow.new_tensor([sx, sy]).view(1, 2, 1, 1)
    return resampled * scale


@dataclass(slots=True)
class EulerianFlowSequence:
    """Container for the bidirectional Eulerian flow stack of a clip.

    Attributes:
        forward: ``(B, T-1, 2, H, W)`` forward flows ``f_{t→t+1}``.
        backward: ``(B, T-1, 2, H, W)`` backward flows ``f_{t+1→t}``.

    The shapes are asserted on construction; downstream code can rely
    on them.  Both tensors share dtype/device; mismatches raise.
    """

    forward: Tensor
    backward: Tensor

    def __post_init__(self) -> None:
        if self.forward.shape != self.backward.shape:
            raise ValueError(
                f"forward {tuple(self.forward.shape)} and backward "
                f"{tuple(self.backward.shape)} must have identical shape"
            )
        if self.forward.dim() != 5 or self.forward.shape[2] != 2:
            raise ValueError(
                f"flow must be (B, T-1, 2, H, W); got {tuple(self.forward.shape)}"
            )
        if self.forward.device != self.backward.device:
            raise ValueError("forward and backward must live on the same device")
        if self.forward.dtype != self.backward.dtype:
            raise ValueError("forward and backward must share dtype")

    @property
    def batch_size(self) -> int:
        """Batch size ``B``."""
        return int(self.forward.shape[0])

    @property
    def num_pairs(self) -> int:
        """Number of adjacent pairs ``T-1``."""
        return int(self.forward.shape[1])

    @property
    def spatial_size(self) -> tuple[int, int]:
        """``(H, W)`` of the flow fields."""
        return int(self.forward.shape[-2]), int(self.forward.shape[-1])

    def flat(self) -> tuple[Tensor, Tensor]:
        """Return forward and backward flows flattened to ``(B*(T-1), 2, H, W)``.

        Useful when feeding the cycle-consistency machinery, which
        operates on a 4-D batch.
        """
        b, n, c, h, w = self.forward.shape
        return (
            self.forward.reshape(b * n, c, h, w),
            self.backward.reshape(b * n, c, h, w),
        )

    def to(self, *args: object, **kwargs: object) -> EulerianFlowSequence:
        """Move both tensors to a new device/dtype simultaneously."""
        return EulerianFlowSequence(
            forward=self.forward.to(*args, **kwargs),  # type: ignore[arg-type]
            backward=self.backward.to(*args, **kwargs),  # type: ignore[arg-type]
        )
