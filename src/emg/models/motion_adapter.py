"""Motion Adapter (Section 3) — per-scale refinement of warped features.

The paper describes a "learnable motion adapter ``M``" that turns sparse
hints into dense flow *and* warps multi-scale reference features.  We
factor those two concerns:

* The S2D network (:mod:`emg.models.s2d`) produces the dense flow.
* The :class:`MotionAdapter` here is a thin per-scale projection that
  refines the warped reference feature before it is consumed by the
  :class:`emg.models.flow_controlnet.FlowControlNet`.

This split makes each component testable in isolation.  See
``DESIGN_NOTES.md`` §3.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from emg.motion.eulerian import rescale_flow
from emg.motion.warping import backward_warp

__all__ = ["MotionAdapter"]


class _ScaleHead(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()
        g = min(num_groups, channels)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(g, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        # Zero-init final conv so adapter starts as identity.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj(x)


class MotionAdapter(nn.Module):
    """Warp reference features by an Eulerian flow and refine.

    Author's interpretation — see ``DESIGN_NOTES.md`` §3.

    Args:
        scales: List of channel counts at each spatial scale (top-down).
            Each scale is processed by an independent
            :class:`_ScaleHead`.

    Shape conventions:
        ``forward(features, flow)`` consumes a list of feature tensors
        ``[(B, C_k, H_k, W_k)]`` and a single flow tensor at the
        full-resolution grid; the flow is rescaled to each feature's
        spatial size before warping.
    """

    def __init__(self, scales: list[int]) -> None:
        super().__init__()
        if not scales:
            raise ValueError("scales must be non-empty")
        self.heads = nn.ModuleList([_ScaleHead(c) for c in scales])
        self.scales = list(scales)

    def forward(self, features: list[Tensor], flow: Tensor) -> list[Tensor]:
        """Warp & refine each feature in the multi-scale list.

        Args:
            features: List of ``(B, C_k, H_k, W_k)`` tensors.
            flow: ``(B, 2, H, W)`` Eulerian flow at any resolution; will
                be bilinearly resampled to each feature's resolution
                with magnitude scaling.

        Returns:
            Refined feature list of the same shape as ``features``.
        """
        if len(features) != len(self.heads):
            raise ValueError(
                f"Got {len(features)} features but adapter has {len(self.heads)} heads"
            )

        out: list[Tensor] = []
        for feat, head in zip(features, self.heads, strict=True):
            f = rescale_flow(flow, (feat.shape[-2], feat.shape[-1]))
            warped = backward_warp(feat, f)
            out.append(head(warped))
        return out
