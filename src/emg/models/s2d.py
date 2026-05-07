"""Sparse-to-Dense (S2D) flow regression network.

The paper introduces an S2D module in Section 4.4 but does not specify
its internal architecture.  We implement a small U-Net (5 down/up
stages, GroupNorm + SiLU) that maps a 3-channel sparse hint tensor to
a 2-channel dense flow field.  See ``DESIGN_NOTES.md`` §1 for details.

Author's interpretation flagged in :class:`SparseToDenseNet`'s docstring.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["SparseToDenseNet"]


def _conv_block(in_ch: int, out_ch: int, *, num_groups: int = 8) -> nn.Sequential:
    g = min(num_groups, out_ch)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(g, out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(g, out_ch),
        nn.SiLU(inplace=True),
    )


class SparseToDenseNet(nn.Module):
    """Sparse-to-Dense U-Net.

    Author's interpretation — see ``DESIGN_NOTES.md`` §1.

    Args:
        in_channels: Number of input channels.  Default 3 = ``(u, v, m)``
            where ``(u, v)`` is the rasterised sparse flow at a few
            pixels and ``m`` is a binary visibility mask.
        out_channels: 2 (the dense flow ``(u, v)``).
        base_channels: Channels at the highest spatial resolution.
        depth: Number of down/up stages.

    Shape:
        Input  ``(B, in_channels, H, W)``
        Output ``(B, 2, H, W)`` flow in pixel units of the input grid.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 32,
        depth: int = 5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        chans: list[int] = [base_channels * (2 ** min(i, 3)) for i in range(depth)]
        # depth=5, base=32 -> [32, 64, 128, 256, 256]

        self.down_blocks = nn.ModuleList()
        prev = in_channels
        for c in chans:
            self.down_blocks.append(_conv_block(prev, c))
            prev = c
        self.pool = nn.AvgPool2d(2, 2)

        self.bottleneck = _conv_block(chans[-1], chans[-1] * 2)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        prev = chans[-1] * 2
        for c in reversed(chans):
            self.upsamples.append(nn.Conv2d(prev, c, kernel_size=1))
            self.up_blocks.append(_conv_block(c * 2, c))
            prev = c

        self.head = nn.Conv2d(chans[0], out_channels, kernel_size=3, padding=1)
        # Initialise the head so that the network outputs near-zero flow
        # at init — useful so training starts from "no motion" rather
        # than random motion.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, sparse: Tensor) -> Tensor:
        """Predict a dense flow from a sparse hint tensor.

        Args:
            sparse: ``(B, in_channels, H, W)`` sparse hints.

        Returns:
            ``(B, 2, H, W)`` dense flow.
        """
        if sparse.dim() != 4 or sparse.shape[1] != self.in_channels:
            raise ValueError(
                f"sparse must be (B, {self.in_channels}, H, W); got {tuple(sparse.shape)}"
            )

        feats: list[Tensor] = []
        x = sparse
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            feats.append(x)
            if i < self.depth - 1:
                x = self.pool(x)
        x = self.bottleneck(x)

        for i, (up, block) in enumerate(zip(self.upsamples, self.up_blocks, strict=True)):
            skip = feats[self.depth - 1 - i]
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return self.head(x)
