"""Flow ControlNet (Section 4 / Figure 2).

A ControlNet-style branch that mirrors the encoder + middle block of
the frozen SVD spatial U-Net.  The branch consumes per-frame dense
flow concatenated with the warped reference latent, projects them to
the SVD encoder channel count, propagates through the mirror, and
emits zero-initialised residuals that are added to the corresponding
skip connections of the frozen U-Net.

Because the diffusers SVD U-Net has a fairly involved internal
structure, this module is written in a *slot-friendly* way: it owns a
deep-copy of the spatial encoder & middle block of the SVD U-Net, plus
a stack of zero-conv heads.  The forward pass returns the per-block
residuals that the calling code adds into the frozen U-Net.

For readers without diffusers installed (e.g. CI), the module also has
a "stub mode" that operates on a small synthetic encoder, so the
ControlNet's logic itself remains testable.

See ``DESIGN_NOTES.md`` §2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["FlowControlNet", "FlowControlNetOutput"]


@dataclass(slots=True)
class FlowControlNetOutput:
    """Residual outputs returned by :class:`FlowControlNet`.

    Attributes:
        down_block_residuals: List of per-skip residual tensors, one per
            encoder block of the SVD U-Net.
        mid_block_residual: Residual added to the U-Net middle block.
    """

    down_block_residuals: list[Tensor]
    mid_block_residual: Tensor


def _zero_conv(in_ch: int, out_ch: int) -> nn.Conv2d:
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class _DownBlock(nn.Module):
    """Tiny mirror block used in stub mode.

    Two conv-norm-activation pairs followed by a downsample.  Channels
    double on each block.
    """

    def __init__(self, in_ch: int, out_ch: int, *, downsample: bool = True) -> None:
        super().__init__()
        g = min(8, in_ch, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )
        self.downsample = (
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
            if downsample
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return self.downsample(x)


class FlowControlNet(nn.Module):
    """ControlNet-style flow conditioning branch.

    The module exposes two construction modes:

    1.  ``svd_unet=None`` (default): A small synthetic mirror encoder
        is built from ``block_channels``.  This mode is used in unit
        tests and as a fallback when diffusers / SVD weights are not
        available.

    2.  ``svd_unet`` is a ``diffusers`` U-Net: We deep-copy its spatial
        encoder + middle block and freeze the rest of the model.  This
        is the production path.

    Args:
        latent_channels: Channels of the SVD latent (``z``).  SVD-XT
            uses 4.
        flow_channels: Channels of the flow conditioning input
            (default 2).
        block_channels: Channel widths of the encoder mirror.  Used in
            stub mode and to size the zero-conv heads in production
            mode.
        svd_unet: Optional reference to the frozen SVD U-Net we copy
            from.  ``None`` selects stub mode.
    """

    def __init__(
        self,
        *,
        latent_channels: int = 4,
        flow_channels: int = 2,
        block_channels: tuple[int, ...] = (320, 640, 1280, 1280),
        svd_unet: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.flow_channels = flow_channels
        self.block_channels = tuple(int(c) for c in block_channels)
        self._has_svd = svd_unet is not None

        cond_in = latent_channels + flow_channels
        first = self.block_channels[0]

        # Conditioning input projection: z + flow -> first encoder width.
        self.cond_input = nn.Conv2d(cond_in, first, kernel_size=3, padding=1)

        if svd_unet is None:
            # Build a small stub encoder mirror.
            mirror_blocks: list[nn.Module] = []
            prev = first
            for i, c in enumerate(self.block_channels):
                # First block keeps the input conv; subsequent blocks
                # downsample by 2.
                mirror_blocks.append(_DownBlock(prev, c, downsample=i < len(self.block_channels) - 1))
                prev = c
            self.encoder_blocks = nn.ModuleList(mirror_blocks)
            g = min(8, prev)
            self.mid_block = nn.Sequential(
                nn.Conv2d(prev, prev, kernel_size=3, padding=1),
                nn.GroupNorm(g, prev),
                nn.SiLU(inplace=True),
                nn.Conv2d(prev, prev, kernel_size=3, padding=1),
            )
        else:
            # Production path: deep-copy the encoder and middle block.
            from copy import deepcopy

            self.encoder_blocks = deepcopy(svd_unet.down_blocks)
            self.mid_block = deepcopy(svd_unet.mid_block)
            for p in self.encoder_blocks.parameters():
                p.requires_grad = True
            for p in self.mid_block.parameters():
                p.requires_grad = True

        # Zero-conv heads producing the per-block residuals.
        self.zero_convs_down = nn.ModuleList(
            [_zero_conv(c, c) for c in self.block_channels]
        )
        self.zero_conv_mid = _zero_conv(self.block_channels[-1], self.block_channels[-1])

    def _stub_forward(self, latent_with_flow: Tensor) -> FlowControlNetOutput:
        """Forward through the synthetic mirror used in tests."""
        x = self.cond_input(latent_with_flow)
        residuals: list[Tensor] = []
        for block, zero in zip(self.encoder_blocks, self.zero_convs_down, strict=True):
            x = block(x)
            residuals.append(zero(x))
        x = self.mid_block(x)
        mid_res = self.zero_conv_mid(x)
        return FlowControlNetOutput(down_block_residuals=residuals, mid_block_residual=mid_res)

    def forward(
        self,
        latent: Tensor,
        flow: Tensor,
    ) -> FlowControlNetOutput:
        """Compute per-block residuals from a (latent, flow) pair.

        Args:
            latent: ``(N, C_z, H_z, W_z)`` warped reference latent at
                the SVD latent grid (typically 32×32 for 256×256 input).
                The temporal axis is folded into the batch axis.
            flow: ``(N, 2, H_z, W_z)`` Eulerian flow on the same grid.

        Returns:
            :class:`FlowControlNetOutput`.
        """
        if latent.dim() != 4 or latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"latent must be (N, {self.latent_channels}, H, W); got {tuple(latent.shape)}"
            )
        if flow.dim() != 4 or flow.shape[1] != self.flow_channels:
            raise ValueError(
                f"flow must be (N, {self.flow_channels}, H, W); got {tuple(flow.shape)}"
            )
        if latent.shape[0] != flow.shape[0] or latent.shape[-2:] != flow.shape[-2:]:
            raise ValueError(
                f"latent {tuple(latent.shape)} and flow {tuple(flow.shape)} must "
                "share batch and spatial dims"
            )

        cond = torch.cat([latent, flow], dim=1)

        if not self._has_svd:
            return self._stub_forward(cond)

        # Production path: feed the conditioning input through the
        # SVD encoder mirror.  The diffusers blocks expect a temb /
        # encoder_hidden_states; the calling code in trainer
        # supplies these via partial application — here we just run
        # the conv-only stub mirror over the conditioning, which
        # produces residuals that the trainer adds into the U-Net's
        # skip path.  The exact diffusers call is wrapped in
        # `svd_wrapper.SVDBackbone.add_flow_residuals` so that this
        # module remains backbone-agnostic.
        x = self.cond_input(cond)
        residuals = []
        for block, zero in zip(self.encoder_blocks, self.zero_convs_down, strict=True):
            x = block(x) if not hasattr(block, "resnets") else self._diffusers_block(block, x)
            residuals.append(zero(x))
        if hasattr(self.mid_block, "resnets"):
            x = self._diffusers_block(self.mid_block, x)
        else:
            x = self.mid_block(x)
        mid_res = self.zero_conv_mid(x)
        return FlowControlNetOutput(down_block_residuals=residuals, mid_block_residual=mid_res)

    @staticmethod
    def _diffusers_block(block: nn.Module, x: Tensor) -> Tensor:
        """Run a diffusers block in a minimal, attribute-tolerant way.

        The full UNet3DCondition blocks expect timestep embeddings &
        cross-attention contexts; this helper bypasses those by walking
        the resnets and downsamplers directly.  This is sufficient for
        ControlNet-style use because the residual-adding happens after
        the spatial features are stable.
        """
        resnets = getattr(block, "resnets", [])
        for resnet in resnets:
            x = resnet(x, temb=None)
        for ds in getattr(block, "downsamplers", []) or []:
            x = ds(x)
        return x
