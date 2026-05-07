"""Bidirectional Geometric Consistency (Sec. 4.2).

This module implements the three numerical primitives that constitute
the paper's main technical contribution:

* :func:`cycle_consistency_energy` — Eq. 8.
* :func:`dynamic_occlusion_mask`   — Eq. 9.
* :func:`geometric_consistency_loss` — Eq. 10.

Defaults match the paper:  ``α₁ = 0.01``, ``α₂ = 0.5``, ``λ_geo`` is
applied externally to combine with the diffusion loss.

All three primitives are written as pure functions.  A thin
:class:`BidirectionalGeometricConsistency` class binds them with the
chosen hyperparameters for use as a regular ``nn.Module`` from the
trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from emg.motion.warping import backward_warp, sample_flow_at_flow

__all__ = [
    "BidirectionalGeometricConsistency",
    "GeometricLossOutput",
    "cycle_consistency_energy",
    "dynamic_occlusion_mask",
    "geometric_consistency_loss",
]


@dataclass(slots=True)
class GeometricLossOutput:
    """Bundle returned by :class:`BidirectionalGeometricConsistency`.

    Attributes:
        loss: Scalar tensor — the masked, normalised L1 loss of Eq. 10.
        mask: ``(B, 1, H, W)`` float tensor in ``{0, 1}`` — the binary
            validity mask :math:`M_{occ}` (1 = valid, 0 = occluded).
        cycle_energy: ``(B, 1, H, W)`` float tensor — the squared
            residual ``E_cycle``.
    """

    loss: Tensor
    mask: Tensor
    cycle_energy: Tensor


def cycle_consistency_energy(flow_fwd: Tensor, flow_bwd: Tensor) -> Tensor:
    """Compute the per-pixel cycle-consistency energy (Eq. 8).

    .. math::

        E_{\\text{cycle}}(x)
            = \\bigl\\| f_{t \\to t+1}(x)
                + \\mathcal{F}\\bigl(f_{t+1 \\to t}, f_{t \\to t+1}\\bigr)(x)
              \\bigr\\|^2_2

    The operator :math:`\\mathcal{F}` is implemented by
    :func:`emg.motion.warping.sample_flow_at_flow`: the backward flow is
    bilinearly resampled at the location indicated by the forward flow.

    Args:
        flow_fwd: ``(B, 2, H, W)`` forward flow ``f_{t→t+1}``.
        flow_bwd: ``(B, 2, H, W)`` backward flow ``f_{t+1→t}``.

    Returns:
        ``(B, 1, H, W)`` non-negative cycle energy.
    """
    if flow_fwd.shape != flow_bwd.shape:
        raise ValueError(
            f"flow_fwd {tuple(flow_fwd.shape)} and flow_bwd "
            f"{tuple(flow_bwd.shape)} must have the same shape"
        )
    if flow_fwd.dim() != 4 or flow_fwd.shape[1] != 2:
        raise ValueError(f"flows must be (B, 2, H, W); got {tuple(flow_fwd.shape)}")

    sampled_bwd = sample_flow_at_flow(flow_bwd, flow_fwd)
    residual = flow_fwd + sampled_bwd  # (B, 2, H, W)
    energy = residual.pow(2).sum(dim=1, keepdim=True)
    return energy


def dynamic_occlusion_mask(
    flow_fwd: Tensor,
    flow_bwd: Tensor,
    *,
    alpha1: float = 0.01,
    alpha2: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Derive the binary validity mask from cycle energy (Eq. 9).

    The mask is

    .. math::

        M_{\\text{occ}}(x) = \\mathbb{1}\\!\\left[
            E_{\\text{cycle}}(x)
            < \\alpha_1
              \\bigl(\\|f_{t \\to t+1}(x)\\|^2
                  + \\|\\mathcal{W}(f_{t+1 \\to t}, f_{t \\to t+1})(x)\\|^2\\bigr)
            + \\alpha_2
        \\right]

    where the indicator ``1[·]`` is 1 inside the (geometrically valid)
    region and 0 in the occluded region.  ``α₁`` provides a relative
    tolerance that scales with motion magnitude — strong motion is
    expected to incur a larger absolute residual; ``α₂`` is the static
    noise floor that prevents over-aggressive masking on plain regions
    (see Appendix D).

    Args:
        flow_fwd: ``(B, 2, H, W)`` forward flow.
        flow_bwd: ``(B, 2, H, W)`` backward flow.
        alpha1: Dynamic threshold weight (default 0.01).
        alpha2: Static noise floor (default 0.5).

    Returns:
        ``(mask, energy)`` where ``mask`` is ``(B, 1, H, W)`` in ``{0,1}``
        and ``energy`` is the unmasked cycle energy ``(B, 1, H, W)``.
    """
    if alpha1 < 0 or alpha2 < 0:
        raise ValueError(f"alpha1 and alpha2 must be >= 0; got {alpha1=}, {alpha2=}")

    energy = cycle_consistency_energy(flow_fwd, flow_bwd)
    sampled_bwd = sample_flow_at_flow(flow_bwd, flow_fwd)

    fwd_sq = flow_fwd.pow(2).sum(dim=1, keepdim=True)
    sampled_sq = sampled_bwd.pow(2).sum(dim=1, keepdim=True)
    threshold = alpha1 * (fwd_sq + sampled_sq) + alpha2

    mask = (energy < threshold).to(flow_fwd.dtype)
    return mask, energy


def geometric_consistency_loss(
    z_t: Tensor,
    z_t_plus_1: Tensor,
    flow_fwd: Tensor,
    mask: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Masked, normalised L1 loss between warped and target latent (Eq. 10).

    .. math::

        \\mathcal{L}_{\\text{geo}}
        = \\frac{1}{\\sum_x M_{\\text{occ}}(x) + \\epsilon}
          \\sum_{x \\in \\Omega} M_{\\text{occ}}(x)
              \\bigl\\| \\mathcal{W}(\\hat z_t, f_{t \\to t+1})(x)
                       - \\hat z_{t+1}(x) \\bigr\\|_1

    The normalisation by the mask sum is critical: without it, the
    effective loss magnitude would shrink with the number of valid
    pixels, weakening supervision on heavily occluded clips.

    Args:
        z_t: ``(B, C, H, W)`` predicted latent at time ``t``.
        z_t_plus_1: ``(B, C, H, W)`` predicted latent at time ``t+1``.
        flow_fwd: ``(B, 2, H, W)`` Eulerian forward flow at the resolution
            of the latents (call :func:`emg.motion.eulerian.rescale_flow`
            beforehand to bring pixel-space flows to latent-space).
        mask: ``(B, 1, H, W)`` validity mask in ``{0, 1}``.
        eps: Small constant added to the mask normaliser.

    Returns:
        Scalar tensor — the spatially weighted L1.
    """
    if z_t.shape != z_t_plus_1.shape:
        raise ValueError(
            f"z_t {tuple(z_t.shape)} and z_{{t+1}} {tuple(z_t_plus_1.shape)} "
            "must have identical shape"
        )
    if z_t.shape[-2:] != flow_fwd.shape[-2:]:
        raise ValueError(
            f"latent {tuple(z_t.shape)} and flow {tuple(flow_fwd.shape)} "
            "must share spatial dims (rescale flow first)"
        )
    if mask.shape[0] != z_t.shape[0] or mask.shape[-2:] != z_t.shape[-2:]:
        raise ValueError("mask must align with latents in batch and spatial dims")
    if mask.shape[1] != 1:
        raise ValueError(f"mask must have a single channel; got {mask.shape[1]}")

    warped = backward_warp(z_t, flow_fwd, mode="bilinear", padding_mode="zeros")
    diff = (warped - z_t_plus_1).abs()  # (B, C, H, W)
    diff = diff.mean(dim=1, keepdim=True)  # average across channels first

    weighted = diff * mask
    denom = mask.sum() + eps
    return weighted.sum() / denom


class BidirectionalGeometricConsistency(nn.Module):
    """``nn.Module`` wrapper bundling the three BGC primitives.

    This is what the trainer actually consumes — instantiate once with
    the chosen ``α₁, α₂``, then call ``forward(z_t, z_{t+1}, flow_fwd,
    flow_bwd)`` to get the masked loss together with diagnostic
    tensors.

    Attributes:
        alpha1: Relative threshold weight (Eq. 9).
        alpha2: Static noise floor (Eq. 9).
        eps: Mask normaliser epsilon.
    """

    alpha1: float
    alpha2: float
    eps: float

    def __init__(
        self,
        alpha1: float = 0.01,
        alpha2: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.eps = float(eps)

    def forward(
        self,
        z_t: Tensor,
        z_t_plus_1: Tensor,
        flow_fwd: Tensor,
        flow_bwd: Tensor,
        *,
        flow_fwd_for_loss: Tensor | None = None,
    ) -> GeometricLossOutput:
        """Run the full BGC pipeline.

        Args:
            z_t: ``(B, C, H_z, W_z)`` predicted latent at time ``t``.
            z_t_plus_1: ``(B, C, H_z, W_z)`` predicted latent at time
                ``t+1``.
            flow_fwd: ``(B, 2, H, W)`` forward Eulerian flow at flow
                resolution (typically pixel space, 256×256).
            flow_bwd: ``(B, 2, H, W)`` backward flow at the same
                resolution.
            flow_fwd_for_loss: optional pre-rescaled flow at *latent*
                resolution to use in :func:`geometric_consistency_loss`.
                If ``None``, the forward flow at the input resolution is
                bilinearly resampled here.

        Returns:
            :class:`GeometricLossOutput`.
        """
        # Build the mask at the (full) flow resolution per Eq. 9.
        mask_full, energy_full = dynamic_occlusion_mask(
            flow_fwd,
            flow_bwd,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
        )

        # Rescale to latent grid for the loss computation.
        from emg.motion.eulerian import rescale_flow  # local import avoids cycle

        latent_size = (z_t.shape[-2], z_t.shape[-1])
        if flow_fwd_for_loss is None:
            flow_fwd_for_loss = rescale_flow(flow_fwd, latent_size)
        else:
            if flow_fwd_for_loss.shape[-2:] != latent_size:
                raise ValueError(
                    f"flow_fwd_for_loss must match latent size {latent_size}; "
                    f"got {tuple(flow_fwd_for_loss.shape)}"
                )

        # Mask must also live on the latent grid; nearest-mode resize
        # preserves the binary semantics.
        if mask_full.shape[-2:] != latent_size:
            mask_latent = nn.functional.interpolate(
                mask_full,
                size=latent_size,
                mode="nearest",
            )
        else:
            mask_latent = mask_full

        loss = geometric_consistency_loss(
            z_t=z_t,
            z_t_plus_1=z_t_plus_1,
            flow_fwd=flow_fwd_for_loss,
            mask=mask_latent,
            eps=self.eps,
        )
        return GeometricLossOutput(
            loss=loss,
            mask=mask_full,
            cycle_energy=energy_full,
        )
