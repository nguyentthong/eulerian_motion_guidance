"""Tests for the dynamic occlusion mask (Eq. 9)."""

from __future__ import annotations

import pytest
import torch

from emg.losses.geometric import (
    BidirectionalGeometricConsistency,
    dynamic_occlusion_mask,
)


def test_zero_flows_yield_full_validity() -> None:
    """When forward and backward flows are zero everywhere, the cycle
    energy is zero and the threshold is α₂ > 0; therefore every pixel
    should be marked as valid (mask = 1)."""
    fwd = torch.zeros(2, 2, 8, 8)
    bwd = torch.zeros(2, 2, 8, 8)
    mask, energy = dynamic_occlusion_mask(fwd, bwd, alpha1=0.01, alpha2=0.5)
    assert mask.shape == (2, 1, 8, 8)
    assert torch.all(mask == 1.0)
    assert torch.all(energy == 0.0)


def test_inconsistent_flows_marked_occluded() -> None:
    """Strong cycle violation -> mask = 0 (occluded)."""
    fwd = torch.full((1, 2, 8, 8), 5.0)
    bwd = torch.zeros(1, 2, 8, 8)
    mask, energy = dynamic_occlusion_mask(fwd, bwd, alpha1=0.001, alpha2=0.001)
    # Cycle energy >> threshold here.  Interior should be flagged.
    interior_mask = mask[0, 0, 1:-1, 1:-1]
    assert torch.all(interior_mask == 0.0)


def test_alpha2_relaxes_mask() -> None:
    """Increasing α₂ should monotonically increase the number of valid pixels."""
    torch.manual_seed(0)
    fwd = torch.randn(1, 2, 8, 8)
    bwd = -fwd + 0.3 * torch.randn(1, 2, 8, 8)  # near-consistent, with noise
    m_low, _ = dynamic_occlusion_mask(fwd, bwd, alpha1=0.01, alpha2=0.1)
    m_high, _ = dynamic_occlusion_mask(fwd, bwd, alpha1=0.01, alpha2=2.0)
    assert m_high.sum() >= m_low.sum()


def test_alpha1_relaxes_mask_for_high_motion() -> None:
    """Larger α₁ should allow more pixels through in high-motion regions."""
    torch.manual_seed(1)
    fwd = 3.0 * torch.randn(1, 2, 8, 8)
    bwd = -fwd + 0.5 * torch.randn(1, 2, 8, 8)
    m_low, _ = dynamic_occlusion_mask(fwd, bwd, alpha1=0.001, alpha2=0.5)
    m_high, _ = dynamic_occlusion_mask(fwd, bwd, alpha1=0.5, alpha2=0.5)
    assert m_high.sum() >= m_low.sum()


def test_negative_alpha_raises() -> None:
    fwd = torch.zeros(1, 2, 4, 4)
    bwd = torch.zeros(1, 2, 4, 4)
    with pytest.raises(ValueError):
        dynamic_occlusion_mask(fwd, bwd, alpha1=-0.01, alpha2=0.5)


def test_bidirectional_module_runs() -> None:
    bgc = BidirectionalGeometricConsistency(alpha1=0.01, alpha2=0.5)
    fwd = torch.zeros(2, 2, 16, 16)
    bwd = torch.zeros(2, 2, 16, 16)
    z_t = torch.randn(2, 4, 16, 16)
    z_tp1 = z_t.clone()  # zero motion -> warped == target -> loss should be ~0
    out = bgc(z_t=z_t, z_t_plus_1=z_tp1, flow_fwd=fwd, flow_bwd=bwd)
    assert out.loss.shape == ()
    assert out.loss.item() < 1e-5
    assert torch.all(out.mask == 1.0)
