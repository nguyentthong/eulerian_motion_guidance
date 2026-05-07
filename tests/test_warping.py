"""Tests for the differentiable warping operator W(·, ·).

We exercise three properties that the rest of the codebase depends on:

1.  ``flow_to_grid`` produces the identity grid when the flow is zero.
2.  ``backward_warp`` is the identity when the flow is zero.
3.  ``backward_warp`` is differentiable end-to-end.
4.  ``sample_flow_at_flow`` is sane (zero flow → zero result).
"""

from __future__ import annotations

import pytest
import torch

from emg.motion.warping import backward_warp, flow_to_grid, sample_flow_at_flow


@pytest.fixture
def small_image() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(2, 3, 8, 8)


def test_flow_to_grid_identity_for_zero_flow() -> None:
    flow = torch.zeros(1, 2, 4, 4)
    grid = flow_to_grid(flow)
    # Grid shape is (B, H, W, 2).
    assert grid.shape == (1, 4, 4, 2)
    # Top-left pixel maps to ~ -1 (with align_corners=True), bottom-right to ~ +1.
    assert torch.allclose(grid[0, 0, 0], torch.tensor([-1.0, -1.0]), atol=1e-5)
    assert torch.allclose(grid[0, -1, -1], torch.tensor([1.0, 1.0]), atol=1e-5)


def test_backward_warp_identity(small_image: torch.Tensor) -> None:
    zero_flow = torch.zeros(2, 2, 8, 8)
    out = backward_warp(small_image, zero_flow)
    assert torch.allclose(out, small_image, atol=1e-5)


def test_backward_warp_translation() -> None:
    """``backward_warp(feat, flow)(x) = feat(x + flow(x))``.

    Place a bright pixel at column 2 and apply a uniform flow ``dx = -1``.
    Output pixel ``(2, 3)`` should sample ``feat(2, 3 + (-1)) = feat(2, 2)
    = 1``; the original bright pixel location samples ``feat(2, 1) = 0``.
    """
    img = torch.zeros(1, 1, 5, 5)
    img[0, 0, 2, 2] = 1.0
    flow = torch.zeros(1, 2, 5, 5)
    flow[0, 0] = -1.0
    warped = backward_warp(img, flow, padding_mode="zeros")
    # Bright pixel arrives at column 3 (one to the right of original).
    assert warped[0, 0, 2, 3].item() == pytest.approx(1.0, abs=1e-4)
    # Original location is now empty.
    assert warped[0, 0, 2, 2].item() == pytest.approx(0.0, abs=1e-4)


def test_backward_warp_is_differentiable(small_image: torch.Tensor) -> None:
    flow = torch.zeros(2, 2, 8, 8, requires_grad=True)
    out = backward_warp(small_image, flow)
    out.sum().backward()
    assert flow.grad is not None
    assert flow.grad.shape == flow.shape


def test_sample_flow_at_flow_zero() -> None:
    fwd = torch.zeros(1, 2, 6, 6)
    bwd = torch.zeros(1, 2, 6, 6)
    out = sample_flow_at_flow(bwd, fwd)
    assert out.shape == fwd.shape
    assert torch.all(out == 0.0)


def test_sample_flow_at_flow_consistent_pair() -> None:
    """If forward flow shifts +1 column and backward flow shifts -1
    column, then F(f_bwd, f_fwd) should equal -f_fwd everywhere except
    near the boundary.  This is the key property exploited by Eq. 8.
    """
    h, w = 5, 5
    fwd = torch.zeros(1, 2, h, w)
    fwd[0, 0] = 1.0  # +1 in x everywhere
    bwd = torch.zeros(1, 2, h, w)
    bwd[0, 0] = -1.0  # -1 in x everywhere
    sampled = sample_flow_at_flow(bwd, fwd)
    # Interior pixels should give exactly -1.
    interior = sampled[0, 0, :, :-1]  # exclude rightmost column (boundary)
    assert torch.allclose(interior, -torch.ones_like(interior), atol=1e-5)


def test_warp_shape_mismatch_raises() -> None:
    img = torch.zeros(2, 3, 8, 8)
    flow = torch.zeros(2, 2, 7, 8)  # mismatched H
    with pytest.raises(ValueError):
        backward_warp(img, flow)
