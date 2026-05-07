"""Tests for the parallel flow batching trick (Eq. 11-12).

Three properties matter here:

1.  The output has shape ``(B, T-1, 2, H, W)`` for both forward and
    backward flows.
2.  The estimator is invoked exactly **once** — this is the whole point
    of Section 4.3.  We use a counting stub estimator to verify.
3.  Forward and backward halves are correctly demultiplexed: a stub
    that returns the *first* image as the flow lets us test that the
    backward half is paired correctly.
"""

from __future__ import annotations

import pytest
import torch

from emg.motion.parallel_flow import (
    build_forward_backward_pairs,
    parallel_bidirectional_flow,
)


class _CountingEstimator:
    """Records how many times it was called and returns zero flow."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        n = image1.shape[0]
        h, w = image1.shape[-2:]
        return torch.zeros(n, 2, h, w)


class _IdentityEstimator:
    """Returns the *first* image's first two channels as the flow.

    This lets the test verify which images actually arrived in the
    forward half vs the backward half: the trainer should pass
    ``I_t`` first for forward pairs and ``I_{t+1}`` first for backward
    pairs.
    """

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        return image1[:, :2].clone()


def test_pair_construction_no_python_loops() -> None:
    """Forward and backward dyads should be assembled without Python-level
    iteration over ``T``.

    The function under test uses pure tensor operations; we verify by
    sanity-checking the resulting shapes and contents.
    """
    b, t, c, h, w = 2, 5, 3, 8, 8
    video = torch.arange(b * t * c * h * w, dtype=torch.float32).reshape(b, t, c, h, w)
    pair_a, pair_b = build_forward_backward_pairs(video)
    n = b * (t - 1)
    assert pair_a.shape == (2 * n, c, h, w)
    assert pair_b.shape == (2 * n, c, h, w)

    # Forward half: pair_a[:n] should be V[:, :-1] flattened.
    expected_fwd_a = video[:, :-1].reshape(n, c, h, w)
    assert torch.equal(pair_a[:n], expected_fwd_a)
    # Forward half target: pair_b[:n] should be V[:, 1:] flattened.
    expected_fwd_b = video[:, 1:].reshape(n, c, h, w)
    assert torch.equal(pair_b[:n], expected_fwd_b)

    # Backward half: roles flip.
    assert torch.equal(pair_a[n:], expected_fwd_b)
    assert torch.equal(pair_b[n:], expected_fwd_a)


def test_parallel_flow_single_call() -> None:
    """The estimator must be invoked exactly once."""
    b, t = 2, 4
    video = torch.rand(b, t, 3, 8, 8)
    est = _CountingEstimator()
    out = parallel_bidirectional_flow(video, est)
    assert est.calls == 1
    assert out.flows.forward.shape == (b, t - 1, 2, 8, 8)
    assert out.flows.backward.shape == (b, t - 1, 2, 8, 8)


def test_parallel_flow_demuxes_correctly() -> None:
    """Forward and backward halves should be unscrambled.

    Using ``_IdentityEstimator`` (which returns the first image as the
    flow), the forward output should equal ``V[:, :-1]`` and the backward
    output should equal ``V[:, 1:]`` — modulo the (2, H, W) channel slice.
    """
    b, t = 2, 4
    video = torch.rand(b, t, 3, 6, 6)
    est = _IdentityEstimator()
    out = parallel_bidirectional_flow(video, est)

    expected_fwd = video[:, :-1, :2]  # (B, T-1, 2, H, W)
    expected_bwd = video[:, 1:, :2]
    assert torch.allclose(out.flows.forward, expected_fwd)
    assert torch.allclose(out.flows.backward, expected_bwd)


def test_parallel_flow_rejects_short_clips() -> None:
    video = torch.rand(1, 1, 3, 8, 8)
    est = _CountingEstimator()
    with pytest.raises(ValueError):
        parallel_bidirectional_flow(video, est)


def test_parallel_flow_rejects_wrong_dim() -> None:
    video = torch.rand(2, 3, 8, 8)  # missing T axis
    est = _CountingEstimator()
    with pytest.raises(ValueError):
        parallel_bidirectional_flow(video, est)
