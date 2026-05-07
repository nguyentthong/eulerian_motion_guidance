"""Tests for the cycle-consistency energy (Eq. 8)."""

from __future__ import annotations

import torch

from emg.losses.geometric import cycle_consistency_energy


def test_zero_flow_zero_energy() -> None:
    fwd = torch.zeros(2, 2, 8, 8)
    bwd = torch.zeros(2, 2, 8, 8)
    e = cycle_consistency_energy(fwd, bwd)
    assert e.shape == (2, 1, 8, 8)
    assert torch.all(e == 0.0)


def test_perfectly_consistent_flows_have_zero_interior_energy() -> None:
    """A pure +x shift in the forward flow paired with a pure -x shift
    in the backward flow should give ~0 cycle energy in the interior."""
    h, w = 8, 8
    fwd = torch.zeros(1, 2, h, w)
    fwd[0, 0] = 1.0
    bwd = torch.zeros(1, 2, h, w)
    bwd[0, 0] = -1.0
    e = cycle_consistency_energy(fwd, bwd)
    # Interior pixels should be ~0 (not boundary).
    interior = e[0, 0, 1:-1, 1:-2]
    assert torch.all(interior < 1e-4)


def test_inconsistent_flows_have_nonzero_energy() -> None:
    """If forward and backward flows do not cycle, energy must be > 0."""
    h, w = 8, 8
    fwd = torch.full((1, 2, h, w), 2.0)  # +2 in both dirs
    bwd = torch.zeros(1, 2, h, w)
    e = cycle_consistency_energy(fwd, bwd)
    # E_cycle = ||fwd + F(bwd, fwd)||^2 = ||fwd + 0||^2 = ||fwd||^2 = 8.
    interior = e[0, 0, 1:-1, 1:-1]
    assert torch.all(interior > 1.0)


def test_energy_is_nonnegative() -> None:
    torch.manual_seed(0)
    fwd = torch.randn(1, 2, 8, 8)
    bwd = torch.randn(1, 2, 8, 8)
    e = cycle_consistency_energy(fwd, bwd)
    assert torch.all(e >= 0.0)
