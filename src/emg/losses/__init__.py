"""Loss modules: bidirectional geometric consistency and standard diffusion loss."""

from __future__ import annotations

from emg.losses.diffusion import diffusion_loss
from emg.losses.geometric import (
    BidirectionalGeometricConsistency,
    GeometricLossOutput,
    cycle_consistency_energy,
    dynamic_occlusion_mask,
    geometric_consistency_loss,
)

__all__ = [
    "BidirectionalGeometricConsistency",
    "GeometricLossOutput",
    "cycle_consistency_energy",
    "diffusion_loss",
    "dynamic_occlusion_mask",
    "geometric_consistency_loss",
]
