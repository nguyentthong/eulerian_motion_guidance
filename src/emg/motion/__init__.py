"""Motion-field primitives: differentiable warping, parallel RAFT batching, Eulerian utilities."""

from __future__ import annotations

from emg.motion.eulerian import (
    EulerianFlowSequence,
    adjacent_pair_indices,
    rescale_flow,
)
from emg.motion.parallel_flow import (
    ParallelFlowOutput,
    build_forward_backward_pairs,
    parallel_bidirectional_flow,
)
from emg.motion.warping import (
    backward_warp,
    flow_to_grid,
    sample_flow_at_flow,
)

__all__ = [
    "EulerianFlowSequence",
    "ParallelFlowOutput",
    "adjacent_pair_indices",
    "backward_warp",
    "build_forward_backward_pairs",
    "flow_to_grid",
    "parallel_bidirectional_flow",
    "rescale_flow",
    "sample_flow_at_flow",
]
