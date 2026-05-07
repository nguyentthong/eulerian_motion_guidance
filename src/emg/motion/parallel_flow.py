"""Parallelised bidirectional flow computation (Section 4.3, Eq. 11–12).

The paper's training-efficiency contribution is a *batched* construction
of the forward and backward adjacent-frame pairs that lets a single
forward pass through the (frozen) flow estimator regress every flow
field in a clip simultaneously, making per-iteration wall-clock time
``O(1)`` w.r.t. ``T``.

We must produce, for an input video tensor ``V ∈ R^{B,T,C,H,W}``,

* ``P_fwd = {(I_t, I_{t+1})}_{t=0}^{T-2}``
* ``P_bwd = {(I_{t+1}, I_t)}_{t=0}^{T-2}``

and feed them through the estimator ``Φ`` as one stacked batch of size
``2 * B * (T-1)``.  No Python-level loops over ``T`` are permitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from emg.motion.eulerian import EulerianFlowSequence

__all__ = [
    "FlowEstimator",
    "ParallelFlowOutput",
    "build_forward_backward_pairs",
    "parallel_bidirectional_flow",
]


class FlowEstimator(Protocol):
    """Minimal protocol every flow estimator must satisfy.

    Takes an ``image1`` and ``image2`` of shape ``(N, 3, H, W)`` and
    returns a tensor of shape ``(N, 2, H, W)`` representing the flow
    from ``image1`` to ``image2``.  Compatible with our RAFT wrapper.
    """

    def __call__(self, image1: Tensor, image2: Tensor) -> Tensor: ...


@dataclass(slots=True)
class ParallelFlowOutput:
    """Result of :func:`parallel_bidirectional_flow`.

    Attributes:
        flows: bidirectional :class:`EulerianFlowSequence`.
        wall_clock_seconds: optional timing of the single estimator
            forward pass — ``None`` unless ``time_it=True`` was passed.
    """

    flows: EulerianFlowSequence
    wall_clock_seconds: float | None = None


def build_forward_backward_pairs(video: Tensor) -> tuple[Tensor, Tensor]:
    """Construct the forward and backward dyad batches.

    Implements Eq. 11–12.  Given a video tensor
    ``V ∈ R^{B,T,C,H,W}`` we return

    * ``A_fwd, A_bwd, B_fwd, B_bwd`` flattened along the temporal
      axis so that the resulting tensors are 4-D and can be passed
      directly to a vanilla CNN flow estimator.

    The construction uses pure tensor operations — no Python loops.

    Args:
        video: ``(B, T, C, H, W)`` clip with ``T ≥ 2``.

    Returns:
        Tuple ``(pair_a, pair_b)`` where each is a tensor of shape
        ``(2 * B * (T-1), C, H, W)``.  The first ``B * (T-1)`` entries
        of ``pair_a`` are the forward sources ``I_t`` and the same
        entries of ``pair_b`` are the forward targets ``I_{t+1}``.
        The second half flips the order — backward pairs.
    """
    if video.dim() != 5:
        raise ValueError(f"video must be (B, T, C, H, W); got {tuple(video.shape)}")
    b, t, c, h, w = video.shape
    if t < 2:
        raise ValueError(f"Need T >= 2 to form adjacent pairs; got T={t}")

    # P_fwd: (I_t, I_{t+1})
    fwd_a = video[:, :-1]  # (B, T-1, C, H, W)
    fwd_b = video[:, 1:]
    # P_bwd: (I_{t+1}, I_t)
    bwd_a = fwd_b
    bwd_b = fwd_a

    # Flatten (B, T-1) -> B*(T-1)
    fwd_a = fwd_a.reshape(b * (t - 1), c, h, w)
    fwd_b = fwd_b.reshape(b * (t - 1), c, h, w)
    bwd_a = bwd_a.reshape(b * (t - 1), c, h, w)
    bwd_b = bwd_b.reshape(b * (t - 1), c, h, w)

    pair_a = torch.cat([fwd_a, bwd_a], dim=0)  # (2*B*(T-1), C, H, W)
    pair_b = torch.cat([fwd_b, bwd_b], dim=0)
    return pair_a, pair_b


@torch.no_grad()
def parallel_bidirectional_flow(
    video: Tensor,
    estimator: FlowEstimator,
    *,
    time_it: bool = False,
) -> ParallelFlowOutput:
    """Run a single estimator pass that yields all bidirectional flows.

    Implements Eq. 11–12 with one batched call:

    .. code-block:: text

        F_all = Φ(stack(P_fwd, P_bwd))
        F_fwd = F_all[: B*(T-1)]
        F_bwd = F_all[B*(T-1) :]

    Args:
        video: ``(B, T, C, H, W)`` clip; values in ``[0, 1]`` or
            normalised to whatever the estimator expects.
        estimator: An object obeying the :class:`FlowEstimator` protocol.
            Should be frozen (the paper uses a frozen RAFT).
        time_it: If True, time the estimator forward pass and stash the
            measurement on the returned :class:`ParallelFlowOutput`.

    Returns:
        :class:`ParallelFlowOutput` with the bidirectional flows.

    Notes:
        We wrap the call in ``torch.no_grad()`` because RAFT is frozen.
        Gradients can still flow through the geometric-consistency loss
        because the loss is computed against the *latents*, not the
        flow itself.
    """
    pair_a, pair_b = build_forward_backward_pairs(video)
    b, t, _, h, w = video.shape
    n = b * (t - 1)

    start: torch.cuda.Event | None = None
    end: torch.cuda.Event | None = None
    if time_it and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    flow_all = estimator(pair_a, pair_b)  # (2N, 2, H, W)

    elapsed: float | None = None
    if start is not None and end is not None:
        end.record()
        torch.cuda.synchronize()
        elapsed = float(start.elapsed_time(end)) / 1000.0  # ms -> s

    if flow_all.shape != (2 * n, 2, h, w):
        raise RuntimeError(
            f"Estimator returned unexpected shape {tuple(flow_all.shape)}; "
            f"expected ({2 * n}, 2, {h}, {w})"
        )

    fwd = flow_all[:n].reshape(b, t - 1, 2, h, w)
    bwd = flow_all[n:].reshape(b, t - 1, 2, h, w)
    return ParallelFlowOutput(
        flows=EulerianFlowSequence(forward=fwd, backward=bwd),
        wall_clock_seconds=elapsed,
    )
