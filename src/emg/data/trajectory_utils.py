"""Sparse trajectory hint utilities.

A *sparse hint* is the per-pair velocity at a small number of pixels.
The S2D network expands these hints into a dense Eulerian flow field.

We provide two entry points:

* :func:`sample_random_trajectories` — used during training to harvest
  sparse hints from RAFT pseudo-ground-truth flows.
* :func:`rasterise_hints` — converts a list of
  :class:`SparseHint` into the 3-channel ``(u, v, mask)`` tensor that
  the S2D network consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "SparseHint",
    "rasterise_hints",
    "sample_random_trajectories",
]


@dataclass(slots=True)
class SparseHint:
    """A single (point, velocity) hint for a particular adjacent pair.

    Attributes:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
        u: Horizontal velocity (in pixels) ``f_x(t→t+1)``.
        v: Vertical velocity (in pixels) ``f_y(t→t+1)``.
    """

    x: int
    y: int
    u: float
    v: float


def rasterise_hints(
    hints: list[list[SparseHint]],
    *,
    height: int,
    width: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Rasterise a list-of-lists of hints into a sparse hint tensor.

    Args:
        hints: ``T-1`` lists, one per adjacent pair.  Each list contains
            an arbitrary number of :class:`SparseHint`.
        height: Spatial height ``H``.
        width: Spatial width ``W``.
        device: Optional device for the output tensor.

    Returns:
        ``(T-1, 3, H, W)`` tensor whose channels are ``(u, v, mask)``.
        ``u`` and ``v`` are zero outside hint locations; ``mask`` is 1
        at hint locations and 0 elsewhere.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height/width must be positive; got {height}x{width}")
    n_pairs = len(hints)
    out = torch.zeros((n_pairs, 3, height, width), dtype=torch.float32, device=device)
    for t, pair_hints in enumerate(hints):
        for h in pair_hints:
            if not (0 <= h.x < width and 0 <= h.y < height):
                continue
            out[t, 0, h.y, h.x] = float(h.u)
            out[t, 1, h.y, h.x] = float(h.v)
            out[t, 2, h.y, h.x] = 1.0
    return out


def sample_random_trajectories(
    flow_fwd: Tensor,
    *,
    num_points: int = 32,
    seed: int | None = None,
) -> list[list[SparseHint]]:
    """Sample sparse hints from a dense forward-flow stack.

    For each adjacent pair, we pick ``num_points`` random pixel
    coordinates (uniformly in the spatial domain) and record their
    flow vector.  These hints simulate the user-supplied trajectories
    that drive inference; during training they let the S2D net learn
    to densify partial hints.

    Args:
        flow_fwd: ``(T-1, 2, H, W)`` forward flow at pixel resolution.
        num_points: Hints per adjacent pair.
        seed: Optional RNG seed for deterministic sampling.

    Returns:
        ``T-1`` lists of :class:`SparseHint`.
    """
    if flow_fwd.dim() != 4 or flow_fwd.shape[1] != 2:
        raise ValueError(f"flow_fwd must be (T-1, 2, H, W); got {tuple(flow_fwd.shape)}")
    n_pairs, _, h, w = flow_fwd.shape
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))

    out: list[list[SparseHint]] = []
    for t in range(n_pairs):
        ys = torch.randint(0, h, (num_points,), generator=g)
        xs = torch.randint(0, w, (num_points,), generator=g)
        flow_t = flow_fwd[t].detach().cpu()
        pair_hints: list[SparseHint] = []
        for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
            u = float(flow_t[0, y, x])
            v = float(flow_t[1, y, x])
            pair_hints.append(SparseHint(x=int(x), y=int(y), u=u, v=v))
        out.append(pair_hints)
    return out
