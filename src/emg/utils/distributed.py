"""Lightweight wrappers around :mod:`torch.distributed`.

We deliberately do not lean on Accelerate as the only entry point —
some users prefer raw DDP with ``torchrun``.  These helpers work with
both and never import Accelerate at module import time.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist

__all__ = [
    "barrier",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "setup_distributed",
]


def _initialised() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the global rank or ``0`` if distributed is not initialised."""
    return dist.get_rank() if _initialised() else 0


def get_world_size() -> int:
    """Return the world size or ``1`` if distributed is not initialised."""
    return dist.get_world_size() if _initialised() else 1


def get_local_rank() -> int:
    """Return ``LOCAL_RANK`` or ``0``."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    """Convenience predicate."""
    return get_rank() == 0


def setup_distributed(
    backend: str = "nccl",
) -> tuple[int, int, int]:
    """Initialise :mod:`torch.distributed` from the environment.

    Reads ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR``,
    ``MASTER_PORT`` (set by ``torchrun``).  If those variables are
    absent, single-process mode is assumed and no init takes place.

    Args:
        backend: Distributed backend.  ``"nccl"`` for GPU, ``"gloo"`` for
            CPU testing.

    Returns:
        ``(rank, world_size, local_rank)``.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    if not dist.is_available():
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if not _initialised():
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank


@contextmanager
def barrier() -> Iterator[None]:
    """Context manager that issues a barrier on entry and exit when DDP is up."""
    if _initialised():
        dist.barrier()
    try:
        yield
    finally:
        if _initialised():
            dist.barrier()
