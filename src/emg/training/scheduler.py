"""Learning-rate schedulers used by the trainer.

We support cosine, linear-warmup-cosine, and constant schedules.  The
paper does not specify which schedule it uses; the trainer defaults to
constant learning rate matching the original paper's lr=2e-5 reading.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

__all__ = ["build_lr_scheduler"]


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: Literal["constant", "cosine", "linear_warmup_cosine"] = "constant",
    total_steps: int = 100_000,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Construct a :class:`LambdaLR` scheduler.

    Args:
        optimizer: The optimiser whose LRs will be scaled.
        schedule: Schedule name.
        total_steps: Total optimisation steps.
        warmup_steps: Linear warm-up steps.
        min_lr_ratio: Floor for the cosine schedule, expressed as a
            fraction of the initial LR.

    Returns:
        A :class:`torch.optim.lr_scheduler.LambdaLR` ready to be stepped
        once per optimiser step.
    """

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if schedule in ("cosine", "linear_warmup_cosine"):
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unknown LR schedule: {schedule}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
