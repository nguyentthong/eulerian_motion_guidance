"""Training utilities: Trainer, EMA, scheduler."""

from __future__ import annotations

from emg.training.ema import EMAWeights
from emg.training.scheduler import build_lr_scheduler
from emg.training.trainer import Trainer, TrainerConfig

__all__ = [
    "EMAWeights",
    "Trainer",
    "TrainerConfig",
    "build_lr_scheduler",
]
