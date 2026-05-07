"""Exponential moving-average parameter shadowing.

A textbook implementation that maintains a CPU- or GPU-resident shadow
copy of every trainable parameter, updated at every optimisation step
with decay ``β``.  The shadow can be temporarily swapped in for
evaluation via :meth:`apply_to`.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn

__all__ = ["EMAWeights"]


class EMAWeights:
    """Maintain an EMA shadow of a model's trainable parameters.

    Args:
        model: The source model.
        decay: Decay factor ``β`` (typical: 0.9999).
        warmup_steps: Number of steps over which to ramp up the decay
            from 0 to ``decay`` for stability.
    """

    def __init__(self, model: nn.Module, *, decay: float = 0.9999, warmup_steps: int = 1000) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1); got {decay}")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self._step = 0
        # Shadow sits on the same device as the source.
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @property
    def step(self) -> int:
        """Number of EMA updates applied."""
        return self._step

    def _effective_decay(self) -> float:
        if self._step < self.warmup_steps:
            return min(self.decay, (1.0 + self._step) / (10.0 + self._step))
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Pull one EMA update from ``model``."""
        decay = self._effective_decay()
        one_minus_decay = 1.0 - decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow = self.shadow[name]
            if shadow.device != param.device:
                shadow = shadow.to(param.device)
                self.shadow[name] = shadow
            shadow.mul_(decay).add_(param.detach(), alpha=one_minus_decay)
        self._step += 1

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialise the shadow."""
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore the shadow from a serialised dict."""
        self.shadow = {k: v.detach().clone() for k, v in state.items()}

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        """Context manager that temporarily swaps the shadow into ``model``.

        Restores the original parameters on exit, even on exception.
        """
        backup = copy.deepcopy({n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad})
        try:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if name in self.shadow:
                        param.data.copy_(self.shadow[name].to(param.device))
            yield
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if name in backup:
                        param.data.copy_(backup[name].to(param.device))
