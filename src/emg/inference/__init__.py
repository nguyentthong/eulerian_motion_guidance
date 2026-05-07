"""Autoregressive inference for the Eulerian generative chain.

See :mod:`emg.inference.autoregressive`.
"""

from __future__ import annotations

from emg.inference.autoregressive import (
    AutoregressiveAnimator,
    AutoregressiveOutput,
    InferenceConfig,
)

__all__ = [
    "AutoregressiveAnimator",
    "AutoregressiveOutput",
    "InferenceConfig",
]
