"""Evaluation metrics and orchestrator.

The metric suite reproduces Tables 1, 2, 3 of the paper:

* :func:`emg.evaluation.metrics.compute_lpips`
* :func:`emg.evaluation.metrics.compute_fid`
* :func:`emg.evaluation.metrics.compute_fvd`
* :func:`emg.evaluation.metrics.compute_clip_consistency`
* :func:`emg.evaluation.metrics.compute_warping_error`
* :func:`emg.evaluation.metrics.compute_cpbd`
* :func:`emg.evaluation.metrics.compute_arcface`

The :class:`emg.evaluation.evaluator.Evaluator` glues them together and
emits a JSON report plus a Markdown table in the paper's format.
"""

from __future__ import annotations

from emg.evaluation.evaluator import Evaluator, EvaluatorConfig
from emg.evaluation.metrics import (
    compute_arcface,
    compute_clip_consistency,
    compute_cpbd,
    compute_fid,
    compute_fvd,
    compute_lpips,
    compute_warping_error,
)

__all__ = [
    "Evaluator",
    "EvaluatorConfig",
    "compute_arcface",
    "compute_clip_consistency",
    "compute_cpbd",
    "compute_fid",
    "compute_fvd",
    "compute_lpips",
    "compute_warping_error",
]
