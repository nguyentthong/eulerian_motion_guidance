"""Standard diffusion / v-prediction loss for SVD.

Stable Video Diffusion uses an EDM-style v-prediction objective.  We
provide a thin wrapper computing the per-element MSE of the U-Net's
v-prediction against the analytic target (which the noise scheduler
provides).  The schedule itself is not re-implemented; we accept either
a sample-space target or a noise/v target.

Notation follows the paper: ``L_diff`` is the standard reconstruction
loss applied to the model's velocity field prediction, whose definition
is delegated to the :class:`diffusers.EulerDiscreteScheduler` we use at
training time.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

__all__ = ["diffusion_loss"]


def diffusion_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    weights: Tensor | None = None,
) -> Tensor:
    """MSE diffusion loss with optional EDM weighting.

    Args:
        prediction: ``(B, T, C, H, W)`` model output (v-prediction by
            default for SVD).
        target: tensor of the same shape — the analytic v-target from
            the scheduler.
        weights: Optional per-sample EDM weighting ``(B,)`` or ``(B, T)``.
            Broadcast across spatial dims.

    Returns:
        Scalar tensor — the (weighted) mean of the squared error.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} and target {tuple(target.shape)} "
            "must have identical shape"
        )
    err = F.mse_loss(prediction, target, reduction="none")
    if weights is not None:
        # Reshape weights so they broadcast across channels and spatial dims.
        while weights.dim() < err.dim():
            weights = weights.unsqueeze(-1)
        err = err * weights
    return err.mean()
