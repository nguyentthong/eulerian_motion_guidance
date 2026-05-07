"""Frozen RAFT optical-flow estimator wrapper.

The paper uses RAFT (Teed & Deng, 2020) pretrained on FlyingThings3D and
explicitly **not** fine-tuned.  We wrap :func:`torchvision.models.optical_flow.raft_large`
behind the :class:`emg.motion.parallel_flow.FlowEstimator` protocol so it
plugs straight into the parallel-flow machinery.

A graceful fallback is provided when torchvision flow models are not
installed (e.g. on minimal CI machines): the wrapper raises a
descriptive error at instantiation time but the import of this module
itself never fails, so unrelated unit tests can still run.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from emg.utils.logging import get_logger

__all__ = ["RAFTFlowEstimator", "build_raft_estimator"]


_log = get_logger()

# Default ImageNet stats — RAFT in torchvision expects values in [-1, 1].
_RAFT_MEAN = (0.5, 0.5, 0.5)
_RAFT_STD = (0.5, 0.5, 0.5)


class RAFTFlowEstimator(nn.Module):
    """Frozen RAFT wrapper.

    The wrapper:

    * Loads ``raft_large`` with ``Things_V1`` (FlyingThings3D) weights.
    * Disables gradients and sets the module to ``eval`` mode.
    * Resizes inputs to ``size`` (default 256×256) and rescales the
      output flow to match the original input resolution.
    * Implements the
      :class:`emg.motion.parallel_flow.FlowEstimator` protocol.

    Attributes:
        size: Spatial size at which RAFT is run.
    """

    def __init__(
        self,
        *,
        size: int = 256,
        weights: Literal["things"] = "things",
        num_flow_updates: int = 12,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models.optical_flow import (
                Raft_Large_Weights,
                raft_large,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "torchvision >= 0.13 with optical-flow models is required for RAFT. "
                f"Original import error: {exc}"
            ) from exc

        if weights == "things":
            wts = Raft_Large_Weights.C_T_V1
        else:  # pragma: no cover - reserved for future weights
            raise ValueError(f"Unknown RAFT weights tag: {weights}")

        _log.info("Loading RAFT-Large (Things) weights — frozen.")
        self.model = raft_large(weights=wts, progress=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.size = int(size)
        self.num_flow_updates = int(num_flow_updates)

        self.register_buffer(
            "mean",
            torch.tensor(_RAFT_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(_RAFT_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def _normalise(self, x: Tensor) -> Tensor:
        # x in [0, 1] -> [-1, 1] via (x - 0.5) / 0.5
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, image1: Tensor, image2: Tensor) -> Tensor:
        """Estimate flow from ``image1`` to ``image2``.

        Args:
            image1: ``(N, 3, H, W)`` source image batch in ``[0, 1]``.
            image2: ``(N, 3, H, W)`` target image batch in ``[0, 1]``.

        Returns:
            ``(N, 2, H, W)`` flow in pixel units of the *input* resolution.
        """
        if image1.shape != image2.shape or image1.dim() != 4 or image1.shape[1] != 3:
            raise ValueError(
                f"image1/2 must be matching (N, 3, H, W); got "
                f"{tuple(image1.shape)} vs {tuple(image2.shape)}"
            )
        h_src, w_src = image1.shape[-2:]

        # Resize to the operating resolution.
        if (h_src, w_src) != (self.size, self.size):
            x1 = F.interpolate(
                image1, size=(self.size, self.size), mode="bilinear", align_corners=True
            )
            x2 = F.interpolate(
                image2, size=(self.size, self.size), mode="bilinear", align_corners=True
            )
        else:
            x1, x2 = image1, image2

        x1 = self._normalise(x1.clamp(0.0, 1.0))
        x2 = self._normalise(x2.clamp(0.0, 1.0))

        flow_predictions = self.model(x1, x2, num_flow_updates=self.num_flow_updates)
        # torchvision returns a list; the last entry is the highest-quality flow.
        flow = flow_predictions[-1] if isinstance(flow_predictions, list) else flow_predictions

        # Rescale flow back to original resolution.
        if (h_src, w_src) != (self.size, self.size):
            flow = F.interpolate(
                flow, size=(h_src, w_src), mode="bilinear", align_corners=True
            )
            sx = w_src / self.size
            sy = h_src / self.size
            scale = flow.new_tensor([sx, sy]).view(1, 2, 1, 1)
            flow = flow * scale
        return flow


def build_raft_estimator(
    *,
    size: int = 256,
    device: str | torch.device | None = None,
) -> RAFTFlowEstimator:
    """Convenience factory.

    Args:
        size: Operating resolution, defaults to 256 (paper).
        device: Device to place the model on.

    Returns:
        A :class:`RAFTFlowEstimator` ready for inference.
    """
    estimator = RAFTFlowEstimator(size=size)
    if device is not None:
        estimator = estimator.to(device)
    return estimator
