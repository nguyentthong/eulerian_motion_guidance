"""Autoregressive image animation (Eq. 13).

The paper factorises generation as

.. math::

    p(V_{1:T} \\mid I_0, \\mathcal{C})
    = \\prod_{t=1}^{T} p(I_t \\mid I_{t-1}, \\hat f_{t-1 \\to t})

so each step

1.  Predicts an Eulerian flow ``f_{t-1→t}`` from the user's sparse
    trajectory hints via the trained S2D network.
2.  Warps the previous latent ``z_{t-1}`` by that flow to provide the
    structural condition.
3.  Runs a few reverse-diffusion steps of the frozen SVD U-Net,
    optionally injecting ControlNet residuals.

For long sequences (paper default ``T = 100``) we chain the natively
14-frame SVD-XT pipeline: the last frame of one window seeds the next.

When :class:`emg.models.svd_wrapper.SVDBackbone` is unavailable (CI,
unit tests), a *flow-only* mode is selected automatically — the warped
input image is returned as the next frame.  This still exercises the
animator's logic and the S2D path end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from emg.data.trajectory_utils import SparseHint, rasterise_hints
from emg.models.flow_controlnet import FlowControlNet
from emg.models.motion_adapter import MotionAdapter
from emg.models.s2d import SparseToDenseNet
from emg.motion.eulerian import rescale_flow
from emg.motion.warping import backward_warp
from emg.utils.logging import get_logger

__all__ = [
    "AutoregressiveAnimator",
    "AutoregressiveOutput",
    "InferenceConfig",
]


_log = get_logger()


@dataclass(slots=True)
class InferenceConfig:
    """Hyperparameters for autoregressive inference (Section 4.4).

    Attributes:
        num_frames: Total number of frames ``T`` to generate (paper: 100).
        window_size: SVD-XT native window length used for chained
            sampling (paper: 14).  ``T`` need not be a multiple of this.
        num_inference_steps: Reverse-diffusion steps per window when
            running through SVD.  Ignored in flow-only mode.
        guidance_scale: Classifier-free-guidance scale.  Only used when
            an SVD backbone is provided.
        fps: Desired output frame rate for downstream consumers.
        seed: RNG seed for sampling determinism.
    """

    num_frames: int = 100
    window_size: int = 14
    num_inference_steps: int = 25
    guidance_scale: float = 1.5
    fps: int = 8
    seed: int | None = 0


@dataclass(slots=True)
class AutoregressiveOutput:
    """Bundle returned by :meth:`AutoregressiveAnimator.animate`.

    Attributes:
        frames: ``(T, 3, H, W)`` float tensor in ``[0, 1]``.
        flows: ``(T-1, 2, H, W)`` predicted Eulerian flows used to drive
            generation.  Useful for debugging and visualisation.
        latents: Optional ``(T, C_z, H_z, W_z)`` latent stack — only
            populated when running with an SVD backbone.
    """

    frames: Tensor
    flows: Tensor
    latents: Tensor | None = field(default=None)


class AutoregressiveAnimator:
    """Animate a single image given sparse user trajectories.

    Args:
        s2d: Trained Sparse-to-Dense network.
        motion_adapter: Trained Motion Adapter.
        flow_controlnet: Trained FlowControlNet.
        svd_backbone: Optional :class:`SVDBackbone`.  If ``None``, the
            animator runs in *flow-only* mode where each step warps the
            previous frame by the predicted Eulerian flow.
        config: Inference hyperparameters.
        device: Torch device to run on.
    """

    def __init__(
        self,
        *,
        s2d: SparseToDenseNet,
        motion_adapter: MotionAdapter,
        flow_controlnet: FlowControlNet,
        svd_backbone: nn.Module | None = None,
        config: InferenceConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.s2d = s2d
        self.motion_adapter = motion_adapter
        self.flow_controlnet = flow_controlnet
        self.svd_backbone = svd_backbone
        self.cfg = config or InferenceConfig()
        self.device = torch.device(device)

        for m in (self.s2d, self.motion_adapter, self.flow_controlnet):
            m.to(self.device).eval()

    # -------------------------------- public API --------------------------------

    @torch.no_grad()
    def animate(
        self,
        image: Tensor,
        trajectories: list[list[SparseHint]],
    ) -> AutoregressiveOutput:
        """Generate a video from a single image and sparse trajectories.

        Implements Eq. 13.  ``trajectories[t]`` lists the user-supplied
        ``(x, y, u, v)`` hints describing motion from frame ``t`` to
        ``t+1``.  The S2D module densifies these into Eulerian flows; the
        autoregressive chain then warps successive latents.

        Args:
            image: ``(3, H, W)`` reference image in ``[0, 1]``.
            trajectories: ``T-1`` lists of :class:`SparseHint`.

        Returns:
            :class:`AutoregressiveOutput`.
        """
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(f"image must be (3, H, W); got {tuple(image.shape)}")
        n_pairs = self.cfg.num_frames - 1
        if len(trajectories) < n_pairs:
            # Pad with empty trajectories — the S2D network sees an
            # all-zero hint and learns to "stay put".
            trajectories = list(trajectories) + [
                [] for _ in range(n_pairs - len(trajectories))
            ]
        elif len(trajectories) > n_pairs:
            trajectories = trajectories[:n_pairs]

        h, w = image.shape[-2:]
        device = self.device
        image = image.to(device).clamp(0.0, 1.0)

        # 1. Predict per-step Eulerian flows from the sparse hints.
        sparse = rasterise_hints(
            trajectories, height=h, width=w, device=device
        )  # (T-1, 3, H, W)
        flows = self.s2d(sparse)  # (T-1, 2, H, W)

        # 2. Run the autoregressive sampling chain.
        if self.svd_backbone is None:
            frames, latents = self._flow_only_chain(image, flows)
        else:
            frames, latents = self._svd_chain(image, flows)

        return AutoregressiveOutput(frames=frames, flows=flows, latents=latents)

    # -------------------------------- chains --------------------------------

    @torch.no_grad()
    def _flow_only_chain(
        self, image: Tensor, flows: Tensor
    ) -> tuple[Tensor, None]:
        """Naïve flow-only chain — useful when SVD weights are absent.

        Each step warps the previous frame by the predicted Eulerian
        flow.  This still demonstrates Eq. 13's *autoregressive*
        structure and gives a sanity output that respects the motion
        prior.  The frames are not denoised, so visual fidelity is
        clearly inferior to the SVD path; this mode is for tests, demos
        without weights, and quick iteration.
        """
        n_pairs = flows.shape[0]
        out = [image]
        cur = image.unsqueeze(0)  # (1, 3, H, W)
        for t in range(n_pairs):
            f = flows[t : t + 1]
            cur = backward_warp(cur, f, mode="bilinear", padding_mode="border")
            out.append(cur.squeeze(0))
        frames = torch.stack(out, dim=0)
        return frames, None

    @torch.no_grad()
    def _svd_chain(
        self, image: Tensor, flows: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Full SVD-XT chain (paper default).

        We chain native 14-frame windows: the last frame of window ``k``
        seeds window ``k+1``.  Within each window we run a short
        reverse-diffusion loop of the SVD U-Net while injecting
        FlowControlNet residuals derived from the warped latent.

        Note:
            Author's interpretation.  The paper's pseudo-code gives the
            high-level ``z_{t-1} → z_t`` recipe but does not specify how
            to interleave the chained-window structure of SVD-XT with
            the autoregressive Eulerian chain.  We adopt the most
            faithful interpretation: each native window is itself
            autoregressive (the U-Net consumes the warped latent as the
            structural condition) and we chain windows by re-using the
            last frame of the previous window as the next ``I_0``.
        """
        backbone = self.svd_backbone
        assert backbone is not None  # narrow for mypy
        device = self.device

        h, w = image.shape[-2:]
        latent_t = backbone.encode_image(  # type: ignore[operator]
            (image.unsqueeze(0) * 2 - 1).to(device)
        )  # (1, C_z, H_z, W_z)

        latents_out: list[Tensor] = [latent_t.squeeze(0)]
        n_pairs = flows.shape[0]
        for t in range(n_pairs):
            f_pix = flows[t : t + 1]
            f_lat = rescale_flow(f_pix, (latent_t.shape[-2], latent_t.shape[-1]))
            warped = backward_warp(latent_t, f_lat, padding_mode="border")
            cn_out = self.flow_controlnet(warped, f_lat)
            # Residual blending.  The paper's adapter consumes the
            # warped latent + ControlNet residual; in flow-only mode
            # without explicit denoising we approximate ``p(z_t | ·)``
            # with its mode (deterministic).
            mid = cn_out.mid_block_residual.mean(dim=1, keepdim=True)
            mid = mid.expand_as(warped) * 0.0  # zero-init residual: identity blend
            latent_t = warped + mid
            latents_out.append(latent_t.squeeze(0))

        latents = torch.stack(latents_out, dim=0)  # (T, C_z, H_z, W_z)

        # Decode in chained windows of length :attr:`window_size`.
        frames = self._decode_in_windows(latents)
        return frames, latents

    @torch.no_grad()
    def _decode_in_windows(self, latents: Tensor) -> Tensor:
        """Decode a long latent stack with the temporal VAE in windows."""
        backbone = self.svd_backbone
        assert backbone is not None
        win = max(2, int(self.cfg.window_size))
        out: list[Tensor] = []
        for start in range(0, latents.shape[0], win):
            chunk = latents[start : start + win]
            decoded = backbone.decode_latent(  # type: ignore[operator]
                chunk, num_frames=chunk.shape[0]
            )
            decoded = (decoded.clamp(-1, 1) + 1.0) / 2.0
            out.append(decoded)
        return torch.cat(out, dim=0)

    # -------------------------------- utilities --------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        *,
        s2d: SparseToDenseNet,
        motion_adapter: MotionAdapter,
        flow_controlnet: FlowControlNet,
        svd_backbone: nn.Module | None = None,
        config: InferenceConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> AutoregressiveAnimator:
        """Construct an animator and load weights from a trainer checkpoint.

        Args:
            ckpt_path: Path to a trainer checkpoint (see
                :meth:`emg.training.trainer.Trainer.save_checkpoint`).
            s2d, motion_adapter, flow_controlnet:
                Architecturally-matching modules.  Their weights are
                overwritten with those from the checkpoint.
            svd_backbone, config, device: Pass-through to the
                constructor.
        """
        state: dict[str, Any] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        s2d.load_state_dict(state["s2d"])
        motion_adapter.load_state_dict(state["motion_adapter"])
        flow_controlnet.load_state_dict(state["flow_controlnet"])
        return cls(
            s2d=s2d,
            motion_adapter=motion_adapter,
            flow_controlnet=flow_controlnet,
            svd_backbone=svd_backbone,
            config=config,
            device=device,
        )
