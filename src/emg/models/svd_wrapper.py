"""Stable Video Diffusion (SVD) frozen backbone wrapper.

We load the pretrained ``stabilityai/stable-video-diffusion-img2vid-xt``
pipeline via :mod:`diffusers` and expose:

* :attr:`unet` — the spatiotemporal denoiser.
* :attr:`vae` — the latent VAE (frozen).
* :attr:`image_encoder` — the CLIP image conditioner (frozen).
* :attr:`scheduler` — the EDM-style sampler.

All three components are set to ``eval()`` and have ``requires_grad =
False``.  Only the FlowControlNet, S2D, and Motion Adapter are trained;
this wrapper provides the helpers they need (encode an image, denoise
a latent with optional ControlNet residuals, decode back to RGB).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from emg.utils.logging import get_logger

__all__ = ["SVDBackbone", "build_svd_backbone"]


_log = get_logger()

DEFAULT_SVD_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"


@dataclass(slots=True)
class _LatentSpec:
    channels: int
    spatial_downsample: int  # latent grid is input_size // spatial_downsample


class SVDBackbone(nn.Module):
    """Wrapper for the frozen SVD model.

    The wrapper *deliberately* does not subclass :class:`StableVideoDiffusionPipeline`;
    we hold individual modules so we can interleave our ControlNet
    residuals between U-Net blocks without monkey-patching diffusers
    internals.

    Args:
        model_id: HuggingFace identifier or local path.
        dtype: Torch dtype to load weights in.
        local_files_only: If True, only use cached files (no network
            access).  Useful for sandboxed environments.

    Attributes:
        unet, vae, image_encoder, image_processor, scheduler:
            The individual components.
        latent_spec: Information about the latent space.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_SVD_MODEL,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        try:
            from diffusers import (  # type: ignore[attr-defined]
                AutoencoderKLTemporalDecoder,
                EulerDiscreteScheduler,
                UNetSpatioTemporalConditionModel,
            )
            from transformers import (
                CLIPImageProcessor,
                CLIPVisionModelWithProjection,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "diffusers >= 0.30 and transformers are required to load SVD. "
                f"Original error: {exc}"
            ) from exc

        kwargs: dict[str, Any] = dict(local_files_only=local_files_only)

        _log.info("Loading SVD components from %s", model_id)
        self.unet = UNetSpatioTemporalConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=dtype, **kwargs
        )
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            model_id, subfolder="vae", torch_dtype=dtype, **kwargs
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_id, subfolder="image_encoder", torch_dtype=dtype, **kwargs
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_id, subfolder="feature_extractor", **kwargs
        )
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler", **kwargs
        )

        self.unet.eval()
        self.vae.eval()
        self.image_encoder.eval()
        for p in self.parameters():
            p.requires_grad = False

        # Determine latent spec.
        latent_channels = int(getattr(self.vae.config, "latent_channels", 4))
        spatial_downsample = 2 ** (
            len(getattr(self.vae.config, "block_out_channels", [128, 256, 512, 512])) - 1
        )
        self.latent_spec = _LatentSpec(latent_channels, spatial_downsample)

    @torch.no_grad()
    def encode_image(self, image: Tensor) -> Tensor:
        """Encode an image batch to latent space via the SVD VAE.

        Args:
            image: ``(N, 3, H, W)`` RGB tensor in ``[-1, 1]``.

        Returns:
            ``(N, C_z, H/8, W/8)`` latent.
        """
        return self.vae.encode(image).latent_dist.mode() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: Tensor, *, num_frames: int) -> Tensor:
        """Decode a latent batch back to RGB.

        Args:
            latent: ``(N, C_z, H_z, W_z)`` latent.
            num_frames: Number of frames per sequence (for the temporal
                decoder).

        Returns:
            ``(N, 3, H, W)`` RGB in ``[-1, 1]``.
        """
        latent = latent / self.vae.config.scaling_factor
        return self.vae.decode(latent, num_frames=num_frames).sample

    @torch.no_grad()
    def clip_embed(self, pil_image: Any) -> Tensor:
        """Compute the SVD-style CLIP image embedding.

        Args:
            pil_image: A PIL image or a list of PIL images.

        Returns:
            ``(N, 1, embed_dim)`` CLIP embedding.
        """
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(
            device=self.image_encoder.device,
            dtype=self.image_encoder.dtype,
        )
        emb = self.image_encoder(pixel_values).image_embeds
        return emb.unsqueeze(1)


def build_svd_backbone(
    *,
    model_id: str | Path = DEFAULT_SVD_MODEL,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
) -> SVDBackbone:
    """Convenience factory.

    Args:
        model_id: HuggingFace id or local path.
        dtype: Loading dtype.
        device: Optional device to move the model to.
        local_files_only: Pass-through to diffusers.

    Returns:
        :class:`SVDBackbone`.
    """
    backbone = SVDBackbone(
        model_id=str(model_id),
        dtype=dtype,
        local_files_only=local_files_only,
    )
    if device is not None:
        backbone = backbone.to(device)
    return backbone
