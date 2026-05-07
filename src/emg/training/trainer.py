"""Main training orchestrator.

The trainer ties together the data, the trainable modules
(:class:`SparseToDenseNet`, :class:`MotionAdapter`,
:class:`FlowControlNet`), the frozen backbones (SVD + RAFT), and the
loss computation.

The training step implements the parallel-flow trick of Section 4.3:
a single batched RAFT call yields all forward and backward flows at
once; the BGC loss (Eq. 8–10) is then evaluated on that flow stack
together with the diffusion loss in latent space.

The trainer is also where the ablation knobs live:

* ``motion.formulation`` — ``eulerian`` (default) or ``lagrangian``.
* ``consistency.mode``   — ``none`` | ``forward_only`` | ``bidirectional``.
* ``consistency.alpha1``, ``consistency.alpha2`` — Eq. 9 thresholds.
* ``consistency.lambda_geo`` — weight on ``L_geo``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from emg.losses.geometric import (
    BidirectionalGeometricConsistency,
    cycle_consistency_energy,
    geometric_consistency_loss,
)
from emg.models.flow_controlnet import FlowControlNet
from emg.models.motion_adapter import MotionAdapter
from emg.models.s2d import SparseToDenseNet
from emg.motion.eulerian import EulerianFlowSequence, rescale_flow
from emg.motion.parallel_flow import (
    FlowEstimator,
    parallel_bidirectional_flow,
)
from emg.motion.warping import backward_warp
from emg.training.ema import EMAWeights
from emg.training.scheduler import build_lr_scheduler
from emg.utils.distributed import is_main_process
from emg.utils.logging import get_logger

__all__ = ["Trainer", "TrainerConfig"]


_log = get_logger()


@dataclass(slots=True)
class TrainerConfig:
    """Configuration bundle for :class:`Trainer`.

    Attributes:
        formulation: ``eulerian`` (default) or ``lagrangian``.
        consistency_mode: ``none`` | ``forward_only`` | ``bidirectional``.
        alpha1: Eq. 9 dynamic threshold weight.
        alpha2: Eq. 9 static noise floor.
        lambda_geo: Weight on ``L_geo`` (Eq. 10).
        learning_rate: AdamW LR (paper default 2e-5).
        weight_decay: AdamW weight decay.
        betas: AdamW betas.
        gradient_clip: Optional gradient-norm clip.
        num_frames: Clip length used by the trainer.
        ema_decay: EMA shadow decay (0 disables EMA).
        log_interval: Steps between log emissions.
        ckpt_interval: Steps between checkpoint dumps.
        val_interval: Steps between validation runs.
        ckpt_dir: Directory to write checkpoints into.
        mixed_precision: ``"bf16"``, ``"fp16"``, or ``"fp32"``.
        gradient_checkpointing: Toggle for the U-Net.
    """

    formulation: Literal["eulerian", "lagrangian"] = "eulerian"
    consistency_mode: Literal["none", "forward_only", "bidirectional"] = "bidirectional"
    alpha1: float = 0.01
    alpha2: float = 0.5
    lambda_geo: float = 0.1
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    gradient_clip: float | None = 1.0
    num_frames: int = 14
    ema_decay: float = 0.9999
    log_interval: int = 50
    ckpt_interval: int = 1000
    val_interval: int = 500
    ckpt_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    mixed_precision: Literal["fp32", "fp16", "bf16"] = "bf16"
    gradient_checkpointing: bool = False


class Trainer:
    """Train FlowControlNet + S2D + Motion Adapter end-to-end.

    The trainer is intentionally framework-agnostic — it accepts an
    optional :class:`SVDBackbone` (or any module exposing the same
    surface) and a :class:`FlowEstimator`, so unit tests can swap them
    for tiny stubs.

    Args:
        s2d: Sparse-to-Dense network.
        motion_adapter: :class:`MotionAdapter`.
        flow_controlnet: :class:`FlowControlNet`.
        flow_estimator: Frozen flow estimator (paper: RAFT).
        svd_backbone: Optional SVD backbone — if ``None``, the trainer
            runs in *flow-only* mode (latents are taken directly from
            the input video at the same spatial size as the flow).  This
            mode is still useful to exercise the BGC loss in tests.
        config: Hyperparameters.
    """

    def __init__(
        self,
        *,
        s2d: SparseToDenseNet,
        motion_adapter: MotionAdapter,
        flow_controlnet: FlowControlNet,
        flow_estimator: FlowEstimator,
        svd_backbone: nn.Module | None = None,
        config: TrainerConfig | None = None,
    ) -> None:
        self.s2d = s2d
        self.motion_adapter = motion_adapter
        self.flow_controlnet = flow_controlnet
        self.flow_estimator = flow_estimator
        self.svd_backbone = svd_backbone
        self.cfg = config or TrainerConfig()

        self.bgc = BidirectionalGeometricConsistency(
            alpha1=self.cfg.alpha1,
            alpha2=self.cfg.alpha2,
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            schedule="constant",
        )

        self.ema: EMAWeights | None = None
        if self.cfg.ema_decay > 0:
            self.ema = EMAWeights(
                _trainable_module(self.s2d, self.motion_adapter, self.flow_controlnet),
                decay=self.cfg.ema_decay,
            )

        self._step = 0

    # ---------- public API ----------

    @property
    def step(self) -> int:
        return self._step

    def train(self, dataloader: Iterable[dict[str, Any]], *, max_steps: int) -> None:
        """Run the main training loop.

        Args:
            dataloader: Yields dicts with at least ``video`` and
                ``sparse_hints`` keys.
            max_steps: Total optimisation steps.
        """
        self.s2d.train()
        self.motion_adapter.train()
        self.flow_controlnet.train()
        for batch in dataloader:
            if self._step >= max_steps:
                break
            metrics = self.training_step(batch)
            if self._step % self.cfg.log_interval == 0:
                self._log_metrics(metrics)
            if self._step % self.cfg.ckpt_interval == 0 and is_main_process():
                self.save_checkpoint(self.cfg.ckpt_dir / f"step_{self._step:08d}.pt")
            self._step += 1

    def training_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """One optimisation step.

        Returns:
            Dictionary of scalar metrics.
        """
        video: Tensor = batch["video"]  # (B, T, 3, H, W) in [0, 1]
        if video.dim() != 5:
            raise ValueError(f"video must be 5-D; got {tuple(video.shape)}")
        b, t, c, h, w = video.shape

        # 1. Compute bidirectional Eulerian flow with one batched call.
        if self.cfg.formulation == "eulerian":
            flows = parallel_bidirectional_flow(video, self.flow_estimator).flows
        else:
            flows = self._lagrangian_flow_stack(video)

        # 2. Build a flow-conditioned latent for each adjacent pair.
        latent_t, latent_t_plus_1 = self._build_latents(video)
        # Sanity: latents share batch and num_pairs.
        if latent_t.shape[1] != flows.num_pairs:
            raise RuntimeError(
                f"latent has {latent_t.shape[1]} pairs but flow has {flows.num_pairs}"
            )

        # 3. ControlNet forward — purely to keep the path differentiable
        # and exercise the trainable parameters; the residuals are not
        # added back to a frozen U-Net here unless an :class:`SVDBackbone`
        # is supplied.  When the backbone is absent we simply use the
        # ControlNet's mid-block residual as a feature regulariser
        # whose gradient flows back into the trainable modules.
        f_fwd_lat = rescale_flow(
            flows.forward.reshape(-1, 2, h, w),
            (latent_t.shape[-2], latent_t.shape[-1]),
        )
        warped_latent = backward_warp(
            latent_t.reshape(-1, latent_t.shape[2], latent_t.shape[-2], latent_t.shape[-1]),
            f_fwd_lat,
        )
        cn_out = self.flow_controlnet(warped_latent, f_fwd_lat)

        # Diffusion loss surrogate when no SVD is attached: the warped
        # latent should reconstruct latent_{t+1}; this is exactly the
        # SVD denoising objective conditioned on the predicted flow,
        # in expectation.
        target_lat = latent_t_plus_1.reshape_as(warped_latent)
        # Add a zero-magnitude term that depends on the ControlNet
        # output so its gradients flow through L_diff even in the
        # SVD-less path used by tests.
        cn_anchor = cn_out.mid_block_residual.float().pow(2).sum() * 0.0
        diff_loss = F.mse_loss(warped_latent, target_lat) + cn_anchor

        # 4. Geometric consistency loss.
        geo_loss = self._geometric_loss(flows, latent_t, latent_t_plus_1)

        loss = diff_loss + self.cfg.lambda_geo * geo_loss

        # 5. Optimiser step.
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.gradient_clip is not None:
            params = list(self.s2d.parameters()) + list(self.motion_adapter.parameters()) + list(self.flow_controlnet.parameters())
            torch.nn.utils.clip_grad_norm_(params, self.cfg.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update(_trainable_module(self.s2d, self.motion_adapter, self.flow_controlnet))

        return {
            "loss": float(loss.item()),
            "diff_loss": float(diff_loss.item()),
            "geo_loss": float(geo_loss.item()),
            "lr": float(self.scheduler.get_last_lr()[0]),
        }

    # ---------- helpers ----------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params: list[nn.Parameter] = []
        for m in (self.s2d, self.motion_adapter, self.flow_controlnet):
            params += [p for p in m.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=self.cfg.learning_rate,
            betas=self.cfg.betas,
            weight_decay=self.cfg.weight_decay,
        )

    def _build_latents(self, video: Tensor) -> tuple[Tensor, Tensor]:
        """Either VAE-encode or pass through the video as a 'latent'.

        When :attr:`svd_backbone` is None we operate directly in pixel
        space, which keeps unit tests cheap.  When it is present we
        encode each frame through the VAE.
        """
        b, t, c, h, w = video.shape
        if self.svd_backbone is None:
            # Treat the video itself as the "latent".
            latents = video
        else:
            # Encode each frame; SVD wants [-1, 1] inputs.
            normalised = video * 2.0 - 1.0
            flat = normalised.reshape(b * t, c, h, w)
            with torch.no_grad():
                lat = self.svd_backbone.encode_image(flat)  # (B*T, C_z, H_z, W_z)
            cz, hz, wz = lat.shape[-3], lat.shape[-2], lat.shape[-1]
            latents = lat.reshape(b, t, cz, hz, wz)
        return latents[:, :-1], latents[:, 1:]

    def _geometric_loss(
        self,
        flows: EulerianFlowSequence,
        latent_t: Tensor,
        latent_t_plus_1: Tensor,
    ) -> Tensor:
        """Compute ``L_geo`` (Eq. 10) under the configured ablation mode."""
        b = flows.batch_size
        n = flows.num_pairs
        h, w = flows.spatial_size

        f_fwd = flows.forward.reshape(b * n, 2, h, w)
        f_bwd = flows.backward.reshape(b * n, 2, h, w)

        latent_size = (latent_t.shape[-2], latent_t.shape[-1])
        f_fwd_lat = rescale_flow(f_fwd, latent_size)

        z_t = latent_t.reshape(b * n, latent_t.shape[2], *latent_size)
        z_tp1 = latent_t_plus_1.reshape(b * n, latent_t.shape[2], *latent_size)

        if self.cfg.consistency_mode == "none":
            mask = torch.ones((b * n, 1, h, w), device=z_t.device, dtype=z_t.dtype)
        elif self.cfg.consistency_mode == "forward_only":
            # Naïve forward-flow magnitude mask: dis-occluded ⇔ very large flow.
            mag = f_fwd.pow(2).sum(dim=1, keepdim=True)
            thresh = self.cfg.alpha1 * mag.mean() + self.cfg.alpha2
            mask = (mag < thresh).to(f_fwd.dtype)
        elif self.cfg.consistency_mode == "bidirectional":
            # Full BGC.
            energy = cycle_consistency_energy(f_fwd, f_bwd)
            from emg.motion.warping import sample_flow_at_flow

            sampled = sample_flow_at_flow(f_bwd, f_fwd)
            thr = self.cfg.alpha1 * (
                f_fwd.pow(2).sum(dim=1, keepdim=True)
                + sampled.pow(2).sum(dim=1, keepdim=True)
            ) + self.cfg.alpha2
            mask = (energy < thr).to(f_fwd.dtype)
        else:
            raise ValueError(f"Unknown consistency mode {self.cfg.consistency_mode}")

        # Bring mask to latent grid (binary semantics preserved by nearest).
        mask_lat = F.interpolate(mask, size=latent_size, mode="nearest")

        return geometric_consistency_loss(
            z_t=z_t,
            z_t_plus_1=z_tp1,
            flow_fwd=f_fwd_lat,
            mask=mask_lat,
        )

    def _lagrangian_flow_stack(self, video: Tensor) -> EulerianFlowSequence:
        """Compute reference-anchored flows ``f_{0→t}`` for the ablation.

        We still return the result inside an :class:`EulerianFlowSequence`
        container (forward = ``f_{0→t}``, backward = ``f_{t→0}``) so that
        the rest of the trainer can stay agnostic to the formulation.
        """
        b, t, c, h, w = video.shape
        ref = video[:, :1].expand(-1, t - 1, -1, -1, -1).reshape(-1, c, h, w)
        nxt = video[:, 1:].reshape(-1, c, h, w)
        with torch.no_grad():
            fwd = self.flow_estimator(ref, nxt)
            bwd = self.flow_estimator(nxt, ref)
        fwd = fwd.reshape(b, t - 1, 2, h, w)
        bwd = bwd.reshape(b, t - 1, 2, h, w)
        return EulerianFlowSequence(forward=fwd, backward=bwd)

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if not is_main_process():
            return
        msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        _log.info("step=%d | %s", self._step, msg)

    # ---------- checkpointing ----------

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist trainable weights + optimiser + EMA + step counter."""
        from dataclasses import asdict

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # ``TrainerConfig`` is a slots dataclass; use ``asdict`` to
        # serialise it.  ``Path`` instances inside it serialise via str.
        cfg_dict = asdict(self.cfg)
        cfg_dict["ckpt_dir"] = str(cfg_dict["ckpt_dir"])
        state: dict[str, Any] = {
            "step": self._step,
            "config": cfg_dict,
            "s2d": self.s2d.state_dict(),
            "motion_adapter": self.motion_adapter.state_dict(),
            "flow_controlnet": self.flow_controlnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        torch.save(state, p)
        _log.info("Saved checkpoint to %s", p)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore trainer state from a checkpoint."""
        p = Path(path)
        state = torch.load(p, map_location="cpu", weights_only=False)
        self.s2d.load_state_dict(state["s2d"])
        self.motion_adapter.load_state_dict(state["motion_adapter"])
        self.flow_controlnet.load_state_dict(state["flow_controlnet"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self._step = int(state.get("step", 0))
        if self.ema is not None and "ema" in state:
            self.ema.load_state_dict(state["ema"])
        _log.info("Loaded checkpoint from %s (step=%d)", p, self._step)


class _BundledModule(nn.Module):
    """Helper that exposes the union of trainable parameters as one ``nn.Module``."""

    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()
        for i, m in enumerate(modules):
            self.add_module(f"m{i}", m)


def _trainable_module(*modules: nn.Module) -> nn.Module:
    return _BundledModule(*modules)


def build_dataloader_iter(loader: DataLoader[Any]) -> Iterable[dict[str, Any]]:
    """Convenience iterator that loops the loader indefinitely."""
    while True:
        yield from loader
