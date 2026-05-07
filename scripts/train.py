#!/usr/bin/env python3
"""Main training entry point for EMG.

Runs single-process or DDP (via ``torchrun``) and orchestrates the
:class:`emg.training.trainer.Trainer`.  Heavy backbones (SVD, RAFT) are
loaded lazily so ``--smoke`` mode can run on CPU without weights.

Examples:
    # Single GPU
    python scripts/train.py --config configs/trajectory_webvid.yaml

    # 4-GPU DDP
    torchrun --nproc_per_node 4 scripts/train.py \\
        --config configs/trajectory_webvid.yaml

    # CPU smoke test (uses random data + stub backbones)
    python scripts/train.py --config configs/default.yaml --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from emg.data.transforms import build_default_transform
from emg.data.webvid import WebVidDataset, collate_webvid
from emg.models.flow_controlnet import FlowControlNet
from emg.models.motion_adapter import MotionAdapter
from emg.models.s2d import SparseToDenseNet
from emg.training.trainer import Trainer, TrainerConfig, build_dataloader_iter
from emg.utils.config import load_config, merge_configs
from emg.utils.distributed import is_main_process, setup_distributed
from emg.utils.logging import get_logger

_log = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--default-config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    p.add_argument("--smoke", action="store_true", help="Synthetic CPU smoke test")
    # Trailing key=value overrides; consumed via OmegaConf.from_dotlist.
    p.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. training.batch_size=2")
    return p.parse_args()


def build_dataset(cfg: DictConfig) -> torch.utils.data.Dataset[dict[str, Any]]:
    name = str(cfg.data.get("dataset", "webvid")).lower()
    if name == "webvid":
        return WebVidDataset(
            manifest_path=cfg.data.manifest,
            video_root=cfg.data.video_root,
            num_frames=int(cfg.training.num_frames),
            transform=build_default_transform(size=int(cfg.training.resolution)),
            num_hints=int(cfg.data.get("num_hints", 32)),
        )
    if name == "portrait":
        from emg.data.portrait import PortraitDataset

        return PortraitDataset(
            video_root=cfg.data.video_root,
            num_frames=int(cfg.training.num_frames),
            transform=build_default_transform(size=int(cfg.training.resolution)),
            landmark_backend=str(cfg.data.get("landmark_backend", "mediapipe")),
            layout=str(cfg.data.get("layout", "celebv_hq")),
        )
    raise ValueError(f"Unknown dataset: {cfg.data.dataset}")


class _SyntheticDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """In-memory dummy dataset for smoke tests."""

    def __init__(self, *, num_frames: int, resolution: int, length: int = 16) -> None:
        self.t = num_frames
        self.r = resolution
        self.n = length

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        torch.manual_seed(idx)
        return {
            "video": torch.rand(self.t, 3, self.r, self.r),
            "sparse_hints": torch.zeros(self.t - 1, 3, self.r, self.r),
            "videoid": f"synth_{idx:04d}",
            "duration": 1.0,
            "name": "synthetic",
        }


def build_flow_estimator(*, smoke: bool, device: torch.device) -> Any:
    if smoke:
        # Stub estimator: return zero flow — tests Trainer pathways
        # without downloading weights.
        class ZeroEstimator:
            def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.zeros(a.shape[0], 2, a.shape[-2], a.shape[-1], device=a.device)

        return ZeroEstimator()
    from emg.models.raft_wrapper import build_raft_estimator

    return build_raft_estimator(device=device)


def build_modules(cfg: DictConfig, *, smoke: bool, device: torch.device) -> dict[str, Any]:
    s2d = SparseToDenseNet(
        in_channels=int(cfg.model.s2d.in_channels),
        base_channels=int(cfg.model.s2d.base_channels),
        depth=int(cfg.model.s2d.depth),
    ).to(device)

    block_channels = tuple(int(c) for c in cfg.model.flow_controlnet.block_channels)
    fcn = FlowControlNet(
        latent_channels=int(cfg.model.flow_controlnet.latent_channels),
        flow_channels=int(cfg.model.flow_controlnet.flow_channels),
        block_channels=block_channels,
        svd_unet=None,  # Production wiring is performed inside the trainer.
    ).to(device)
    adapter = MotionAdapter(scales=list(block_channels)).to(device)

    svd = None
    if not smoke:
        try:
            from emg.models.svd_wrapper import build_svd_backbone

            svd = build_svd_backbone(model_id=str(cfg.model.svd_model_id), device=device)
        except Exception as exc:  # pragma: no cover
            _log.warning("Could not load SVD (%s); training in flow-only mode", exc)

    return {
        "s2d": s2d,
        "motion_adapter": adapter,
        "flow_controlnet": fcn,
        "flow_estimator": build_flow_estimator(smoke=smoke, device=device),
        "svd_backbone": svd,
    }


def main() -> int:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    base = load_config(args.default_config) if args.default_config.exists() else OmegaConf.create({})
    user = load_config(args.config)
    cli = OmegaConf.from_dotlist(list(args.overrides))
    cfg = merge_configs(base, user, cli)

    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(cfg.get("seed", 0)) + rank)

    if is_main_process():
        _log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
        _log.info("World size = %d, rank = %d, device = %s", world_size, rank, device)

    # Dataset
    if args.smoke:
        dataset: torch.utils.data.Dataset[dict[str, Any]] = _SyntheticDataset(
            num_frames=int(cfg.training.num_frames),
            resolution=int(cfg.training.resolution),
        )
    else:
        dataset = build_dataset(cfg)

    sampler: torch.utils.data.Sampler[int] | None = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(  # type: ignore[arg-type]
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=int(cfg.training.get("num_workers", 0)),
        collate_fn=collate_webvid,
        pin_memory=torch.cuda.is_available(),
    )

    # Models
    parts = build_modules(cfg, smoke=args.smoke, device=device)

    trainer_cfg = TrainerConfig(
        formulation=str(cfg.motion.formulation),
        consistency_mode=str(cfg.consistency.mode),
        alpha1=float(cfg.consistency.alpha1),
        alpha2=float(cfg.consistency.alpha2),
        lambda_geo=float(cfg.consistency.lambda_geo),
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        betas=tuple(float(b) for b in cfg.training.betas),  # type: ignore[arg-type]
        gradient_clip=float(cfg.training.gradient_clip) if cfg.training.gradient_clip else None,
        num_frames=int(cfg.training.num_frames),
        ema_decay=float(cfg.training.ema_decay),
        log_interval=int(cfg.training.log_interval),
        ckpt_interval=int(cfg.training.ckpt_interval),
        val_interval=int(cfg.training.val_interval),
        ckpt_dir=Path(cfg.training.ckpt_dir),
        mixed_precision=str(cfg.training.mixed_precision),  # type: ignore[arg-type]
        gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
    )

    trainer = Trainer(
        s2d=parts["s2d"],
        motion_adapter=parts["motion_adapter"],
        flow_controlnet=parts["flow_controlnet"],
        flow_estimator=parts["flow_estimator"],
        svd_backbone=parts["svd_backbone"],
        config=trainer_cfg,
    )
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    trainer.train(build_dataloader_iter(loader), max_steps=int(cfg.training.max_steps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
