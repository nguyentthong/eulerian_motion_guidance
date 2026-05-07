#!/usr/bin/env python3
"""Reproduce one of Tables 1, 2, 3 with the full metric suite.

This script runs inference over a held-out split, then evaluates with
the metrics listed in the resolved config under ``evaluation.metrics``.

Example:

    # Reproduce Table 1 (trajectory-based, WebVid)
    python scripts/evaluate.py \\
        --config configs/trajectory_webvid.yaml \\
        --checkpoint checkpoints/step_00100000.pt \\
        --output eval_outputs/table1.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from emg.data.transforms import build_default_transform
from emg.data.webvid import WebVidDataset, collate_webvid
from emg.evaluation.evaluator import Evaluator, EvaluatorConfig
from emg.inference.autoregressive import AutoregressiveAnimator, InferenceConfig
from emg.models.flow_controlnet import FlowControlNet
from emg.models.motion_adapter import MotionAdapter
from emg.models.s2d import SparseToDenseNet
from emg.utils.config import load_config, merge_configs
from emg.utils.logging import get_logger

_log = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--default-config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("./eval_outputs/report.json"))
    p.add_argument("--method-name", type=str, default="EMG (Ours)")
    p.add_argument("--max-samples", type=int, default=1000, help="Cap on evaluation size")
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    base = load_config(args.default_config) if args.default_config.exists() else OmegaConf.create({})
    user = load_config(args.config)
    cli = OmegaConf.from_dotlist(list(args.overrides))
    cfg = merge_configs(base, user, cli)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset
    dataset_name = str(cfg.data.dataset).lower()
    if dataset_name == "webvid":
        ds = WebVidDataset(
            manifest_path=cfg.data.manifest,
            video_root=cfg.data.video_root,
            num_frames=int(cfg.inference.num_frames),
            transform=build_default_transform(size=int(cfg.training.resolution)),
        )
    elif dataset_name == "portrait":
        from emg.data.portrait import PortraitDataset

        ds = PortraitDataset(  # type: ignore[assignment]
            video_root=cfg.data.video_root,
            num_frames=int(cfg.inference.num_frames),
            transform=build_default_transform(size=int(cfg.training.resolution)),
            landmark_backend=str(cfg.data.get("landmark_backend", "mediapipe")),
            layout=str(cfg.data.get("layout", "celebv_hq")),
        )
    else:
        raise ValueError(f"Unknown dataset {cfg.data.dataset}")

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_webvid if dataset_name == "webvid" else None,
    )

    # Build modules + animator
    block_channels = tuple(int(c) for c in cfg.model.flow_controlnet.block_channels)
    animator = AutoregressiveAnimator.from_checkpoint(
        ckpt_path=str(args.checkpoint),
        s2d=SparseToDenseNet(
            in_channels=int(cfg.model.s2d.in_channels),
            base_channels=int(cfg.model.s2d.base_channels),
            depth=int(cfg.model.s2d.depth),
        ),
        motion_adapter=MotionAdapter(scales=list(block_channels)),
        flow_controlnet=FlowControlNet(
            latent_channels=int(cfg.model.flow_controlnet.latent_channels),
            flow_channels=int(cfg.model.flow_controlnet.flow_channels),
            block_channels=block_channels,
        ),
        config=InferenceConfig(
            num_frames=int(cfg.inference.num_frames),
            window_size=int(cfg.inference.window_size),
            num_inference_steps=int(cfg.inference.num_inference_steps),
            guidance_scale=float(cfg.inference.guidance_scale),
            fps=int(cfg.inference.fps),
            seed=int(cfg.inference.get("seed", 0)),
        ),
        device=device,
    )

    # Run inference + collect predictions/GTs
    preds: list[torch.Tensor] = []
    gts: list[torch.Tensor] = []
    refs: list[torch.Tensor] = []

    n_done = 0
    for batch in loader:
        if n_done >= args.max_samples:
            break
        gt = batch["video"][0]  # (T, 3, H, W)
        ref = gt[0]
        out = animator.animate(ref, trajectories=[[] for _ in range(gt.shape[0] - 1)])
        preds.append(out.frames)
        gts.append(gt)
        refs.append(ref)
        n_done += 1

    pred_tensor = torch.stack(preds, dim=0)
    gt_tensor = torch.stack(gts, dim=0)
    ref_tensor = torch.stack(refs, dim=0)

    # Evaluator
    metrics = tuple(str(m) for m in cfg.evaluation.metrics)
    flow_est = None
    if "e_warp" in metrics:
        from emg.models.raft_wrapper import build_raft_estimator

        flow_est = build_raft_estimator(device=device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(
        config=EvaluatorConfig(metrics=metrics, device=str(device), output_dir=args.output.parent),
        flow_estimator=flow_est,
    )
    results = evaluator.evaluate(pred_tensor, gt_tensor, reference_images=ref_tensor)
    evaluator.save_report(
        results,
        method_name=args.method_name,
        extra_meta={"checkpoint": str(args.checkpoint), "n_samples": int(n_done)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
