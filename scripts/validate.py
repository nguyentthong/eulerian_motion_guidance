#!/usr/bin/env python3
"""Lightweight validation pass during training.

Runs a small subset of metrics (LPIPS, CLIP-Cons, E_warp) on a held-out
split.  Intended to be invoked every ``training.val_interval`` steps
from :class:`emg.training.trainer.Trainer`.

Example:
    python scripts/validate.py \\
        --config configs/trajectory_webvid.yaml \\
        --checkpoint checkpoints/step_00010000.pt \\
        --num-samples 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from emg.evaluation.evaluator import Evaluator, EvaluatorConfig
from emg.inference.autoregressive import (
    AutoregressiveAnimator,
    InferenceConfig,
)
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
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--output-dir", type=Path, default=Path("./val_outputs"))
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    base = load_config(args.default_config) if args.default_config.exists() else OmegaConf.create({})
    user = load_config(args.config)
    cli = OmegaConf.from_dotlist(list(args.overrides))
    cfg = merge_configs(base, user, cli)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block_channels = tuple(int(c) for c in cfg.model.flow_controlnet.block_channels)
    s2d = SparseToDenseNet(
        in_channels=int(cfg.model.s2d.in_channels),
        base_channels=int(cfg.model.s2d.base_channels),
        depth=int(cfg.model.s2d.depth),
    )
    adapter = MotionAdapter(scales=list(block_channels))
    fcn = FlowControlNet(
        latent_channels=int(cfg.model.flow_controlnet.latent_channels),
        flow_channels=int(cfg.model.flow_controlnet.flow_channels),
        block_channels=block_channels,
    )

    inf_cfg = InferenceConfig(
        num_frames=int(cfg.inference.num_frames),
        window_size=int(cfg.inference.window_size),
        num_inference_steps=int(cfg.inference.num_inference_steps),
        guidance_scale=float(cfg.inference.guidance_scale),
        fps=int(cfg.inference.fps),
        seed=int(cfg.inference.get("seed", 0)),
    )

    animator = AutoregressiveAnimator.from_checkpoint(
        ckpt_path=str(args.checkpoint),
        s2d=s2d,
        motion_adapter=adapter,
        flow_controlnet=fcn,
        config=inf_cfg,
        device=device,
    )

    # We synthesise a small validation batch from random noise — the
    # full training script's validation hook will substitute real data.
    n = args.num_samples
    res = int(cfg.training.resolution)
    t = int(cfg.inference.num_frames)
    preds = []
    gts = torch.rand(n, t, 3, res, res)
    for i in range(n):
        ref = gts[i, 0]
        out = animator.animate(ref, trajectories=[[] for _ in range(t - 1)])
        preds.append(out.frames)
    pred_tensor = torch.stack(preds, dim=0)

    flow_estimator = None
    try:
        from emg.models.raft_wrapper import build_raft_estimator

        flow_estimator = build_raft_estimator(device=device)
    except Exception as exc:  # pragma: no cover
        _log.warning("Could not load RAFT (%s); skipping E_warp", exc)

    evaluator = Evaluator(
        config=EvaluatorConfig(
            metrics=("lpips", "clip_cons", "e_warp") if flow_estimator else ("lpips", "clip_cons"),
            device=str(device),
            output_dir=args.output_dir,
        ),
        flow_estimator=flow_estimator,
    )
    results = evaluator.evaluate(pred_tensor, gts)
    evaluator.save_report(results, method_name="EMG-val", extra_meta={"checkpoint": str(args.checkpoint)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
