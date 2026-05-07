#!/usr/bin/env python3
"""Animate a single image given sparse trajectories.

Trajectory file format (JSON):

    {
      "trajectories": [
        [{"x": 64, "y": 64, "u": 1.5, "v": 0.0},
         {"x": 192, "y": 96, "u": -1.0, "v": 0.5}],
        [...]   # one list per (T-1) adjacent pair
      ]
    }

If only a single list is provided we replicate it across all adjacent
pairs (i.e. constant motion).

Example:

    python scripts/animate.py \\
        --image inputs/lighthouse.png \\
        --trajectories inputs/lighthouse_traj.json \\
        --checkpoint checkpoints/step_00100000.pt \\
        --config configs/default.yaml \\
        --output outputs/lighthouse.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from emg.data.trajectory_utils import SparseHint
from emg.inference.autoregressive import AutoregressiveAnimator, InferenceConfig
from emg.models.flow_controlnet import FlowControlNet
from emg.models.motion_adapter import MotionAdapter
from emg.models.s2d import SparseToDenseNet
from emg.utils.config import load_config, merge_configs
from emg.utils.logging import get_logger
from emg.utils.visualization import save_video

_log = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--trajectories", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--default-config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--output", type=Path, default=Path("./outputs/animation.mp4"))
    p.add_argument("--num-frames", type=int, default=None, help="Override config.inference.num_frames")
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


def load_image(path: Path, *, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_trajectories(path: Path, *, n_pairs: int) -> list[list[SparseHint]]:
    raw = json.loads(path.read_text())
    payload = raw.get("trajectories", raw)
    if not isinstance(payload, list):
        raise ValueError(f"Bad trajectories file {path}: expected list at top level")
    if payload and isinstance(payload[0], dict):
        # Single list — replicate across all pairs.
        single = [SparseHint(**p) for p in payload]
        return [list(single) for _ in range(n_pairs)]
    out: list[list[SparseHint]] = []
    for entry in payload:
        out.append([SparseHint(**p) for p in entry])
    return out


def main() -> int:
    args = parse_args()

    base = load_config(args.default_config) if args.default_config.exists() else OmegaConf.create({})
    user = load_config(args.config)
    cli = OmegaConf.from_dotlist(list(args.overrides))
    cfg = merge_configs(base, user, cli)

    if args.num_frames is not None:
        cfg.inference.num_frames = int(args.num_frames)

    res = int(cfg.training.resolution)
    image = load_image(args.image, size=res)
    trajectories = load_trajectories(
        args.trajectories, n_pairs=int(cfg.inference.num_frames) - 1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            fps=int(cfg.inference.fps),
            seed=int(cfg.inference.get("seed", 0)),
        ),
        device=device,
    )

    out = animator.animate(image, trajectories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_video(out.frames, args.output, fps=int(cfg.inference.fps))
    _log.info("wrote %s (T=%d)", args.output, out.frames.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
