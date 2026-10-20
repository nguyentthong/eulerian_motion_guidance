"""Tests for :class:`emg.training.trainer.Trainer` in flow-only mode.

Without an SVD backbone the trainer treats RGB frames as latents, so the
FlowControlNet must be built with 3 latent channels.  These tests run a
few optimisation steps with tiny modules and a zero-flow stub estimator.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from emg.models.flow_controlnet import FlowControlNet
from emg.models.motion_adapter import MotionAdapter
from emg.models.s2d import SparseToDenseNet
from emg.training.trainer import Trainer, TrainerConfig

_BLOCKS = (8, 16)


class _ZeroFlow:
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.zeros(a.shape[0], 2, a.shape[-2], a.shape[-1])


def _make_trainer(tmp_path: Path, **overrides: object) -> Trainer:
    cfg = TrainerConfig(ckpt_dir=tmp_path, ema_decay=0.9, **overrides)  # type: ignore[arg-type]
    return Trainer(
        s2d=SparseToDenseNet(base_channels=8, depth=2),
        motion_adapter=MotionAdapter(scales=list(_BLOCKS)),
        flow_controlnet=FlowControlNet(latent_channels=3, block_channels=_BLOCKS),
        flow_estimator=_ZeroFlow(),
        config=cfg,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {"video": torch.rand(1, 4, 3, 16, 16)}


@pytest.mark.parametrize("mode", ["none", "forward_only", "bidirectional"])
@pytest.mark.parametrize("formulation", ["eulerian", "lagrangian"])
def test_training_step_returns_finite_metrics(tmp_path: Path, mode: str, formulation: str) -> None:
    trainer = _make_trainer(tmp_path, consistency_mode=mode, formulation=formulation)
    metrics = trainer.training_step(_batch())
    assert set(metrics) == {"loss", "diff_loss", "geo_loss", "lr"}
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_cosine_schedule_decays_learning_rate(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, lr_schedule="cosine", total_steps=4)
    lrs = [trainer.training_step(_batch())["lr"] for _ in range(3)]
    assert lrs[0] > lrs[1] > lrs[2]


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.train(iter([_batch(), _batch()]), max_steps=2)
    ckpt = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt)

    restored = _make_trainer(tmp_path)
    restored.load_checkpoint(ckpt)
    assert restored.step == trainer.step == 2
    for a, b in zip(
        trainer.flow_controlnet.state_dict().values(),
        restored.flow_controlnet.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(a, b)


def test_train_script_smoke_mode(tmp_path: Path) -> None:
    """``scripts/train.py --smoke`` must run end to end without SVD weights."""
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "train.py"),
            "--config",
            str(root / "configs" / "default.yaml"),
            "--default-config",
            str(root / "configs" / "default.yaml"),
            "--smoke",
            "training.max_steps=2",
            "training.num_frames=3",
            "training.resolution=16",
            "training.num_workers=0",
            f"training.ckpt_dir={tmp_path}",
            "model.s2d.base_channels=8",
            "model.s2d.depth=2",
            "model.flow_controlnet.block_channels=[8,16]",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        check=False,
    )
    assert result.returncode == 0, result.stderr
