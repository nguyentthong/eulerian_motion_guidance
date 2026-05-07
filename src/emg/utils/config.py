"""Configuration loading utilities built on :mod:`omegaconf`.

The training and evaluation scripts use a YAML-driven configuration
system.  This module provides the (very thin) wrappers around OmegaConf
the rest of the codebase imports.

A canonical config is structured as:

.. code-block:: yaml

    seed: 42
    motion:
        formulation: eulerian      # or 'lagrangian'
    consistency:
        mode: bidirectional        # 'none' | 'forward_only' | 'bidirectional'
        alpha1: 0.01
        alpha2: 0.5
        lambda_geo: 0.1
    training:
        batch_size: 1
        num_frames: 14
        learning_rate: 2.0e-5
        ...
    data:
        dataset: webvid
        manifest: /path/to/manifest.csv
        video_root: /path/to/videos
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

__all__ = ["load_config", "merge_configs", "save_config"]


def load_config(path: str | os.PathLike[str]) -> DictConfig:
    """Load a YAML config and resolve all interpolations.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Resolved :class:`DictConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Config root must be a mapping; got {type(cfg).__name__}")
    OmegaConf.resolve(cfg)
    return cfg


def merge_configs(*configs: DictConfig | dict[str, Any]) -> DictConfig:
    """Merge multiple configs left-to-right.

    Later entries override earlier ones (standard OmegaConf semantics).

    Args:
        configs: Any mix of :class:`DictConfig` or plain dicts.

    Returns:
        The merged :class:`DictConfig`.
    """
    if not configs:
        return OmegaConf.create({})
    merged = OmegaConf.merge(*configs)
    if not isinstance(merged, DictConfig):
        raise TypeError("Merged config must be a mapping")
    return merged


def save_config(cfg: DictConfig, path: str | os.PathLike[str]) -> None:
    """Persist a config to YAML.

    Args:
        cfg: The config to dump.
        path: Output filesystem path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out)
