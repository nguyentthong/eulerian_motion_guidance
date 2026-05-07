"""Shared pytest fixtures and utilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``import emg.*`` without installing the package.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

torch.manual_seed(1234)
