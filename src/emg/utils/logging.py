"""Rich-based console logging with rank-zero gating.

Distributed training tends to produce heavily duplicated logs because
every rank writes the same message.  We expose a tiny wrapper that
logs only on rank 0 unless the caller opts in to per-rank logging.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from rich.logging import RichHandler

__all__ = ["RankZeroLogger", "get_logger"]


def _detect_rank() -> int:
    """Detect the process rank in a way that works without torch.distributed initialised."""
    for var in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
        v = os.environ.get(var)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                continue
    return 0


_FORMAT = "%(message)s"


class RankZeroLogger:
    """Logger that filters by rank.

    On rank 0 every level is forwarded to the underlying
    :class:`logging.Logger`; on other ranks only ERROR and above pass
    through unless ``all_ranks=True`` is set on the call.
    """

    def __init__(self, name: str = "emg") -> None:
        self._logger = logging.getLogger(name)
        self._rank = _detect_rank()
        self._configure()

    def _configure(self) -> None:
        if self._logger.handlers:
            return
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        handler.setFormatter(logging.Formatter(_FORMAT))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    @property
    def rank(self) -> int:
        return self._rank

    def info(self, msg: str, *args: Any, all_ranks: bool = False) -> None:
        if all_ranks or self._rank == 0:
            self._logger.info(msg, *args)

    def warning(self, msg: str, *args: Any, all_ranks: bool = False) -> None:
        if all_ranks or self._rank == 0:
            self._logger.warning(msg, *args)

    def error(self, msg: str, *args: Any) -> None:
        # Errors print on every rank.
        self._logger.error("[rank %d] " + msg, self._rank, *args)

    def debug(self, msg: str, *args: Any, all_ranks: bool = False) -> None:
        if all_ranks or self._rank == 0:
            self._logger.debug(msg, *args)


_DEFAULT_LOGGER: RankZeroLogger | None = None


def get_logger(name: str = "emg") -> RankZeroLogger:
    """Return the singleton rank-zero logger.

    Args:
        name: The logger name (rarely needed; we keep a single global
            logger).

    Returns:
        A :class:`RankZeroLogger`.
    """
    global _DEFAULT_LOGGER
    if _DEFAULT_LOGGER is None:
        _DEFAULT_LOGGER = RankZeroLogger(name)
    return _DEFAULT_LOGGER


def _stderr_print(msg: str) -> None:  # pragma: no cover - convenience
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
