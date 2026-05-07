"""WebVid-10M dataset loader.

The original WebVid-10M release has been taken down by Shutterstock
(2024).  We do not pretend a download URL exists.  Instead, the loader
operates against a *user-supplied* CSV manifest with the canonical
schema:

  videoid,name,contentUrl,duration,page_dir

and a local directory of MP4 files named ``{videoid}.mp4`` (or grouped
under ``{page_dir}/{videoid}.mp4``).  Each item in the dataset returns a
``(T, 3, H, W)`` clip plus a sparse-hint tensor and metadata.

The loader is intentionally **resilient** to missing files: any video
that fails to decode is skipped at iteration time and a warning is
logged.  This matches the expectations of streaming WebVid pipelines.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from emg.data.transforms import VideoTransform, build_default_transform
from emg.utils.logging import get_logger

__all__ = ["WebVidDataset"]


_log = get_logger()


@dataclass(slots=True)
class _VideoRecord:
    videoid: str
    page_dir: str | None
    duration: float | None
    name: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_path(self, root: Path) -> Path:
        """Look up ``{root}/{page_dir}/{videoid}.mp4`` or ``{root}/{videoid}.mp4``."""
        if self.page_dir:
            cand = root / self.page_dir / f"{self.videoid}.mp4"
            if cand.exists():
                return cand
        return root / f"{self.videoid}.mp4"


def _read_manifest(manifest_path: Path) -> list[_VideoRecord]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"WebVid manifest not found: {manifest_path}")
    out: list[_VideoRecord] = []
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            vid = row.get("videoid") or row.get("video_id")
            if not vid:
                continue
            duration_raw = row.get("duration")
            try:
                duration = float(duration_raw) if duration_raw else None
            except ValueError:
                duration = None
            out.append(
                _VideoRecord(
                    videoid=str(vid).strip(),
                    page_dir=(row.get("page_dir") or None),
                    duration=duration,
                    name=(row.get("name") or None),
                    extra={k: v for k, v in row.items() if k not in {"videoid", "video_id", "page_dir", "duration", "name"}},
                )
            )
    return out


def _decode_clip(
    path: Path,
    *,
    num_frames: int,
    stride: int = 1,
    seed: int | None = None,
) -> np.ndarray | None:
    """Decode ``num_frames`` from the video at ``path`` with stride ``stride``.

    Args:
        path: Filesystem path to an MP4 (or any container readable by
            :mod:`av`).
        num_frames: Number of frames to return.
        stride: Frame stride.
        seed: Optional RNG seed for the random temporal offset.

    Returns:
        ``(num_frames, H, W, 3)`` ``uint8`` array, or ``None`` if the
        video cannot be decoded.
    """
    try:
        import av  # type: ignore[import-untyped]
    except ImportError:
        try:
            import imageio.v3 as iio
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Either pyav or imageio[ffmpeg] is required") from exc
        try:
            arr = iio.imread(path, plugin="pyav")
        except Exception:  # pragma: no cover
            return None
        if arr.ndim != 4:
            return None
        total = arr.shape[0]
        rng = np.random.default_rng(seed)
        max_start = max(0, total - num_frames * stride)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        idx = start + np.arange(num_frames) * stride
        idx = np.clip(idx, 0, total - 1)
        return arr[idx]

    try:
        container = av.open(str(path))
    except Exception:
        return None
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= 1024:
                break
        container.close()
    except Exception:
        return None
    if len(frames) == 0:
        return None
    arr = np.stack(frames, axis=0)
    total = arr.shape[0]
    rng = np.random.default_rng(seed)
    max_start = max(0, total - num_frames * stride)
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    idx = start + np.arange(num_frames) * stride
    idx = np.clip(idx, 0, total - 1)
    return arr[idx]


class WebVidDataset(Dataset[dict[str, Any]]):
    """Iterable WebVid-10M dataset reading from a CSV manifest.

    Args:
        manifest_path: Path to a CSV manifest with columns at least
            ``videoid``.  If ``page_dir`` is present it is used to
            organise the video tree.
        video_root: Directory containing the actual videos.
        num_frames: Clip length ``T``.
        stride: Frame stride for sub-sampling.
        transform: Optional transform.  Defaults to
            :func:`build_default_transform`.
        num_hints: Number of sparse hints to emit per adjacent pair.
        seed: RNG seed used for hint sampling and clip offsets.
    """

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        video_root: str | Path,
        num_frames: int = 14,
        stride: int = 1,
        transform: VideoTransform | None = None,
        num_hints: int = 32,
        seed: int = 0,
    ) -> None:
        self.records = _read_manifest(Path(manifest_path))
        self.video_root = Path(video_root)
        self.num_frames = int(num_frames)
        self.stride = int(stride)
        self.transform = transform or build_default_transform()
        self.num_hints = int(num_hints)
        self.seed = int(seed)

        if not self.records:
            _log.warning("Empty WebVid manifest at %s", manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def _decode(self, idx: int) -> np.ndarray | None:
        rec = self.records[idx]
        path = rec.resolve_path(self.video_root)
        return _decode_clip(
            path,
            num_frames=self.num_frames,
            stride=self.stride,
            seed=self.seed + idx,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        attempts = 0
        while True:
            arr = self._decode(idx)
            if arr is not None and arr.shape[0] == self.num_frames:
                break
            attempts += 1
            if attempts > 8:
                raise RuntimeError(
                    f"Failed to decode any video for index {idx} after 8 attempts"
                )
            idx = (idx + 1) % len(self.records)

        clip = self.transform(arr)  # (T, 3, H, W) float
        # Sparse hints (channel ``mask`` is zero by default; user can
        # populate them via trajectory_utils).  We pre-allocate the
        # tensor here so collation is straightforward.
        hint = torch.zeros(
            (self.num_frames - 1, 3, clip.shape[-2], clip.shape[-1]),
            dtype=torch.float32,
        )
        return {
            "video": clip,
            "sparse_hints": hint,
            "videoid": rec.videoid,
            "duration": rec.duration if rec.duration is not None else -1.0,
            "name": rec.name or "",
        }


def collate_webvid(batch: list[dict[str, Any]]) -> dict[str, Tensor | list[Any]]:
    """Default collate for :class:`WebVidDataset`.

    Stacks ``video`` and ``sparse_hints`` and lists the metadata.
    """
    out: dict[str, Tensor | list[Any]] = {
        "video": torch.stack([b["video"] for b in batch], dim=0),
        "sparse_hints": torch.stack([b["sparse_hints"] for b in batch], dim=0),
        "videoid": [b["videoid"] for b in batch],
        "duration": [b["duration"] for b in batch],
        "name": [b["name"] for b in batch],
    }
    return out
