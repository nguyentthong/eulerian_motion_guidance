"""Tests for :class:`emg.data.webvid.WebVidDataset` against a synthetic manifest.

We build a tiny CSV manifest pointing at one or two synthetic MP4 files
generated on the fly with imageio.  The test verifies:

1.  ``__len__`` matches the number of CSV rows.
2.  ``__getitem__`` returns the expected keys and tensor shapes.
3.  ``collate_webvid`` correctly stacks a small batch.
4.  ``rasterise_hints`` and ``sample_random_trajectories`` round-trip.

If imageio's ffmpeg plugin is not available we mark the relevant test
as skipped — the rest of the dataloader logic is still exercised via
the manifest-reading path.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from emg.data.trajectory_utils import (
    SparseHint,
    rasterise_hints,
    sample_random_trajectories,
)
from emg.data.webvid import WebVidDataset, collate_webvid


def _has_ffmpeg() -> bool:
    """Return ``True`` if we can encode an MP4 with imageio."""
    try:
        import imageio.v3 as iio  # noqa: F401
        import imageio_ffmpeg  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def _write_synthetic_video(path: Path, *, num_frames: int = 16, size: int = 32) -> None:
    """Write a tiny RGB MP4 with random pixel content."""
    import imageio.v3 as iio

    rng = np.random.default_rng(0)
    arr = (rng.uniform(0, 255, (num_frames, size, size, 3))).astype(np.uint8)
    iio.imwrite(path, arr, fps=8, codec="libx264", quality=4)


def _write_manifest(path: Path, video_root: Path, ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["videoid", "name", "contentUrl", "duration", "page_dir"],
        )
        writer.writeheader()
        for vid in ids:
            writer.writerow(
                {
                    "videoid": vid,
                    "name": f"synth-{vid}",
                    "contentUrl": "",
                    "duration": "1.0",
                    "page_dir": "",
                }
            )


@pytest.mark.skipif(not _has_ffmpeg(), reason="imageio-ffmpeg not available")
def test_dataset_loads_and_returns_expected_shapes(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    ids = ["aa", "bb"]
    for vid in ids:
        _write_synthetic_video(video_root / f"{vid}.mp4", num_frames=8, size=32)

    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, video_root, ids)

    ds = WebVidDataset(
        manifest_path=manifest,
        video_root=video_root,
        num_frames=4,
        num_hints=4,
    )
    assert len(ds) == 2

    item = ds[0]
    assert set(item.keys()) >= {"video", "sparse_hints", "videoid", "duration", "name"}
    assert item["video"].shape == (4, 3, 256, 256)  # default transform resizes to 256
    assert item["sparse_hints"].shape == (3, 3, 256, 256)
    assert item["video"].dtype == torch.float32
    assert item["video"].min() >= 0.0 - 1e-5
    assert item["video"].max() <= 1.0 + 1e-5


@pytest.mark.skipif(not _has_ffmpeg(), reason="imageio-ffmpeg not available")
def test_collate_stacks_batch(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    ids = ["xx", "yy"]
    for vid in ids:
        _write_synthetic_video(video_root / f"{vid}.mp4", num_frames=8, size=32)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, video_root, ids)
    ds = WebVidDataset(manifest_path=manifest, video_root=video_root, num_frames=4)

    batch = collate_webvid([ds[0], ds[1]])
    assert isinstance(batch["video"], torch.Tensor)
    assert batch["video"].shape == (2, 4, 3, 256, 256)
    assert batch["sparse_hints"].shape == (2, 3, 3, 256, 256)
    assert isinstance(batch["videoid"], list)
    assert len(batch["videoid"]) == 2


def test_rasterise_hints_roundtrip() -> None:
    """A handful of explicit hints should appear at the correct pixels."""
    hints = [
        [SparseHint(x=2, y=3, u=1.0, v=-2.0), SparseHint(x=4, y=4, u=0.5, v=0.5)],
        [SparseHint(x=0, y=0, u=2.0, v=2.0)],
    ]
    out = rasterise_hints(hints, height=8, width=8)
    assert out.shape == (2, 3, 8, 8)
    # Pair 0
    assert out[0, 0, 3, 2] == pytest.approx(1.0)
    assert out[0, 1, 3, 2] == pytest.approx(-2.0)
    assert out[0, 2, 3, 2] == pytest.approx(1.0)  # mask
    assert out[0, 0, 4, 4] == pytest.approx(0.5)
    # Pair 1
    assert out[1, 0, 0, 0] == pytest.approx(2.0)
    # Empty pixels remain zero
    assert out[0, 0, 0, 0] == 0.0


def test_sample_random_trajectories_shapes() -> None:
    flow = torch.zeros(3, 2, 16, 16)
    flow[0, 0, 4, 4] = 1.5
    hints = sample_random_trajectories(flow, num_points=8, seed=0)
    assert len(hints) == 3
    for pair in hints:
        assert len(pair) == 8
        for h in pair:
            assert 0 <= h.x < 16
            assert 0 <= h.y < 16


def test_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        WebVidDataset(
            manifest_path=tmp_path / "nope.csv",
            video_root=tmp_path,
            num_frames=4,
        )
