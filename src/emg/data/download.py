"""Dataset download helpers.

WebVid-10M is **not** publicly hosted any longer (Shutterstock takedown,
late 2024).  This module deliberately does **not** try to download the
videos themselves; instead, it provides utilities that:

* Validate a user-supplied CSV manifest.
* Stream-download the actual MP4s from the URLs *the user supplies*
  (so that the user is responsible for licensing).

For the portrait datasets (VFHQ, CelebV-HQ) we expose the official
download URLs as constants without auto-downloading: portraits are
typically gated behind a request form, and we should not bypass that.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from emg.utils.logging import get_logger

__all__ = [
    "PORTRAIT_DATASET_INFO",
    "stream_webvid_manifest",
    "validate_webvid_manifest",
]


_log = get_logger()

PORTRAIT_DATASET_INFO: dict[str, dict[str, str]] = {
    "VFHQ": {
        "homepage": "https://liangbinxie.github.io/projects/VFHQ/",
        "note": "Request access via the official form.",
    },
    "CelebV-HQ": {
        "homepage": "https://celebv-hq.github.io/",
        "note": "Use the official `celebv-hq` downloader; YouTube IDs are gated by ToS.",
    },
}


def validate_webvid_manifest(manifest_path: str | os.PathLike[str]) -> int:
    """Sanity-check a WebVid manifest.

    Args:
        manifest_path: Filesystem path to a CSV file.

    Returns:
        Number of valid rows.
    """
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(p)
    n_total = 0
    n_valid = 0
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "videoid" not in reader.fieldnames:
            raise ValueError(
                f"Manifest must have a 'videoid' column; got fieldnames={reader.fieldnames}"
            )
        for row in reader:
            n_total += 1
            if row.get("videoid"):
                n_valid += 1
    _log.info("Validated WebVid manifest %s: %d / %d valid rows", p, n_valid, n_total)
    return n_valid


def stream_webvid_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    out_dir: str | os.PathLike[str],
    max_videos: int | None = None,
) -> int:
    """Download videos referenced by ``contentUrl`` in the manifest.

    Args:
        manifest_path: CSV manifest with at least ``videoid`` and
            ``contentUrl`` columns.
        out_dir: Destination directory.
        max_videos: Optional cap on number of videos to download.

    Returns:
        Number of successful downloads.
    """
    import urllib.error
    import urllib.request

    p = Path(manifest_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if max_videos is not None and i >= max_videos:
                break
            vid = row.get("videoid")
            url = row.get("contentUrl")
            if not vid or not url:
                continue
            target = out / f"{vid}.mp4"
            if target.exists():
                n_ok += 1
                continue
            try:
                urllib.request.urlretrieve(url, target)  # noqa: S310 - user-supplied URL
                n_ok += 1
            except (urllib.error.URLError, OSError) as exc:
                _log.warning("Failed to download %s: %s", vid, exc)
                continue
    _log.info("Downloaded %d videos to %s", n_ok, out)
    return n_ok
