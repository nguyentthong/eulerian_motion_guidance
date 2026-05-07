#!/usr/bin/env python3
"""Validate (and optionally summarise) a WebVid-10M CSV manifest.

We do *not* download videos — the original WebVid-10M URLs are no
longer publicly hosted.  Use this script after acquiring videos through
your own legitimate means; it walks the manifest and reports how many
records resolve to local files under ``--video-root``.

Example:

    python scripts/download_webvid.py \\
        --manifest /data/webvid/manifest.csv \\
        --video-root /data/webvid/videos \\
        --json-out /data/webvid/manifest_health.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emg.data.download import stream_webvid_manifest, validate_webvid_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path, help="CSV manifest path")
    p.add_argument("--video-root", required=True, type=Path, help="Root directory for video files")
    p.add_argument("--json-out", type=Path, default=None, help="Optional health report JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    n = validate_webvid_manifest(args.manifest)
    print(f"manifest contains {n} rows")

    found = 0
    missing: list[str] = []
    for rec in stream_webvid_manifest(args.manifest):
        path = args.video_root / (rec.page_dir or "") / f"{rec.videoid}.mp4"
        if path.exists():
            found += 1
        else:
            missing.append(rec.videoid)

    print(f"resolved {found}/{n} videos under {args.video_root}")
    if missing:
        print(f"first few missing videoids: {missing[:5]}")

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "video_root": str(args.video_root),
                    "n_rows": n,
                    "n_resolved": found,
                    "n_missing": len(missing),
                    "missing_sample": missing[:50],
                },
                indent=2,
            )
        )
        print(f"wrote health report to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
