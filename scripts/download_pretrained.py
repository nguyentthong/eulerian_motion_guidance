#!/usr/bin/env python3
"""Download the frozen backbones used by EMG.

* Stable Video Diffusion XT — pulled via :mod:`huggingface_hub`.  Set
  ``HF_TOKEN`` if access is gated.
* RAFT-Large (FlyingThings3D weights) — pulled via
  :mod:`torchvision.models.optical_flow`.

Example:

    HF_TOKEN=... python scripts/download_pretrained.py \\
        --cache-dir ~/.cache/huggingface/hub
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from emg.utils.logging import get_logger

_log = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--svd-id",
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        help="HuggingFace model id for SVD",
    )
    p.add_argument("--cache-dir", type=Path, default=None, help="HF cache directory")
    p.add_argument(
        "--skip-svd", action="store_true", help="Skip SVD (useful if already cached)"
    )
    p.add_argument("--skip-raft", action="store_true", help="Skip RAFT")
    return p.parse_args()


def fetch_svd(model_id: str, cache_dir: Path | None) -> None:
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    kwargs: dict[str, object] = {"repo_id": model_id}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if "HF_TOKEN" in os.environ:
        kwargs["token"] = os.environ["HF_TOKEN"]
    _log.info("Downloading SVD weights for %s ...", model_id)
    path = snapshot_download(**kwargs)  # type: ignore[arg-type]
    _log.info("SVD cached at %s", path)


def fetch_raft() -> None:
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    _log.info("Downloading RAFT-Large (Things) weights ...")
    raft_large(weights=Raft_Large_Weights.C_T_V1, progress=True)
    _log.info("RAFT cached.")


def main() -> int:
    args = parse_args()
    if not args.skip_svd:
        fetch_svd(args.svd_id, args.cache_dir)
    if not args.skip_raft:
        fetch_raft()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
