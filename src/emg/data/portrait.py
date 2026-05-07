"""Portrait video dataset for keypoint-based animation.

Supports two layouts in their public form:

* **VFHQ** — directories of ``{seq}/{frame}.jpg`` plus a JSON metadata
  file.
* **CelebV-HQ** — flat directory of MP4 clips.

Both yield a ``(T, 3, H, W)`` clip plus per-frame facial landmarks (478
points from MediaPipe Face Mesh) when the optional MediaPipe / FaceAlignment
extras are installed.  Landmarks are returned as a tensor of shape
``(T-1, 3, H, W)`` rasterised in the same format as the trajectory
hints (so the rest of the pipeline is uniform).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from emg.data.transforms import VideoTransform, build_default_transform
from emg.data.webvid import _decode_clip
from emg.utils.logging import get_logger

__all__ = ["PortraitDataset"]


_log = get_logger()


@dataclass(slots=True)
class _PortraitItem:
    path: Path
    seq_id: str


class _LandmarkExtractor:
    """Wrap MediaPipe / FaceAlignment landmark extraction with a graceful fallback."""

    def __init__(self, backend: Literal["mediapipe", "face_alignment", "none"] = "mediapipe") -> None:
        self.backend = backend
        self._impl: Any | None = None
        if backend == "mediapipe":
            try:
                import mediapipe as mp  # type: ignore[import-untyped]

                self._impl = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,
                )
            except ImportError:
                _log.warning("MediaPipe not installed; landmarks will be empty.")
                self.backend = "none"
        elif backend == "face_alignment":
            try:
                import face_alignment  # type: ignore[import-untyped]

                self._impl = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D, device="cpu"
                )
            except ImportError:
                _log.warning("face-alignment not installed; landmarks will be empty.")
                self.backend = "none"

    def extract(self, frames: np.ndarray) -> list[np.ndarray]:
        """Extract landmarks from a ``(T, H, W, 3)`` uint8 array.

        Returns:
            ``T`` arrays of shape ``(N, 2)`` with pixel coordinates.
        """
        if self.backend == "none" or self._impl is None:
            return [np.empty((0, 2), dtype=np.float32) for _ in range(frames.shape[0])]
        out: list[np.ndarray] = []
        if self.backend == "mediapipe":
            for fr in frames:
                res = self._impl.process(fr)
                if res.multi_face_landmarks:
                    pts = np.array(
                        [[lm.x * fr.shape[1], lm.y * fr.shape[0]] for lm in res.multi_face_landmarks[0].landmark],
                        dtype=np.float32,
                    )
                else:
                    pts = np.empty((0, 2), dtype=np.float32)
                out.append(pts)
        else:  # face_alignment
            for fr in frames:
                pts = self._impl.get_landmarks(fr)
                out.append(np.asarray(pts[0], dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32))
        return out


def _list_videos(root: Path) -> list[_PortraitItem]:
    items: list[_PortraitItem] = []
    if not root.exists():
        return items
    for p in sorted(root.rglob("*.mp4")):
        items.append(_PortraitItem(path=p, seq_id=p.stem))
    for p in sorted(root.rglob("*.mov")):
        items.append(_PortraitItem(path=p, seq_id=p.stem))
    return items


def _landmark_velocity_to_hints(
    lms_per_frame: list[np.ndarray],
    *,
    height: int,
    width: int,
) -> Tensor:
    """Convert per-frame landmark lists into the (T-1, 3, H, W) hint tensor.

    For each adjacent pair ``(I_t, I_{t+1})`` we take all landmarks
    that are present in both frames and rasterise the per-landmark
    velocity at its position in ``I_t``.
    """
    n_pairs = len(lms_per_frame) - 1
    hints = torch.zeros((n_pairs, 3, height, width), dtype=torch.float32)
    for t in range(n_pairs):
        a = lms_per_frame[t]
        b = lms_per_frame[t + 1]
        n = min(len(a), len(b))
        if n == 0:
            continue
        a, b = a[:n], b[:n]
        for i in range(n):
            xa, ya = int(a[i, 0]), int(a[i, 1])
            if not (0 <= xa < width and 0 <= ya < height):
                continue
            u = float(b[i, 0] - a[i, 0])
            v = float(b[i, 1] - a[i, 1])
            hints[t, 0, ya, xa] = u
            hints[t, 1, ya, xa] = v
            hints[t, 2, ya, xa] = 1.0
    return hints


class PortraitDataset(Dataset[dict[str, Any]]):
    """Portrait video dataset with optional landmark hints.

    Args:
        video_root: Directory holding portrait videos.
        num_frames: Clip length.
        stride: Frame stride.
        transform: Video transform (defaults to 256-square ``[0, 1]``).
        seed: RNG seed used for clip offsets.
        landmark_backend: Which landmark extractor to use, or
            ``"none"`` to skip landmarks (hints will be all zeros).
    """

    def __init__(
        self,
        *,
        video_root: str | Path,
        num_frames: int = 14,
        stride: int = 1,
        transform: VideoTransform | None = None,
        seed: int = 0,
        landmark_backend: Literal["mediapipe", "face_alignment", "none"] = "mediapipe",
    ) -> None:
        self.items = _list_videos(Path(video_root))
        self.num_frames = int(num_frames)
        self.stride = int(stride)
        self.transform = transform or build_default_transform()
        self.seed = int(seed)
        self.extractor = _LandmarkExtractor(backend=landmark_backend)
        if not self.items:
            _log.warning("No portrait videos found under %s", video_root)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        attempts = 0
        while True:
            arr = _decode_clip(
                item.path,
                num_frames=self.num_frames,
                stride=self.stride,
                seed=self.seed + idx,
            )
            if arr is not None and arr.shape[0] == self.num_frames:
                break
            attempts += 1
            if attempts > 8:
                raise RuntimeError(f"Failed to decode portrait video {item.path}")
            idx = (idx + 1) % len(self.items)

        clip = self.transform(arr)  # (T, 3, H, W)
        h, w = clip.shape[-2:]
        # Resize landmarks to the transform output resolution.
        lms = self.extractor.extract(arr)
        if lms and lms[0].size > 0:
            scale_x = w / arr.shape[2]
            scale_y = h / arr.shape[1]
            lms = [pts * np.array([scale_x, scale_y], dtype=np.float32) for pts in lms]
        hints = _landmark_velocity_to_hints(lms, height=h, width=w)

        return {
            "video": clip,
            "sparse_hints": hints,
            "seq_id": item.seq_id,
        }
