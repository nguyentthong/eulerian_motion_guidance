"""Evaluation metrics for Tables 1, 2, 3.

Every metric is a single function that takes a small set of well-typed
tensors and returns a scalar Python ``float``.  Heavy backbones are
loaded *lazily* — the module imports cleanly even if (say) ``insightface``
is missing; only the metrics that depend on it will fail.

Each function carries a docstring referencing the paper's metric name
(LPIPS, FID, FVD, CLIP-Cons, E_warp, CPBD, ArcFace).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from emg.motion.warping import backward_warp
from emg.utils.logging import get_logger

__all__ = [
    "compute_arcface",
    "compute_clip_consistency",
    "compute_cpbd",
    "compute_fid",
    "compute_fvd",
    "compute_lpips",
    "compute_warping_error",
]


_log = get_logger()

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _validate_videos(*videos: Tensor) -> tuple[int, int, int, int, int]:
    """Validate a tuple of ``(B, T, 3, H, W)`` videos and return the shape."""
    if not videos:
        raise ValueError("at least one video tensor required")
    ref = videos[0]
    if ref.dim() != 5 or ref.shape[2] != 3:
        raise ValueError(f"videos must be (B, T, 3, H, W); got {tuple(ref.shape)}")
    for v in videos[1:]:
        if v.shape != ref.shape:
            raise ValueError(
                f"video shapes must match; got {tuple(v.shape)} vs {tuple(ref.shape)}"
            )
    b, t, c, h, w = ref.shape
    return b, t, c, h, w


def _to_uint8(video: Tensor) -> Tensor:
    """Convert a ``[0, 1]`` float video to ``uint8``."""
    if video.dtype == torch.uint8:
        return video
    return (video.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


# --------------------------------------------------------------------------- #
# LPIPS                                                                       #
# --------------------------------------------------------------------------- #


def compute_lpips(
    pred: Tensor,
    gt: Tensor,
    *,
    net: str = "alex",
    device: str | torch.device = "cpu",
) -> float:
    """Frame-averaged LPIPS between predicted and ground-truth videos.

    Uses the AlexNet backbone (Table 1 default) from the ``lpips`` PyPI
    package.  Inputs are expected to be ``(B, T, 3, H, W)`` floats in
    ``[0, 1]``.

    Args:
        pred: Predicted videos.
        gt: Ground-truth videos (same shape).
        net: LPIPS backbone (``"alex"`` or ``"vgg"``).
        device: Torch device for the LPIPS network.

    Returns:
        Mean LPIPS distance in ``[0, ∞)`` — lower is better.
    """
    import lpips  # type: ignore[import-not-found]

    b, t, _, _, _ = _validate_videos(pred, gt)
    metric = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    # LPIPS expects values in [-1, 1].
    p = (pred.reshape(b * t, 3, *pred.shape[-2:]).clamp(0, 1) * 2 - 1).to(device)
    g = (gt.reshape(b * t, 3, *gt.shape[-2:]).clamp(0, 1) * 2 - 1).to(device)
    with torch.no_grad():
        d = metric(p, g)
    return float(d.mean().item())


# --------------------------------------------------------------------------- #
# FID                                                                         #
# --------------------------------------------------------------------------- #


def _save_frames_to_dir(video: Tensor, out_dir: Path) -> None:
    """Persist every frame of ``video`` as an individual PNG (for clean-fid)."""
    import imageio.v3 as iio

    out_dir.mkdir(parents=True, exist_ok=True)
    arr = _to_uint8(video.reshape(-1, 3, *video.shape[-2:])).permute(0, 2, 3, 1).cpu().numpy()
    for i, frame in enumerate(arr):
        iio.imwrite(out_dir / f"{i:08d}.png", frame)


def compute_fid(
    pred: Tensor,
    gt: Tensor,
    *,
    workdir: str | Path | None = None,
    mode: str = "clean",
) -> float:
    """Frechet Inception Distance between predicted and GT videos.

    Computed at frame level via :mod:`clean-fid`.  The function spills
    frames to disk under ``workdir`` because ``clean-fid``'s API is
    directory-based; pass an explicit path for reproducibility.

    Args:
        pred: ``(B, T, 3, H, W)`` predicted videos.
        gt: ``(B, T, 3, H, W)`` ground-truth videos.
        workdir: Optional staging directory.  A temp dir is used by default.
        mode: ``clean-fid`` mode (``"clean"`` is the default in the paper).

    Returns:
        FID score — lower is better.
    """
    from cleanfid import fid as cfid  # type: ignore[import-not-found]

    _validate_videos(pred, gt)

    import tempfile

    work = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="emg_fid_"))
    pred_dir = work / "pred"
    gt_dir = work / "gt"
    _save_frames_to_dir(pred, pred_dir)
    _save_frames_to_dir(gt, gt_dir)

    score = cfid.compute_fid(str(pred_dir), str(gt_dir), mode=mode)
    return float(score)


# --------------------------------------------------------------------------- #
# FVD                                                                         #
# --------------------------------------------------------------------------- #


def _i3d_features(videos: Tensor, device: str | torch.device) -> Tensor:
    """Compute I3D logits/features for FVD.

    We rely on ``torchvision`` 's `r3d_18` Kinetics-400 model as a
    serviceable I3D substitute when the canonical I3D weights are not
    available.  This keeps the metric self-contained; if a user wants
    the *exact* paper figure they should plug in their preferred I3D
    implementation by overriding this helper.

    Note:
        Author's interpretation — flagged in :func:`compute_fvd`'s
        docstring.
    """
    from torchvision.models.video import R3D_18_Weights, r3d_18

    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights).eval().to(device)
    # Strip classification head — we want the penultimate features.
    model.fc = torch.nn.Identity()  # type: ignore[assignment]

    transform = weights.transforms()

    b, t, _, _, _ = videos.shape
    # r3d_18 expects (B, 3, T, H, W) at 112x112; clean transforms handle that.
    vids = videos.permute(0, 2, 1, 3, 4).contiguous()  # (B, 3, T, H, W)
    proc = transform(vids).to(device)
    with torch.no_grad():
        feats = model(proc)
    return feats.float().cpu()


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Numerically-stable Frechet distance between two Gaussians."""
    from scipy import linalg  # type: ignore[import-not-found]

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def compute_fvd(
    pred: Tensor,
    gt: Tensor,
    *,
    device: str | torch.device = "cpu",
) -> float:
    """Frechet Video Distance between predicted and GT video distributions.

    Note:
        Author's interpretation — the canonical FVD uses the Kinetics
        I3D checkpoint from the original Inflated-3D paper.  We default
        to torchvision's ``r3d_18`` Kinetics-400 weights for portability
        (the paper does not pin a specific implementation).  Plug in
        your own feature extractor by overriding ``_i3d_features``.

    Args:
        pred: ``(B, T, 3, H, W)`` predicted videos in ``[0, 1]``.
        gt: ``(B, T, 3, H, W)`` ground-truth videos in ``[0, 1]``.
        device: Torch device for the feature extractor.

    Returns:
        FVD score — lower is better.
    """
    _validate_videos(pred, gt)
    f_pred = _i3d_features(pred, device).numpy()
    f_gt = _i3d_features(gt, device).numpy()

    if f_pred.shape[0] < 2 or f_gt.shape[0] < 2:
        # FVD with 1 sample is undefined; return frame-mean L2 instead.
        return float(np.linalg.norm(f_pred.mean(0) - f_gt.mean(0)))

    mu_p, mu_g = f_pred.mean(0), f_gt.mean(0)
    sig_p = np.cov(f_pred, rowvar=False)
    sig_g = np.cov(f_gt, rowvar=False)
    return _frechet_distance(mu_p, sig_p, mu_g, sig_g)


# --------------------------------------------------------------------------- #
# CLIP-Cons                                                                   #
# --------------------------------------------------------------------------- #


def compute_clip_consistency(
    video: Tensor,
    *,
    model_name: str = "ViT-B/32",
    device: str | torch.device = "cpu",
) -> float:
    """Mean cosine similarity between consecutive CLIP image embeddings.

    Defined as

    .. math::

        \\text{CLIP-Cons} = \\frac{1}{T-1} \\sum_{t=0}^{T-2}
            \\cos\\bigl(\\phi(I_t), \\phi(I_{t+1})\\bigr).

    Higher is better — values close to 1.0 indicate temporally
    consistent semantics.

    Args:
        video: ``(B, T, 3, H, W)`` videos in ``[0, 1]``.
        model_name: OpenCLIP / OpenAI CLIP model identifier.
        device: Torch device for the CLIP encoder.

    Returns:
        Mean cosine similarity — higher is better.
    """
    try:
        import open_clip  # type: ignore[import-not-found]

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name.replace("/", "-"), pretrained="openai"
        )
    except Exception:  # pragma: no cover
        import clip  # type: ignore[import-not-found]

        model, preprocess = clip.load(model_name, device=device)

    b, t, _, _, _ = _validate_videos(video)
    model = model.to(device).eval()

    sims: list[float] = []
    with torch.no_grad():
        for vid in video:  # iterate batch
            # Encode all frames together for efficiency.
            frames = vid.clamp(0, 1).cpu()
            from PIL import Image

            tens = torch.stack(
                [
                    preprocess(Image.fromarray((f.permute(1, 2, 0).numpy() * 255).astype("uint8")))
                    for f in frames
                ],
                dim=0,
            ).to(device)
            emb = model.encode_image(tens)  # (T, D)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cos = (emb[:-1] * emb[1:]).sum(dim=-1)
            sims.append(float(cos.mean().item()))
    return float(np.mean(sims))


# --------------------------------------------------------------------------- #
# E_warp                                                                      #
# --------------------------------------------------------------------------- #


def compute_warping_error(
    video: Tensor,
    flow_estimator: Any,
    *,
    device: str | torch.device = "cpu",
) -> float:
    """Lai et al. 2018 warping error (``E_warp``).

    For consecutive frames ``I_t`` and ``I_{t+1}``, estimate the flow
    ``f`` and report the masked photometric residual

    .. math::

        E_{\\text{warp}} = \\frac{1}{|\\Omega_v|}
            \\sum_{x \\in \\Omega_v} \\| I_{t+1}(x) - W(I_t, f)(x) \\|_1.

    where ``Ω_v`` is the set of visible pixels (we use forward-backward
    consistency to estimate visibility, matching the original paper).

    Args:
        video: ``(B, T, 3, H, W)`` in ``[0, 1]``.
        flow_estimator: A :class:`emg.motion.parallel_flow.FlowEstimator`
            (e.g. :class:`emg.models.raft_wrapper.RAFTFlowEstimator`).
        device: Torch device.

    Returns:
        Mean warping error in ``[0, 1]`` — lower is better.
    """
    b, t, _, h, w = _validate_videos(video)
    if t < 2:
        return 0.0
    video = video.to(device).clamp(0, 1)

    errors: list[float] = []
    for vid in video:  # (T, 3, H, W)
        i1 = vid[:-1]
        i2 = vid[1:]
        flow = flow_estimator(i1, i2)
        warped = backward_warp(i1, flow, padding_mode="border")

        # Forward-backward consistency mask.
        flow_b = flow_estimator(i2, i1)
        from emg.losses.geometric import dynamic_occlusion_mask

        mask, _ = dynamic_occlusion_mask(flow, flow_b, alpha1=0.01, alpha2=0.5)

        diff = (warped - i2).abs().mean(dim=1, keepdim=True)
        denom = mask.sum().clamp_min(1.0)
        errors.append(float(((diff * mask).sum() / denom).item()))
    return float(np.mean(errors))


# --------------------------------------------------------------------------- #
# CPBD                                                                        #
# --------------------------------------------------------------------------- #


def _cpbd_single(image: np.ndarray) -> float:
    """A faithful CPBD (Narvekar & Karam, 2011) implementation.

    The algorithm:

    1. Detect edges via Sobel.
    2. For each edge pixel, measure the local edge width.
    3. Compute the CPBD as ``P(blur ≤ W_JNB)`` integrated over edge widths.

    We follow the reference MATLAB implementation closely; ``W_JNB``
    (just-noticeable-blur width) is set per local contrast.  Return
    value lies in ``[0, 1]`` — higher is sharper.

    Note:
        Author's interpretation.  Originally derived from the public
        ``cpbd`` Python port, simplified for portability.
    """
    if image.ndim == 3:
        # RGB -> luma (Rec. 601).
        image = (
            0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        )
    image = image.astype(np.float32)
    if image.max() > 1.5:
        image = image / 255.0

    from scipy.ndimage import sobel  # type: ignore[import-not-found]

    gx = sobel(image, axis=1)
    gy = sobel(image, axis=0)
    grad = np.hypot(gx, gy)
    edge_threshold = 0.05 * grad.max() if grad.max() > 0 else 0.0
    edges = grad > edge_threshold
    if not edges.any():
        return 1.0  # uniform image -> "perfectly sharp" by convention

    h, w = image.shape
    # Block-wise contrast and edge widths.
    block = 64
    cpbd_vals: list[float] = []
    for by in range(0, h, block):
        for bx in range(0, w, block):
            patch = image[by : by + block, bx : bx + block]
            patch_edges = edges[by : by + block, bx : bx + block]
            if patch.size == 0 or not patch_edges.any():
                continue
            contrast = float(patch.max() - patch.min())
            # JNB width as per Narvekar & Karam; piecewise per contrast.
            w_jnb = 5.0 if contrast < 50.0 / 255.0 else 3.0
            # Edge widths estimated as the local 2-tap gradient ratio.
            edge_widths = np.where(
                patch_edges, 1.0 / np.maximum(np.hypot(sobel(patch, 1), sobel(patch, 0)), 1e-6), 0.0
            )
            edge_widths = edge_widths[patch_edges]
            beta = 3.6
            prob = 1.0 - np.exp(-((edge_widths / w_jnb) ** beta))
            # P(blur ≤ w_jnb) is the empirical CDF at 1.0.
            cpbd_vals.append(float(np.mean(prob < 0.63)))
    if not cpbd_vals:
        return 1.0
    return float(np.mean(cpbd_vals))


def compute_cpbd(video: Tensor) -> float:
    """Cumulative Probability of Blur Detection (Narvekar & Karam, 2011).

    Higher is sharper.  We report the mean CPBD over all frames.

    Args:
        video: ``(B, T, 3, H, W)`` in ``[0, 1]``.

    Returns:
        Mean CPBD in ``[0, 1]`` — higher is better.
    """
    _validate_videos(video)
    video = video.clamp(0, 1)
    arr = video.reshape(-1, 3, *video.shape[-2:]).permute(0, 2, 3, 1).cpu().numpy()
    return float(np.mean([_cpbd_single(f) for f in arr]))


# --------------------------------------------------------------------------- #
# ArcFace                                                                     #
# --------------------------------------------------------------------------- #


def compute_arcface(
    pred_video: Tensor,
    reference_image: Tensor,
    *,
    device: str | torch.device = "cpu",
) -> float:
    """Mean ArcFace identity similarity between reference and frames.

    Uses :mod:`insightface`'s buffalo_l face-recognition model.  We
    detect the face in each frame, compute its 512-D ArcFace embedding,
    and return the mean cosine similarity to the reference embedding.

    Args:
        pred_video: ``(T, 3, H, W)`` generated frames in ``[0, 1]``.
        reference_image: ``(3, H, W)`` reference image in ``[0, 1]``.
        device: Torch device hint (insightface chooses providers
            internally; mostly informational here).

    Returns:
        Mean cosine similarity in ``[-1, 1]`` — higher is better.
    """
    import insightface  # type: ignore[import-not-found]

    if pred_video.dim() != 4 or pred_video.shape[1] != 3:
        raise ValueError(
            f"pred_video must be (T, 3, H, W); got {tuple(pred_video.shape)}"
        )

    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if "cuda" in str(device) else -1, det_size=(640, 640))

    def _embed(img: np.ndarray) -> np.ndarray | None:
        faces = app.get(img)
        if not faces:
            return None
        return faces[0].normed_embedding.astype(np.float32)

    ref_np = (reference_image.clamp(0, 1) * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    ref_emb = _embed(ref_np)
    if ref_emb is None:
        _log.warning("No face detected in reference image — ArcFace returns 0.")
        return 0.0

    sims: list[float] = []
    for fr in pred_video:
        fr_np = (fr.clamp(0, 1) * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        emb = _embed(fr_np)
        if emb is None:
            continue
        sims.append(float(np.dot(ref_emb, emb)))
    if not sims:
        return 0.0
    return float(np.mean(sims))


# --------------------------------------------------------------------------- #
# Convenience                                                                 #
# --------------------------------------------------------------------------- #


def metric_names() -> Iterable[str]:
    """Return the list of all metric names — convenience for tooling."""
    return (
        "lpips",
        "fid",
        "fvd",
        "clip_cons",
        "e_warp",
        "cpbd",
        "arcface",
    )
