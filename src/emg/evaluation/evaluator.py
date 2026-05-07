"""High-level evaluator orchestrating all metrics.

The evaluator runs a pre-defined set of metrics over a stream of
``(prediction, ground_truth)`` pairs and emits two artifacts:

* a JSON report (machine-readable),
* a Markdown table that mirrors the format of Tables 1 / 2 of the
  paper (human-readable).

Each metric is *opt-in*: a user might want to skip ArcFace because
``insightface`` is heavy, or skip FID because they only have a tiny
test set.  This is controlled via :class:`EvaluatorConfig`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import Tensor

from emg.evaluation.metrics import (
    compute_arcface,
    compute_clip_consistency,
    compute_cpbd,
    compute_fid,
    compute_fvd,
    compute_lpips,
    compute_warping_error,
)
from emg.utils.logging import get_logger

__all__ = ["Evaluator", "EvaluatorConfig"]


_log = get_logger()


@dataclass(slots=True)
class EvaluatorConfig:
    """Configuration for :class:`Evaluator`.

    Attributes:
        metrics: Iterable of metric names to run.  Defaults to the full
            suite from Tables 1 + 2.
        device: Torch device for metric backbones.
        output_dir: Where the evaluator writes ``report.json`` and
            ``table.md``.
    """

    metrics: tuple[str, ...] = (
        "lpips",
        "fid",
        "fvd",
        "clip_cons",
        "e_warp",
        "cpbd",
        "arcface",
    )
    device: str = "cpu"
    output_dir: Path = field(default_factory=lambda: Path("./eval_outputs"))


class Evaluator:
    """Compute the paper's metric suite over a video set.

    Args:
        config: An :class:`EvaluatorConfig`.
        flow_estimator: Required for ``e_warp`` (typically a
            :class:`emg.models.raft_wrapper.RAFTFlowEstimator`).
    """

    def __init__(
        self,
        *,
        config: EvaluatorConfig | None = None,
        flow_estimator: Any | None = None,
    ) -> None:
        self.cfg = config or EvaluatorConfig()
        self.flow_estimator = flow_estimator

    # ----------------------- public API -----------------------

    def evaluate(
        self,
        predictions: Tensor,
        ground_truths: Tensor,
        *,
        reference_images: Tensor | None = None,
    ) -> dict[str, float]:
        """Run all configured metrics on a batch of videos.

        Args:
            predictions: ``(N, T, 3, H, W)`` predicted videos in ``[0, 1]``.
            ground_truths: ``(N, T, 3, H, W)`` ground-truth videos in ``[0, 1]``.
            reference_images: ``(N, 3, H, W)`` reference images.  Required
                only if ``arcface`` is in the metric list.

        Returns:
            Dictionary mapping metric name to its scalar value.
        """
        if predictions.shape != ground_truths.shape:
            raise ValueError(
                f"predictions {tuple(predictions.shape)} and ground_truths "
                f"{tuple(ground_truths.shape)} must match"
            )

        results: dict[str, float] = {}
        device = self.cfg.device
        names = set(self.cfg.metrics)

        if "lpips" in names:
            results["lpips"] = compute_lpips(predictions, ground_truths, device=device)
            _log.info("LPIPS = %.4f", results["lpips"])

        if "fid" in names:
            results["fid"] = compute_fid(predictions, ground_truths)
            _log.info("FID = %.4f", results["fid"])

        if "fvd" in names:
            results["fvd"] = compute_fvd(predictions, ground_truths, device=device)
            _log.info("FVD = %.4f", results["fvd"])

        if "clip_cons" in names:
            results["clip_cons"] = compute_clip_consistency(predictions, device=device)
            _log.info("CLIP-Cons = %.4f", results["clip_cons"])

        if "e_warp" in names:
            if self.flow_estimator is None:
                raise RuntimeError("e_warp requires a flow_estimator")
            results["e_warp"] = compute_warping_error(
                predictions, self.flow_estimator, device=device
            )
            _log.info("E_warp = %.4f", results["e_warp"])

        if "cpbd" in names:
            results["cpbd"] = compute_cpbd(predictions)
            _log.info("CPBD = %.4f", results["cpbd"])

        if "arcface" in names:
            if reference_images is None:
                raise RuntimeError("arcface requires reference_images")
            scores: list[float] = []
            for vid, ref in zip(predictions, reference_images, strict=True):
                scores.append(compute_arcface(vid, ref, device=device))
            results["arcface"] = float(sum(scores) / max(len(scores), 1))
            _log.info("ArcFace = %.4f", results["arcface"])

        return results

    def save_report(
        self,
        results: dict[str, float],
        *,
        method_name: str = "Ours",
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        """Persist results as both JSON and Markdown.

        Args:
            results: The dict returned by :meth:`evaluate`.
            method_name: Row label used in the Markdown table.
            extra_meta: Free-form metadata to attach to the JSON.

        Returns:
            Path to the written JSON file.  The Markdown file lives
            alongside it as ``table.md``.
        """
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "report.json"
        md_path = out_dir / "table.md"

        payload = {
            "method": method_name,
            "metrics": results,
            "meta": extra_meta or {},
        }
        json_path.write_text(json.dumps(payload, indent=2))

        md_path.write_text(self._format_markdown_table(method_name, results))

        _log.info("Wrote evaluation report to %s and %s", json_path, md_path)
        return json_path

    # ----------------------- helpers -----------------------

    @staticmethod
    def _format_markdown_table(method: str, results: dict[str, float]) -> str:
        """Format the metric dict in a paper-style row.

        Columns appear in the canonical Table 1 order; missing metrics
        are rendered as ``—``.
        """
        cols = ("lpips", "fid", "fvd", "clip_cons", "e_warp", "cpbd", "arcface")
        headers = (
            "Method",
            "LPIPS ↓",
            "FID ↓",
            "FVD ↓",
            "CLIP-Cons ↑",
            "E_warp (×10⁻³) ↓",
            "CPBD ↑",
            "ArcFace ↑",
        )
        sep = "|".join(["---"] * len(headers))
        row_vals: list[str] = [method]
        for col in cols:
            if col not in results:
                row_vals.append("—")
                continue
            v = results[col]
            if col == "e_warp":
                row_vals.append(f"{v * 1000:.2f}")
            else:
                row_vals.append(f"{v:.4f}")
        return (
            "| " + " | ".join(headers) + " |\n"
            "|" + sep + "|\n"
            "| " + " | ".join(row_vals) + " |\n"
        )

    @staticmethod
    def merge_reports(reports: Iterable[Path]) -> str:
        """Compose a multi-row Markdown table from multiple JSON reports."""
        rows: list[str] = []
        cols = ("lpips", "fid", "fvd", "clip_cons", "e_warp", "cpbd", "arcface")
        headers = (
            "Method", "LPIPS ↓", "FID ↓", "FVD ↓", "CLIP-Cons ↑",
            "E_warp (×10⁻³) ↓", "CPBD ↑", "ArcFace ↑",
        )
        sep = "|".join(["---"] * len(headers))
        rows.append("| " + " | ".join(headers) + " |")
        rows.append("|" + sep + "|")
        for rp in reports:
            data = json.loads(Path(rp).read_text())
            method = data.get("method", "?")
            metrics = data.get("metrics", {})
            row_vals = [method]
            for col in cols:
                if col not in metrics:
                    row_vals.append("—")
                else:
                    v = float(metrics[col])
                    row_vals.append(f"{v * 1000:.2f}" if col == "e_warp" else f"{v:.4f}")
            rows.append("| " + " | ".join(row_vals) + " |")
        return "\n".join(rows) + "\n"
