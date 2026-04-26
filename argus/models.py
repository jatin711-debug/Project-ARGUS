"""Data models shared across pipeline phases."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from PIL import Image


@dataclass
class DepthAnalysis:
    """Phase 2 output — depth statistics for a single crop."""

    depth_std: float
    depth_range: float
    norm_std: float
    is_3d: bool
    verdict: str          # "REAL" or "DECOY" — human-readable label
    verdict_confidence: float  # confidence in the verdict above

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Target:
    """A detected target flowing through the pipeline.

    ``crop`` is the PIL image region; it is **not** serialised into reports.
    ``vlm_assessment`` is populated by Phase 1 (LFM detection) and contains
    the VLM's tactical analysis including threat level and reasoning.
    """

    crop: Image.Image
    class_name: str
    confidence: float
    bbox: list[float]
    angle: float = 0.0  # reserved for future OBB support

    # Enriched by Phase 1 (LFM detection + assessment)
    vlm_assessment: dict[str, Any] = field(default_factory=dict)

    # Enriched by Phase 2 (DA3 depth verification)
    depth_analysis: DepthAnalysis | None = None
