"""Phase 5 — Tactical report assembly."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from argus.models import Target


def build_report(
    targets: list[Target],
    satellite_position: dict[str, Any],
    *,
    timings: dict[str, float] | None = None,
    image_size_bytes: int = 0,
    decoy_count: int = 0,
) -> dict[str, Any]:
    """Build a JSON-serialisable tactical report (no PIL objects).

    Args:
        targets: All targets (real + decoys) that passed through the pipeline.
        satellite_position: ``{lon, lat, alt}`` of the satellite.
        timings: Per-phase latency in seconds.
        image_size_bytes: Raw image size for bandwidth-savings calculation.
        decoy_count: Number of flat targets filtered by Phase 2.
    """
    entries: list[dict[str, Any]] = []
    for idx, tgt in enumerate(targets, 1):
        entry: dict[str, Any] = {
            "id": idx,
            "class": tgt.class_name,
            "confidence": tgt.confidence,
            "bbox": tgt.bbox,
        }
        if tgt.depth_analysis:
            entry["depth_analysis"] = tgt.depth_analysis.to_dict()
            entry["is_decoy"] = not tgt.depth_analysis.is_3d
        if tgt.vlm_assessment:
            entry["vlm_assessment"] = tgt.vlm_assessment
        entries.append(entry)

    report: dict[str, Any] = {
        "mission": "PROJECT ARGUS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "satellite_position": satellite_position,
        "targets_detected": len(entries),
        "decoys_filtered": decoy_count,
        "targets": entries,
    }

    # Per-phase latency breakdown
    if timings:
        report["phase_timings_sec"] = timings
        report["total_pipeline_sec"] = round(sum(timings.values()), 3)

    # Edge-compute bandwidth savings — the core value proposition
    report_bytes = len(json.dumps(report))
    if image_size_bytes > 0:
        report["edge_compute_savings"] = {
            "raw_image_bytes": image_size_bytes,
            "report_bytes": report_bytes,
            "compression_ratio": f"{image_size_bytes / max(report_bytes, 1):.0f}x",
            "bandwidth_saved_pct": round(
                (1 - report_bytes / max(image_size_bytes, 1)) * 100, 2
            ),
        }

    return report
