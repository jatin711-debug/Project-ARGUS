"""Pipeline orchestrator — LFM-first 3-phase architecture.

Phase 1  LFM2.5-VL  Detect + locate + classify + assess (single pass)
Phase 2  DA3        Depth verification — flag flat 2-D decoys
Phase 3  Report     Assemble JSON tactical downlink
"""

from __future__ import annotations

import json
import time
from typing import Any

from argus.config import get_logger
from argus.loader import ModelRegistry
from argus.phases.detection import detect
from argus.phases.depth import analyse_depth
from argus.report import build_report
from argus.satellite import SatelliteClient

log = get_logger(__name__)


class Pipeline:
    """Single-shot scan cycle over the satellite feed."""

    def __init__(self, registry: ModelRegistry, client: SatelliteClient) -> None:
        self._registry = registry
        self._client = client

    def run(self) -> dict[str, Any] | None:
        """Execute one full ARGUS scan and return the report, or *None*."""
        log.info("Initiating orbital scan ...")
        timings: dict[str, float] = {}

        # Fetch position
        sat_pos = self._client.get_position()
        if sat_pos is None:
            return None
        log.info("Satellite at: %s", sat_pos)

        # Fetch image
        t0 = time.perf_counter()
        image = self._client.get_image()
        if image is None:
            return None
        timings["acquisition"] = round(time.perf_counter() - t0, 3)
        image_size_bytes = image.width * image.height * 3  # raw RGB size
        log.info(
            "Image acquired (%dx%d, ~%.1f KB raw) -- entering pipeline.",
            image.width,
            image.height,
            image_size_bytes / 1024,
        )

        # Phase 1 — LFM2.5-VL unified detection + assessment
        t0 = time.perf_counter()
        targets = detect(image, self._registry)
        timings["phase1_vlm_detection"] = round(time.perf_counter() - t0, 3)
        if not targets:
            log.info("  Sector clear -- no targets of interest.")
            return None

        # Phase 2 — Depth verification (decoy check)
        t0 = time.perf_counter()
        targets = analyse_depth(targets, self._registry)
        timings["phase2_depth"] = round(time.perf_counter() - t0, 3)

        # Decoy filtering
        real_targets = [
            t for t in targets if t.depth_analysis and t.depth_analysis.is_3d
        ]
        decoys = [
            t for t in targets if not (t.depth_analysis and t.depth_analysis.is_3d)
        ]
        if decoys:
            log.info(
                "  Filtered %d flat decoy(s) via depth verification.",
                len(decoys),
            )

        # Merge: real targets first, then filtered decoys
        all_targets = real_targets + decoys

        # Phase 3 — assemble report
        satellite_position = {
            "lon": sat_pos[0] if isinstance(sat_pos, list) else sat_pos,
            "lat": sat_pos[1] if isinstance(sat_pos, list) else None,
            "alt": (
                sat_pos[2]
                if isinstance(sat_pos, list) and len(sat_pos) > 2
                else None
            ),
        }
        report = build_report(
            all_targets,
            satellite_position,
            timings=timings,
            image_size_bytes=image_size_bytes,
            decoy_count=len(decoys),
        )
        report_json = json.dumps(report, indent=2)

        log.info(
            "\n%s\n  DOWNLINK -- TACTICAL REPORT\n%s\n%s\n%s",
            "=" * 60,
            "=" * 60,
            report_json,
            "=" * 60,
        )
        savings = report.get("edge_compute_savings", {})
        log.info(
            "Report size: %d bytes | Bandwidth saved: %s | Downlink complete.",
            len(report_json),
            savings.get("compression_ratio", "N/A"),
        )
        return report
