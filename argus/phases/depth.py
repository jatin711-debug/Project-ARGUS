"""Phase 2 — Depth Anything 3 — 3-D reality check (decoy filtering)."""

from __future__ import annotations

import numpy as np

from argus.config import cfg, get_logger
from argus.loader import ModelRegistry
from argus.models import DepthAnalysis, Target

log = get_logger(__name__)


def analyse_depth(targets: list[Target], registry: ModelRegistry) -> list[Target]:
    """Enrich each target with depth statistics; flag flat 2-D decoys."""
    log.info("  Phase 2: Depth analysis …")
    depth_model = registry.depth

    for i, tgt in enumerate(targets):
        try:
            prediction = depth_model.inference([tgt.crop])
            depth_map = prediction.depth[0]

            d_std = float(np.std(depth_map))
            d_range = float(np.ptp(depth_map))
            d_mean = float(np.mean(depth_map))
            norm_std = d_std / d_mean if d_mean > 1e-6 else 0.0

            is_3d = norm_std > cfg.depth_3d_threshold
            conf = (
                min(1.0, norm_std / cfg.depth_3d_threshold)
                if is_3d
                else max(0.0, 1.0 - norm_std / cfg.depth_3d_threshold)
            )

            tgt.depth_analysis = DepthAnalysis(
                depth_std=round(d_std, 4),
                depth_range=round(d_range, 4),
                norm_std=round(norm_std, 4),
                is_3d=is_3d,
                verdict="REAL" if is_3d else "DECOY",
                verdict_confidence=round(conf, 4),
            )
            log.info(
                "     Target %d: %s (norm_std=%.4f, confidence=%.2f)",
                i + 1,
                "3D REAL" if is_3d else "2D DECOY",
                norm_std,
                conf,
            )
        except Exception:
            log.exception("     Target %d: depth analysis failed", i + 1)
            tgt.depth_analysis = DepthAnalysis(0, 0, 0, False, "DECOY", 0)

    return targets
