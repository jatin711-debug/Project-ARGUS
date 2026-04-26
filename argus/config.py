"""
Centralised configuration — all tunables in one place.

Values come from environment variables (or .env via a loader), with
sensible defaults for local development.  Import this module anywhere:

    from argus.config import cfg
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
#  Resolve project root (parent of this file's package)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load .env BEFORE Settings reads os.getenv()
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings — built once at import time."""

    # -- Satellite API ------------------------------------------------------
    api_base_url: str = os.getenv("ARGUS_API_URL", "http://localhost:9005")
    simsat_api_base_url: str = os.getenv("ARGUS_SIMSAT_API_URL", "http://localhost:9005")
    http_timeout: int = int(os.getenv("ARGUS_HTTP_TIMEOUT", "30"))

    # -- VLM (LFM2.5-VL — the core of the pipeline) ------------------------
    vlm_model_id: str = os.getenv("ARGUS_VLM_MODEL", "LiquidAI/LFM2.5-VL-450M")
    vlm_adapter_path: str = os.getenv(
        "ARGUS_VLM_ADAPTER",
        str(PROJECT_ROOT / "weights" / "argus-lfm-lora"),
    )
    vlm_max_tokens: int = int(os.getenv("ARGUS_VLM_MAX_TOKENS", "512"))
    min_crop_px: int = int(os.getenv("ARGUS_MIN_CROP_PX", "10"))

    # -- Depth analysis (DA3 — decoy verification) --------------------------
    da3_model_id: str = os.getenv("ARGUS_DA3_MODEL", "depth-anything/DA3-BASE")
    depth_3d_threshold: float = float(os.getenv("ARGUS_DEPTH_THRESH", "0.1"))

    # -- Scanner ------------------------------------------------------------
    scan_interval_sec: int = int(os.getenv("ARGUS_SCAN_INTERVAL", "1"))

    # -- Ghost image detection ----------------------------------------------
    ghost_image_std_thresh: float = float(os.getenv("ARGUS_GHOST_STD", "10.0"))

    # -- Device -------------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Logging ------------------------------------------------------------
    log_level: str = os.getenv("ARGUS_LOG_LEVEL", "INFO")
    log_dir: Path = PROJECT_ROOT / "logs"


# Singleton — importable everywhere as `cfg`
cfg = Settings()


# ---------------------------------------------------------------------------
#  Logger factory
# ---------------------------------------------------------------------------

def get_logger(name: str = "argus") -> logging.Logger:
    """Return a consistently-configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

        # File (rotates manually; swap for RotatingFileHandler if desired)
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(cfg.log_dir / "argus.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        logger.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    return logger
