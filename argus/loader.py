"""Model loader — VLM + Depth models loaded once at startup."""

from __future__ import annotations

from typing import Any

import torch

from argus.config import cfg, get_logger

log = get_logger(__name__)


class ModelRegistry:
    """Holds references to all loaded models.

    Usage::

        registry = ModelRegistry()
        registry.load_all()
        vlm = registry.vlm
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    # -- public accessors ---------------------------------------------------

    @property
    def depth(self) -> Any:
        return self._models["depth"]

    @property
    def vlm(self) -> Any:
        return self._models["vlm"]

    @property
    def vlm_processor(self) -> Any:
        return self._models["vlm_processor"]

    # -- loading ------------------------------------------------------------

    def load_all(self) -> None:
        """Load every model. Raises on missing weight files."""
        log.info("Loading ARGUS AI Pipeline models …")
        self._load_vlm()
        self._load_depth()
        log.info("All models loaded — pipeline ready.")

    def _load_depth(self) -> None:
        from depth_anything_3.api import DepthAnything3

        self._models["depth"] = DepthAnything3.from_pretrained(
            cfg.da3_model_id
        ).to(cfg.device)
        log.info("  Depth ready — Depth Anything 3 (%s)", cfg.da3_model_id)

    def _load_vlm(self) -> None:
        from pathlib import Path
        from transformers import AutoProcessor, AutoModelForImageTextToText

        # Load base model
        base = AutoModelForImageTextToText.from_pretrained(
            cfg.vlm_model_id, device_map="auto", torch_dtype=torch.bfloat16,
        )

        # Apply fine-tuned LoRA adapter if available
        adapter_path = Path(cfg.vlm_adapter_path)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            from peft import PeftModel
            base = PeftModel.from_pretrained(base, str(adapter_path))
            base = base.merge_and_unload()  # merge for faster inference
            log.info("  VLM adapter loaded + merged from %s", adapter_path)
        else:
            log.warning("  No adapter at %s — using base model", adapter_path)

        self._models["vlm"] = base
        self._models["vlm_processor"] = AutoProcessor.from_pretrained(
            cfg.vlm_model_id,
        )
        log.info("  VLM ready -- %s", cfg.vlm_model_id)

    # -- teardown -----------------------------------------------------------

    def unload(self) -> None:
        """Release model references and free GPU memory."""
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Models unloaded, GPU cache cleared.")
