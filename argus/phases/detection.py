"""Phase 1 — LFM2.5-VL unified detection, localization & assessment.

Replaces the previous YOLO-based detection. The VLM detects objects,
returns normalized bounding boxes, classifies them, and provides a
tactical assessment — all in a single inference pass.
"""

from __future__ import annotations

import json
import re
from typing import Any

import torch
from PIL import Image

from argus.config import cfg, get_logger
from argus.loader import ModelRegistry
from argus.models import Target

log = get_logger(__name__)

_DETECTION_PROMPT = (
    "You are an orbital intelligence analyst examining satellite imagery "
    "from a defense reconnaissance satellite at ~800 km altitude.\n\n"
    "Analyze this image and detect ALL military-relevant objects visible. "
    "This includes:\n"
    "- Military vehicles (tanks, APCs, trucks, convoys)\n"
    "- Aircraft (fighters, bombers, helicopters, transport planes)\n"
    "- Naval vessels (warships, patrol boats, submarines)\n"
    "- Military infrastructure (bases, airstrips, radar, bunkers)\n"
    "- Storage facilities (fuel depots, ammo dumps, warehouses)\n"
    "- Strategic structures (bridges, harbors, ports)\n\n"
    "For each detected object, provide:\n"
    '- "label": specific type of object\n'
    '- "bbox": normalized bounding box [x1, y1, x2, y2] in [0,1] range\n'
    '- "threat_level": "LOW", "MEDIUM", or "HIGH"\n'
    '- "confidence": detection confidence 0.0 to 1.0\n'
    '- "reasoning": brief tactical assessment\n\n'
    "Response MUST be a valid JSON array:\n"
    '[{"label": "...", "bbox": [x1, y1, x2, y2], "threat_level": "...", '
    '"confidence": 0.0, "reasoning": "..."}]\n\n'
    "If no military targets are visible, return: []\n"
    "Respond ONLY with the JSON array, no other text."
)


def detect(image: Image.Image, registry: ModelRegistry) -> list[Target]:
    """Run LFM2.5-VL unified detection + assessment on a satellite frame.

    Returns a list of :class:`Target` objects with bounding-box crops
    and pre-populated ``vlm_assessment`` dicts.
    """
    log.info("  Phase 1: LFM2.5-VL detection + assessment ...")

    vlm = registry.vlm
    processor = registry.vlm_processor

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _DETECTION_PROMPT},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(vlm.device)

    with torch.inference_mode():
        outputs = vlm.generate(
            **inputs,
            max_new_tokens=cfg.vlm_max_tokens,
            temperature=0.1,
            do_sample=True,
            min_p=0.15,
            repetition_penalty=1.05,
        )

    raw_output = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Strip the prompt echo if the model repeats it
    if _DETECTION_PROMPT[:40] in raw_output:
        raw_output = raw_output.split(_DETECTION_PROMPT[:40])[-1].strip()

    detections = _parse_detections(raw_output)
    log.info("     %d target(s) detected by LFM", len(detections))

    # Convert VLM detections → Target objects with image crops
    targets: list[Target] = []
    w, h = image.size

    for det in detections:
        bbox_norm = det.get("bbox", [0, 0, 0, 0])
        if len(bbox_norm) != 4:
            continue

        # Denormalize [0,1] → pixel coordinates
        x1 = max(0, int(bbox_norm[0] * w))
        y1 = max(0, int(bbox_norm[1] * h))
        x2 = min(w, int(bbox_norm[2] * w))
        y2 = min(h, int(bbox_norm[3] * h))

        # Skip implausibly tiny crops
        if (x2 - x1) < cfg.min_crop_px or (y2 - y1) < cfg.min_crop_px:
            continue

        crop = image.crop((x1, y1, x2, y2))
        label = det.get("label", "unknown")
        conf = float(det.get("confidence", 0.5))
        threat = det.get("threat_level", "UNKNOWN")
        reasoning = det.get("reasoning", "")

        target = Target(
            crop=crop,
            class_name=label,
            confidence=conf,
            bbox=[x1, y1, x2, y2],
            vlm_assessment={
                "target_type": label,
                "threat_level": threat,
                "confidence": conf,
                "reasoning": reasoning,
            },
        )
        targets.append(target)

        log.info(
            "     [%s] %s (conf=%.2f) — %s",
            threat,
            label,
            conf,
            reasoning[:80],
        )

    return targets


def _parse_detections(raw: str) -> list[dict[str, Any]]:
    """Extract JSON array from VLM output (best-effort)."""
    # Strategy 1: find a JSON array [...] in the output
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 2: try the entire stripped output
    try:
        result = json.loads(raw.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    log.warning("  Could not parse VLM detection output as JSON")
    log.warning("  Raw VLM output: %.500s", raw)
    return []
