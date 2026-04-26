"""HTTP client for the satellite dashboard API."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

from argus.config import cfg, get_logger

log = get_logger(__name__)


class SatelliteClient:
    """Thin wrapper around the satellite dashboard REST endpoints."""

    def __init__(self) -> None:
        self._session = self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    # ------------------------------------------------------------------

    def get_position(self) -> list | None:
        """Return ``[lon, lat, alt]`` or *None* on failure."""
        try:
            resp = self._session.get(
                f"{cfg.api_base_url}/data/current/position",
                timeout=cfg.http_timeout,
            )
            resp.raise_for_status()
            return resp.json()["lon-lat-alt"]
        except Exception:
            log.exception("Failed to fetch satellite position")
            return None

    def get_image(self) -> Image.Image | None:
        """Download the latest nadir Mapbox frame (current satellite position).

        Uses SimSat's ``/data/current/image/mapbox`` endpoint which
        automatically targets the satellite's current ground track.
        Single API call — no need to fetch position first.
        """
        try:
            resp = self._session.get(
                f"{cfg.simsat_api_base_url}/data/current/image/mapbox",
                timeout=cfg.http_timeout,
            )
        except Exception:
            log.exception("Nadir image request failed")
            return None

        return self._parse_image_response(resp, label="nadir")

    def get_image_at(
        self,
        lon_target: float,
        lat_target: float,
        lon_satellite: float | None = None,
        lat_satellite: float | None = None,
        alt_satellite: float | None = None,
    ) -> Image.Image | None:
        """Download a Mapbox frame for a specific ground target.

        Uses SimSat's ``/data/image/mapbox`` endpoint, which renders
        Mapbox satellite imagery from the satellite's oblique perspective
        rather than the default nadir view. Useful for off-nadir targets.

        Args:
            lon_target:  Longitude of the ground target.
            lat_target:   Latitude of the ground target.
            lon_satellite: Longitude of the satellite (default: fetched from /data/current/position).
            lat_satellite: Latitude of the satellite (default: fetched from /data/current/position).
            alt_satellite: Altitude of the satellite in km (default: fetched from /data/current/position).

        Returns:
            PIL Image or *None* on any failure.
        """
        if lon_satellite is None or lat_satellite is None or alt_satellite is None:
            pos = self.get_position()
            if pos is None:
                log.error("Cannot call targeted mapbox endpoint — satellite position unavailable")
                return None
            lon_satellite, lat_satellite, alt_satellite = pos

        try:
            resp = self._session.get(
                f"{cfg.simsat_api_base_url}/data/image/mapbox",
                params={
                    "lon_target": lon_target,
                    "lat_target": lat_target,
                    "lon_satellite": lon_satellite,
                    "lat_satellite": lat_satellite,
                    "alt_satellite": alt_satellite,
                },
                timeout=cfg.http_timeout,
            )
        except Exception:
            log.exception("Targeted image request failed")
            return None

        return self._parse_image_response(resp, label="targeted")

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _parse_image_response(
        self, resp: requests.Response, *, label: str = "image"
    ) -> Image.Image | None:
        """Validate and decode an image response from SimSat."""
        if resp.status_code in (401, 500):
            log.error(
                "API %d — check your MAPBOX_ACCESS_TOKEN on the SimSat side.",
                resp.status_code,
            )
            return None
        if resp.status_code != 200:
            log.warning("Unexpected status %d from %s API.", resp.status_code, label)
            return None
        if not resp.content:
            log.warning("Empty image payload — skipping frame.")
            return None

        ctype = resp.headers.get("Content-Type", "")
        if "image/" not in ctype:
            log.warning("Expected image content-type, got '%s'.", ctype)
            return None

        try:
            raw = io.BytesIO(resp.content)
            img = Image.open(raw)
            img.verify()
            raw.seek(0)
            img = Image.open(raw)
            img = img.convert("RGB")
        except Exception:
            log.exception("Failed to decode %s image payload", label)
            return None

        # Ghost image detection (ocean / blank / Mapbox sampling bugs)
        if self._is_ghost_image(img):
            log.info("Ghost image detected (ocean/blank) — skipping frame.")
            return None

        return img

    @staticmethod
    def _is_ghost_image(img: Image.Image) -> bool:
        """Detect near-blank frames (ocean, white, Mapbox sampling bugs).

        Computes the pixel-level standard deviation of the RGB image.
        Frames with very low variance are ocean/blank shots that would
        produce no useful detections.
        """
        arr = np.array(img, dtype=np.float32)
        return float(np.std(arr)) < cfg.ghost_image_std_thresh
