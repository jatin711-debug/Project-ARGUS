"""``python -m argus`` entrypoint."""

from __future__ import annotations

import signal
import time

from argus.config import cfg, get_logger
from argus.loader import ModelRegistry
from argus.pipeline import Pipeline
from argus.satellite import SatelliteClient

log = get_logger(__name__)

_shutdown = False


def _handle_signal(signum: int, _frame) -> None:  # noqa: ANN001
    global _shutdown
    _shutdown = True
    log.info("Shutdown signal received — finishing current scan …")


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    registry = ModelRegistry()
    registry.load_all()

    client = SatelliteClient()
    pipeline = Pipeline(registry, client)

    if cfg.scan_interval_sec <= 0:
        pipeline.run()
        registry.unload()
        return

    scan_num = 0
    log.info(
        "Continuous scanning — interval %ds  (Ctrl+C to stop)",
        cfg.scan_interval_sec,
    )

    try:
        while not _shutdown:
            scan_num += 1
            log.info("%s  Scan #%d  %s", "-" * 24, scan_num, "-" * 24)
            try:
                pipeline.run()
            except Exception:
                log.exception("Unhandled error during scan #%d", scan_num)

            # Interruptible sleep
            for _ in range(cfg.scan_interval_sec * 10):
                if _shutdown:
                    break
                time.sleep(0.1)
    finally:
        log.info("Stopped after %d scan(s).", scan_num)
        registry.unload()


if __name__ == "__main__":
    main()
