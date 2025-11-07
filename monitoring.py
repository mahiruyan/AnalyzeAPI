"""Basit dosya tabanlı izleme yardımcıları."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils import safe_print

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "scraper.log"


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_scraper_event(
    level: str,
    platform: str,
    event: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    _ensure_log_dir()
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "platform": platform,
        "event": event,
        "extra": extra or {},
    }
    with LOG_FILE.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    safe_print(f"[SCRAPER][{platform}][{level.upper()}] {event} -> {extra or {}}")


def log_scraper_error(platform: str, stage: str, error: str, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = extra or {}
    payload.update({"stage": stage, "error": error})
    log_scraper_event("error", platform, "scraper_error", payload)

