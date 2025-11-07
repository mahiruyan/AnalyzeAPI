"""Scraper temel sfler ve yardmc14lar."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import httpx

from monitoring import log_scraper_event
from .cookies import CookieStore, build_cookie_store


class ScrapeError(RuntimeError):
    """Scraping hatalarını temsil eden özel istisna."""


@dataclass
class ScraperConfig:
    """Platform scraper'ları için ortak konfig."""

    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    )
    cookies: Optional[Dict[str, str]] = None
    timeout: int = 15
    cookie_store_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.cookie_store: Optional[CookieStore] = build_cookie_store(self.cookie_store_path)

    @contextmanager
    def client(self, platform: str) -> Iterator[httpx.Client]:
        headers = {"User-Agent": self.user_agent}
        cookies = self.cookie_store.load() if self.cookie_store else httpx.Cookies()
        if self.cookies:
            for name, value in self.cookies.items():
                cookies.set(name, value)

        log_scraper_event("info", platform, "scraper_client_init", {"timeout": self.timeout})

        with httpx.Client(headers=headers, cookies=cookies, timeout=self.timeout) as client:
            try:
                yield client
            finally:
                if self.cookie_store:
                    self.cookie_store.update_from_client(client)
                    log_scraper_event(
                        "info", platform, "scraper_cookies_persisted", {"path": self.cookie_store_path}
                    )

