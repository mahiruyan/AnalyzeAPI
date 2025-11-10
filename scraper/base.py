"""Scraper temel sfler ve yardmc14lar."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import httpx
import os

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
    proxy_url: Optional[str] = None

    def __post_init__(self) -> None:
        self.cookie_store: Optional[CookieStore] = build_cookie_store(self.cookie_store_path)
        if not self.proxy_url:
            env_proxy_url = os.getenv("SCRAPINGBEE_PROXY_URL")
            if env_proxy_url:
                self.proxy_url = env_proxy_url
            else:
                proxy_host = os.getenv("SCRAPINGBEE_PROXY_HOST")
                proxy_port = os.getenv("SCRAPINGBEE_PROXY_PORT")
                if proxy_host and proxy_port:
                    proxy_user = os.getenv("SCRAPINGBEE_PROXY_USER")
                    proxy_pass = os.getenv("SCRAPINGBEE_PROXY_PASS") or os.getenv("SCRAPINGBEE_API_KEY")
                    if proxy_user and proxy_pass:
                        self.proxy_url = f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
                    else:
                        self.proxy_url = f"http://{proxy_host}:{proxy_port}"

    @contextmanager
    def client(self, platform: str) -> Iterator[httpx.Client]:
        headers = {"User-Agent": self.user_agent}
        cookies = self.cookie_store.load() if self.cookie_store else httpx.Cookies()
        if self.cookies:
            for name, value in self.cookies.items():
                cookies.set(name, value)

        log_scraper_event("info", platform, "scraper_client_init", {"timeout": self.timeout})

        proxies = None
        if self.proxy_url:
            proxies = {
                "http://": self.proxy_url,
                "https://": self.proxy_url,
            }
            host_display = self.proxy_url.split("@")[-1]
            host_display = host_display.replace("http://", "").replace("https://", "")
            log_scraper_event("info", platform, "scraper_proxy_enabled", {"host": host_display})

        client_kwargs = {
            "headers": headers,
            "cookies": cookies,
            "timeout": self.timeout,
        }
        if proxies:
            client_kwargs["proxies"] = proxies

        with httpx.Client(**client_kwargs) as client:
            try:
                yield client
            finally:
                if self.cookie_store:
                    self.cookie_store.update_from_client(client)
                    log_scraper_event(
                        "info", platform, "scraper_cookies_persisted", {"path": self.cookie_store_path}
                    )

