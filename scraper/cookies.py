"""Cookie yönetimi yardımcıları."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Optional

import httpx


class CookieStore:
    """httpx cookie jar'ını disk üzerinde saklamak için basit yardımcı."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def load(self) -> httpx.Cookies:
        with self._lock:
            if not self.path.exists():
                return httpx.Cookies()
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            cookies = httpx.Cookies()
            for name, value in data.items():
                cookies.set(name, value)
            return cookies

    def save(self, cookies: httpx.Cookies) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {cookie[0]: cookie[1] for cookie in cookies.jar}
            self.path.write_text(json.dumps(data), encoding="utf-8")

    def update_from_client(self, client: httpx.Client) -> None:
        self.save(client.cookies)


def build_cookie_store(path: Optional[str]) -> Optional[CookieStore]:
    if not path:
        return None
    return CookieStore(path)

