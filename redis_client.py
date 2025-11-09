"""
Redis istemci yardımcıları.

Bu modül, uygulamanın Redis bağlantısını yönetir. Tek hesaplanan
bağlantı nesnesini `get_redis` fonksiyonu ile elde edebilirsiniz.
`REDIS_URL` tanımlı değilse veya bağlantı kurulamazsa kontrollü
exception fırlatırız ki çağıran katman fallback ya da hata
mesajını açıkça gösterebilsin.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import redis

from utils import safe_print


class RedisNotConfigured(RuntimeError):
    """REDIS_URL tanımlı değilse fırlatılır."""


def _build_redis_client(url: str) -> redis.Redis:
    return redis.from_url(
        url,
        decode_responses=True,
        health_check_interval=30,
        socket_timeout=5,
        socket_connect_timeout=5,
    )


@lru_cache(maxsize=1)
def get_redis() -> redis.Redis:
    """
    Paylaşılan Redis bağlantısını döndürür.

    Raises:
        RedisNotConfigured: REDIS_URL set edilmemişse.
        redis.RedisError: Bağlantı kurulamazsa.
    """

    url = os.getenv("REDIS_URL")
    if not url:
        raise RedisNotConfigured("REDIS_URL environment variable is not set")

    safe_print("[REDIS] Creating Redis connection...")
    client = _build_redis_client(url)
    client.ping()
    safe_print("[REDIS] Connection established")
    return client


def try_get_redis() -> Optional[redis.Redis]:
    """
    REDIS_URL tanımlı değilse None döner, aksi halde bağlantı döner.
    Bağlantı hatalarında exception fırlatır ki loglanabilsin.
    """

    url = os.getenv("REDIS_URL")
    if not url:
        return None
    return get_redis()


def reset_connection_cache() -> None:
    """Testlerde cache'i temizlemek için."""

    get_redis.cache_clear()

