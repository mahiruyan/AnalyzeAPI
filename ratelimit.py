import os
import time
import threading
from typing import Optional


_lock = threading.RLock()
_memory_buckets = {}

_redis = None
try:
    import redis  # type: ignore

    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        _redis = redis.from_url(redis_url, decode_responses=True)
except Exception:
    _redis = None


def allow(key: str, limit: int, window_seconds: int) -> bool:
    now = int(time.time())
    window = now // window_seconds
    if _redis is not None:
        bucket_key = f"rl:{key}:{window}"
        try:
            n = _redis.incr(bucket_key)
            if n == 1:
                _redis.expire(bucket_key, window_seconds)
            return n <= limit
        except Exception:
            pass
    # Memory fallback
    with _lock:
        bucket_key = (key, window)
        n = _memory_buckets.get(bucket_key, 0) + 1
        _memory_buckets[bucket_key] = n
        return n <= limit


