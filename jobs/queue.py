"""
Redis tabanlı job kuyruğu için minimal yardımcılar.

Şu aşamada kuyruk yalnızca isteğe bağlı. Redis yapılandırılmamışsa
fonksiyonlar None döndürür ve uygulama senkron şekilde davranmaya
devam eder.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from utils import safe_print
from redis_client import RedisNotConfigured, try_get_redis

QUEUE_NAME = "analyze:queue"


def _json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, set):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def enqueue_analysis(payload: Dict[str, Any]) -> Optional[str]:
    """
    Analiz işini kuyruğa ekler.

    Return:
        job_id (str) Redis'e yazıldıysa; aksi halde None.
    """

    try:
        redis_client = try_get_redis()
    except RedisNotConfigured:
        safe_print("[QUEUE] Redis yok, senkron modda devam.")
        return None
    except Exception as exc:
        safe_print(f"[QUEUE] Redis bağlantı hatası: {exc}")
        return None

    if redis_client is None:
        safe_print("[QUEUE] Redis yapılandırılmamış, senkron mod.")
        return None

    job_id = payload.get("job_id") or str(uuid.uuid4())
    entry = {"job_id": job_id, "payload": payload}
    try:
        redis_client.rpush(QUEUE_NAME, json.dumps(entry, default=_json_default))
        safe_print(f"[QUEUE] Job kuyruğa eklendi: {job_id}")
        return job_id
    except Exception as exc:
        safe_print(f"[QUEUE] Kuyruğa eklenemedi: {exc}")
        return None


def dequeue_analysis(block: bool = True, timeout: int = 5) -> Optional[Dict[str, Any]]:
    """
    Kuyruktan bir analiz işi çeker.

    Return:
        {"job_id": ..., "payload": {...}} veya None.
    """

    try:
        redis_client = try_get_redis()
    except Exception as exc:
        safe_print(f"[QUEUE] Redis bağlantı hatası (dequeue): {exc}")
        return None

    if redis_client is None:
        return None

    try:
        if block:
            result = redis_client.blpop(QUEUE_NAME, timeout=timeout)
            if not result:
                return None
            _, raw_entry = result
        else:
            raw_entry = redis_client.lpop(QUEUE_NAME)
            if raw_entry is None:
                return None
        return json.loads(raw_entry)
    except Exception as exc:
        safe_print(f"[QUEUE] Kuyruktan okunamadı: {exc}")
        return None

