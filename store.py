from __future__ import annotations

import os
import time
import threading
import uuid
import json
from typing import Dict, Any, Optional


_lock = threading.RLock()
_jobs: Dict[str, Dict[str, Any]] = {}
_cancellations: Dict[str, bool] = {}

# Redis opsiyonel
_redis = None
_redis_prefix = os.getenv("REDIS_PREFIX", "analyzeapi:")
try:
    import redis  # type: ignore

    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        _redis = redis.from_url(redis_url, decode_responses=True)
except Exception:
    _redis = None


def _now() -> float:
    return time.time()


def create_job(payload: Dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())
    doc = {
        "id": job_id,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "payload": payload,
        "intermediate": {},
        "result": None,
        "error": None,
    }
    if _redis is not None:
        _redis.set(f"{_redis_prefix}job:{job_id}", json.dumps(doc), ex=60 * 60 * 24)
    else:
        with _lock:
            _jobs[job_id] = doc
    return job_id


def update_status(job_id: str, status: str, intermediate: Optional[Dict[str, Any]] = None) -> None:
    if _redis is not None:
        key = f"{_redis_prefix}job:{job_id}"
        raw = _redis.get(key)
        if not raw:
            return
        doc = json.loads(raw)
        doc["status"] = status
        doc["updated_at"] = _now()
        if intermediate:
            doc.setdefault("intermediate", {}).update(intermediate)
        _redis.set(key, json.dumps(doc), ex=60 * 60 * 24)
        return
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = status
            _jobs[job_id]["updated_at"] = _now()
            if intermediate:
                _jobs[job_id]["intermediate"].update(intermediate)


def complete_job(job_id: str, result: Dict[str, Any]) -> None:
    if _redis is not None:
        key = f"{_redis_prefix}job:{job_id}"
        raw = _redis.get(key)
        if not raw:
            return
        doc = json.loads(raw)
        doc["status"] = "done"
        doc["updated_at"] = _now()
        doc["result"] = result
        _redis.set(key, json.dumps(doc), ex=60 * 60 * 24)
        return
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["updated_at"] = _now()
            _jobs[job_id]["result"] = result


def fail_job(job_id: str, error_message: str) -> None:
    if _redis is not None:
        key = f"{_redis_prefix}job:{job_id}"
        raw = _redis.get(key)
        if not raw:
            return
        doc = json.loads(raw)
        doc["status"] = "failed"
        doc["updated_at"] = _now()
        doc["error"] = error_message
        _redis.set(key, json.dumps(doc), ex=60 * 60 * 24)
        return
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["updated_at"] = _now()
            _jobs[job_id]["error"] = error_message


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    if _redis is not None:
        raw = _redis.get(f"{_redis_prefix}job:{job_id}")
        if raw:
            return json.loads(raw)
        return None
    with _lock:
        return _jobs.get(job_id)


# ---- Idempotency helpers (signature → job_id) ----

def _sig_key(signature: str) -> str:
    return f"{_redis_prefix}sig:{signature}"


def remember_signature(signature: str, job_id: str, ttl_seconds: int = 3600) -> None:
    if _redis is not None:
        _redis.set(_sig_key(signature), job_id, ex=ttl_seconds)
        return
    with _lock:
        # In-memory basit saklama: _jobs içine koymuyoruz; ayrı hafıza kullanmayalım diye intermediate'a yazalım
        _jobs[f"_sig_{signature}"] = {"id": job_id, "ttl": _now() + ttl_seconds}


def find_job_by_signature(signature: str) -> Optional[str]:
    if _redis is not None:
        val = _redis.get(_sig_key(signature))
        return val
    with _lock:
        key = f"_sig_{signature}"
        doc = _jobs.get(key)
        if not doc:
            return None
        if doc.get("ttl", 0) < _now():
            try:
                del _jobs[key]
            except Exception:
                pass
            return None
        return doc.get("id")


def list_jobs(limit: int = 100) -> list[Dict[str, Any]]:
    # Dönen kayıtlar updated_at'e göre DESC sıralanır
    items: list[Dict[str, Any]] = []
    if _redis is not None:
        try:
            pattern = f"{_redis_prefix}job:*"
            for key in _redis.scan_iter(match=pattern, count=500):
                raw = _redis.get(key)
                if not raw:
                    continue
                try:
                    items.append(json.loads(raw))
                except Exception:
                    continue
        except Exception:
            pass
    else:
        with _lock:
            for k, v in _jobs.items():
                if isinstance(v, dict) and v.get("id") and v.get("status"):
                    items.append(v)
    items.sort(key=lambda d: float(d.get("updated_at", 0.0)), reverse=True)
    return items[:limit]


def cancel_job(job_id: str) -> None:
    if _redis is not None:
        _redis.set(f"{_redis_prefix}cancel:{job_id}", "1", ex=60 * 60)
        return
    with _lock:
        _cancellations[job_id] = True


def is_cancelled(job_id: str) -> bool:
    if _redis is not None:
        return _redis.get(f"{_redis_prefix}cancel:{job_id}") == "1"
    with _lock:
        return _cancellations.get(job_id, False)


