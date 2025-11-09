import json

import pytest

from jobs import queue


class DummyRedis:
    def __init__(self):
        self.items = []

    def rpush(self, key, value):
        self.items.append((key, value))
        return len(self.items)

    def blpop(self, key, timeout=0):
        if not self.items:
            return None
        entry_key, value = self.items.pop(0)
        return entry_key, value

    def lpop(self, key):
        if not self.items:
            return None
        _, value = self.items.pop(0)
        return value


def test_enqueue_returns_none_without_redis(monkeypatch):
    monkeypatch.setattr(queue, "try_get_redis", lambda: None)
    job_id = queue.enqueue_analysis({"file_url": "x"})
    assert job_id is None


def test_enqueue_and_dequeue_roundtrip(monkeypatch):
    dummy = DummyRedis()
    monkeypatch.setattr(queue, "try_get_redis", lambda: dummy)

    payload = {"file_url": "https://example.com/video.mp4"}
    job_id = queue.enqueue_analysis(payload)
    assert job_id is not None
    assert len(dummy.items) == 1

    popped = queue.dequeue_analysis(block=False)
    assert popped is not None
    assert popped["job_id"] == job_id
    assert popped["payload"]["file_url"] == payload["file_url"]

