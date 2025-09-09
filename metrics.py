try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST  # type: ignore
except Exception:  # Paket yoksa no-op metrikler kullan
    Counter = lambda *args, **kwargs: _Noop()  # type: ignore
    Histogram = lambda *args, **kwargs: _Noop()  # type: ignore
    def generate_latest():  # type: ignore
        return b""
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"  # type: ignore

    class _Noop:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            return None
        def observe(self, *args, **kwargs):
            return None

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Toplam API istek sayısı",
    ["path", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API istek gecikmesi",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)


def track_request(path: str, method: str, status_code: int, latency_seconds: float) -> None:
    REQUEST_COUNT.labels(path=path, method=method, status=str(status_code)).inc()
    REQUEST_LATENCY.observe(latency_seconds)


def metrics_response():
    from fastapi import Response

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


