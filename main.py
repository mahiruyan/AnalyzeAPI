from fastapi import FastAPI, HTTPException
from fastapi import BackgroundTasks
from fastapi import Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import tempfile
from pathlib import Path

from utils import download_video, extract_audio_via_ffmpeg, grab_frames, get_video_duration
from utils import with_retry
from features import extract_features
from scoring import score_features
from why import explain_why
from suggest import generate_suggestions
from mailer import send_analysis_email
from store import create_job, get_job, remember_signature, find_job_by_signature, list_jobs, cancel_job
from worker import process_job
from db import init_schema, save_platform_metrics
from ratelimit import allow as rl_allow
from metrics import track_request, metrics_response
from sheets import append_row
from storage import generate_presigned_put_url


class AnalyzeRequest(BaseModel):
    platform: str = Field(..., description="Platform name, e.g., instagram, tiktok")
    file_url: str = Field(..., description="Publicly accessible video URL")
    caption: Optional[str] = ""
    title: Optional[str] = ""
    tags: Optional[List[str]] = []
    mode: Optional[str] = "FAST"  # FAST veya FULL
    email: Optional[str] = None


class AnalyzeResponse(BaseModel):
    duration_seconds: float
    features: Dict[str, Any]
    scores: Dict[str, float]
    why: str
    viral: bool
    mode: str
    analysis_complete: bool
    suggestions: Dict[str, Any]


# .env yükleme (varsa)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


app = FastAPI(title="analyzeAPI", version="0.1.1")


# ---- Sentry (opsiyonel) ----
try:
    import sentry_sdk  # type: ignore
    if os.getenv("SENTRY_DSN"):
        sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.0")))
except Exception:
    pass


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks, request: Request):
    try:
        # Rate limit: IP başına 60/dk (Redis varsa), aksi halde memory fallback
        ip = request.client.host if request and request.client else "unknown"
        if not rl_allow(f"analyze:{ip}", limit=60, window_seconds=60):
            raise HTTPException(status_code=429, detail="Rate limit aşıldı. Lütfen sonra tekrar deneyin.")

        import time
        start = time.time()
        payload = req.model_dump()
        # Idempotensi: aynı platform+post_id ya da aynı file_url için son 1 saat içinde job varsa onu döndür
        sig_parts = [payload.get("platform", ""), payload.get("post_id", ""), payload.get("file_url", "")]
        signature = "|".join(sig_parts)
        existing = find_job_by_signature(signature)
        if existing:
            resp = {"analysis_id": existing, "status": "queued"}
            track_request("/api/analyze", "POST", 200, time.time() - start)
            return resp
        job_id = create_job(payload)
        remember_signature(signature, job_id, ttl_seconds=3600)

        def _run_job():
            try:
                result = process_job(job_id, payload)
                # optional email
                if req.email:
                    try:
                        send_analysis_email(
                            to_email=req.email,
                            platform=req.platform,
                            scores=result.get("scores", {}),
                            why=result.get("why", ""),
                            suggestions=result.get("suggestions", {}),
                        )
                    except Exception:
                        pass
                # optional Google Sheets
                try:
                    sheet_id = os.getenv("GOOGLE_SHEET_ID")
                    sheet_ws = os.getenv("GOOGLE_SHEET_WS", "Analyses")
                    if sheet_id and os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"):
                        scores = result.get("scores", {})
                        append_row(
                            sheet_id,
                            sheet_ws,
                            [
                                job_id,
                                req.platform,
                                result.get("viral", False),
                                scores.get("overall_score", 0.0),
                                result.get("pdf_url", ""),
                            ],
                        )
                except Exception:
                    pass
            except Exception:
                pass

        background_tasks.add_task(_run_job)
        resp = {"analysis_id": job_id, "status": "queued"}
        track_request("/api/analyze", "POST", 200, time.time() - start)
        return resp
    except HTTPException:
        raise
    except Exception as e:
        try:
            import time
            track_request("/api/analyze", "POST", 500, 0.0)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"İşleme hatası: {e}")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/version")
def version():
    return {"name": "analyzeAPI", "version": app.version}


@app.get("/metrics")
def metrics():
    return metrics_response()


def _require_admin(request: Request) -> None:
    token = os.getenv("ADMIN_TOKEN", "")
    provided = request.headers.get("x-admin-token") if request and request.headers else None
    if not token or provided != token:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/api/admin/jobs")
def admin_jobs(request: Request, limit: int = 100):
    _require_admin(request)
    return {"items": list_jobs(limit=limit)}


@app.get("/api/admin/jobs/{analysis_id}")
def admin_job_detail(analysis_id: str, request: Request):
    _require_admin(request)
    job = get_job(analysis_id)
    if not job:
        raise HTTPException(status_code=404, detail="analysis_id bulunamadı")
    return job


@app.post("/api/admin/jobs/{analysis_id}/cancel")
def admin_job_cancel(analysis_id: str, request: Request):
    _require_admin(request)
    cancel_job(analysis_id)
    return {"status": "cancellation_requested"}


@app.post("/api/admin/jobs/{analysis_id}/retry")
def admin_job_retry(analysis_id: str, request: Request):
    _require_admin(request)
    job = get_job(analysis_id)
    if not job:
        raise HTTPException(status_code=404, detail="analysis_id bulunamadı")
    payload = job.get("payload", {})
    # Yeni job id ile yeniden kuyruğa al
    new_id = create_job(payload)
    from fastapi import BackgroundTasks
    # Not: admin endpoints fonksiyon dışı BackgroundTasks yok; senkron tetiklemeyelim.
    # Kullanıcı /api/analyze ile aynı sonuçtadır. Basitçe id döndürelim.
    return {"analysis_id": new_id, "status": "queued"}


@app.post("/api/upload/presign")
def presign_put(payload: Dict[str, Any]):
    key = payload.get("key")
    content_type = payload.get("content_type", "application/octet-stream")
    expires = int(payload.get("expires", 60))
    if not key:
        raise HTTPException(status_code=400, detail="key zorunlu")
    try:
        url = generate_presigned_put_url(key, content_type=content_type, expires_minutes=expires)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Presign hatası: {e}")
    return {"url": url}


@app.on_event("startup")
def on_startup():
    # DB şemasını (varsa DATABASE_URL) hazırla
    try:
        init_schema()
    except Exception:
        pass


@app.get("/api/analysis/{analysis_id}")
def analysis_status(analysis_id: str):
    job = get_job(analysis_id)
    if not job:
        raise HTTPException(status_code=404, detail="analysis_id bulunamadı")
    # Ara çıktıları ve sonuçları döndür
    return {
        "analysis_id": analysis_id,
        "status": job["status"],
        "intermediate": job.get("intermediate", {}),
        "result": job.get("result"),
        "error": job.get("error"),
        "updated_at": job.get("updated_at"),
    }


@app.post("/api/ingest/graph")
def ingest_graph(payload: Dict[str, Any]):
    required = {"platform", "post_id", "metrics"}
    if not required.issubset(set(payload.keys())):
        raise HTTPException(status_code=400, detail="platform, post_id, metrics zorunlu")
    try:
        save_platform_metrics(payload["platform"], payload["post_id"], payload["metrics"])
    except Exception:
        # DB yoksa veya hata olursa yine accepted döndür; loglama ileride
        pass
    return {"status": "accepted"}


@app.get("/api/webhooks/ig")
def webhook_ig_verify(hub_mode: Optional[str] = None, hub_challenge: Optional[str] = None, hub_verify_token: Optional[str] = None):
    # Meta webhook doğrulaması
    verify_token = os.getenv("IG_VERIFY_TOKEN", "")
    if hub_mode == "subscribe" and hub_verify_token and hub_verify_token == verify_token:
        return int(hub_challenge or 0)
    raise HTTPException(status_code=403, detail="verification failed")


@app.post("/api/webhooks/ig")
def webhook_ig(event: Dict[str, Any], request: Any = None):
    # HMAC imza doğrulaması (opsiyonel) — Meta X-Hub-Signature-256
    app_secret = os.getenv("IG_APP_SECRET", "")
    try:
        import hmac, hashlib, json
        if request is not None and app_secret:
            raw_body = getattr(request, "_body", None)
            if raw_body is None and hasattr(request, "body"):
                raw_body = request.body()
            if callable(raw_body):
                raw_body = raw_body()
            if isinstance(raw_body, bytes):
                body_bytes = raw_body
            else:
                body_bytes = json.dumps(event).encode("utf-8")
            signature = request.headers.get("X-Hub-Signature-256") if hasattr(request, "headers") else None
            if signature and signature.startswith("sha256="):
                expected = hmac.new(app_secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
                provided = signature.split("=", 1)[1]
                if not hmac.compare_digest(expected, provided):
                    raise HTTPException(status_code=403, detail="invalid signature")
    except HTTPException:
        raise
    except Exception:
        # İmza doğrulaması opsiyonel; hatada devam et
        pass
    if not event:
        raise HTTPException(status_code=400, detail="Boş payload")
    return {"status": "received"}


