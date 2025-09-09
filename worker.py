import traceback
from typing import Dict, Any

from store import update_status, complete_job, fail_job, is_cancelled
from db import save_analysis
from utils import download_video, extract_audio_via_ffmpeg, grab_frames, get_video_duration, with_retry
from features import extract_features
from scoring import score_features
from why import explain_why
from suggest import generate_suggestions
from report import build_pdf
from storage import put_bytes, generate_presigned_url
from posting_window import best_posting_windows


def process_job(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        update_status(job_id, "processing", {"step": "download"})
        save_analysis(job_id, payload.get("platform", ""), "processing")
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            video_path = tmp_path / "input.mp4"
            audio_path = tmp_path / "audio.wav"
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            # 1) Medya indirme/süre
            with_retry(lambda: download_video(payload["file_url"], str(video_path)))
            duration = get_video_duration(str(video_path))
            if duration <= 0:
                raise RuntimeError("Video süresi tespit edilemedi veya 0.")

            # 2) Ön-işleme
            update_status(job_id, "processing", {"step": "preprocess"})
            if is_cancelled(job_id):
                raise RuntimeError("Job cancelled")
            extract_audio_via_ffmpeg(str(video_path), str(audio_path), sample_rate=16000, mono=True)
            frames = grab_frames(str(video_path), output_dir=str(frames_dir), fps=1)

            # 3) Özellik çıkarımı
            fast_mode = (payload.get("mode", "FAST").upper() == "FAST")
            update_status(job_id, "processing", {"step": "feature_engineering"})
            save_analysis(job_id, payload.get("platform", ""), "processing")
            if is_cancelled(job_id):
                raise RuntimeError("Job cancelled")
            features = extract_features(
                audio_path=str(audio_path),
                frames=frames,
                caption=payload.get("caption") or "",
                title=payload.get("title") or "",
                tags=payload.get("tags") or [],
                duration_seconds=duration,
                fast_mode=fast_mode,
            )

            # 4) Skorlama
            scores = score_features(features, platform=payload.get("platform", ""), fast_mode=fast_mode)

            # 5) Rapor/öneri üretimi
            update_status(job_id, "generating_report", {"step": "why_and_suggestions"})
            if is_cancelled(job_id):
                raise RuntimeError("Job cancelled")
            save_analysis(job_id, payload.get("platform", ""), "generating_report")
            feature_summary = {
                "first3s_text": bool(features.get("textual", {}).get("caption", "")),
                "has_on_screen_text": bool(features.get("textual", {}).get("ocr_text", "")),
                "lufs": features.get("audio", {}).get("loudness_lufs"),
                "clip_rate": 0.0,
                "optical_flow": features.get("visual", {}).get("optical_flow_mean", 0.0),
                "scene_count": features.get("visual", {}).get("scene_count", 0),
                "duration_s": features.get("duration_seconds", 0.0),
                "title_len": len(features.get("textual", {}).get("title", "")),
                "caption_len": len(features.get("textual", {}).get("caption", "")),
            }
            viral_threshold = 0.6
            why_explanation = explain_why(feature_summary, scores, viral_threshold)
            suggestions = generate_suggestions(
                platform=payload.get("platform", ""),
                caption=payload.get("caption") or "",
                title=payload.get("title") or "",
                tags=payload.get("tags") or [],
                features=features,
                scores=scores,
                fast_mode=fast_mode,
            )

            # PDF üretimi (opsiyonel – S3/R2 ayarlıysa yükle ve linki al)
            pdf_url = None
            try:
                pdf_bytes = build_pdf(scores, why_explanation, suggestions)
                key = f"reports/{job_id}.pdf"
                put_bytes(key, pdf_bytes, content_type="application/pdf")
                pdf_url = generate_presigned_url(key, expires_minutes=60)
            except Exception:
                pdf_url = None

            tz = int(payload.get("tz_offset_hours", 3) or 3)
            country = str(payload.get("country", "TR") or "TR")
            windows = best_posting_windows(payload.get("platform", ""), tz_offset_hours=tz, country=country)

            result: Dict[str, Any] = {
                "duration_seconds": duration,
                "features": features,
                "scores": scores,
                "why": why_explanation,
                "viral": scores.get("overall_score", 0.0) >= viral_threshold,
                "mode": payload.get("mode", "FAST").upper(),
                "analysis_complete": True,
                "suggestions": suggestions,
                "pdf_url": pdf_url,
                "posting_window": windows,
            }

            complete_job(job_id, result)
            save_analysis(job_id, payload.get("platform", ""), "done", result=result)
            return result
    except Exception as e:
        tb = traceback.format_exc()
        fail_job(job_id, f"{e}\n{tb}")
        try:
            save_analysis(job_id, payload.get("platform", ""), "failed", error=str(e))
        except Exception:
            pass
        raise


