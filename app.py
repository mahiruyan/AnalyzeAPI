
def safe_print(*args, **kwargs):
    """Safe print function that handles Unicode errors"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Replace problematic characters and try again
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_arg = arg.encode('ascii', 'replace').decode('ascii')
                safe_args.append(safe_arg)
            else:
                safe_args.append(arg)
        print(*safe_args, **kwargs)


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os
import json
import hashlib

from analytics import AudienceSentimentAnalyzer
from audience import (
    AudienceEngagementMetrics,
    AudienceSentimentSummary,
    CommentThread,
)
from scraper import (
    InstagramScraper,
    ScrapeError,
    ScraperConfig,
    TikTokScraper,
)
from redis_client import RedisNotConfigured, try_get_redis

# Railway iin ana uygulama dosyas
app = FastAPI(title="analyzeAPI", version="0.2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Runtime warmup kaldırıldı - modeller Docker build sırasında preload ediliyor
# Bu sayede container her başladığında modeller zaten hazır ve timeout riski yok

# Pydantic modelleri
class InstagramMetrics(BaseModel):
    plays: int = Field(..., ge=0)
    reach: int = Field(..., ge=0)
    likes: int = Field(..., ge=0)
    comments: int = Field(..., ge=0)
    shares: int = Field(..., ge=0)
    saves: int = Field(..., ge=0)
    avg_watch_time_s: Optional[float] = Field(None, ge=0)
    completion_rate: Optional[float] = Field(None, ge=0, le=1)

class TikTokMetrics(BaseModel):
    views: int = Field(..., ge=0)
    likes: int = Field(..., ge=0)
    comments: int = Field(..., ge=0)
    shares: int = Field(..., ge=0)
    avg_watch_time_s: Optional[float] = Field(None, ge=0)
    completion_rate: Optional[float] = Field(None, ge=0, le=1)

class InstagramIngest(BaseModel):
    media_id: str
    published_at: str
    metrics: InstagramMetrics

class TikTokIngest(BaseModel):
    video_id: str
    published_at: str
    metrics: TikTokMetrics

class AnalyzeRequest(BaseModel):
    platform: str = Field(..., pattern="^(instagram|tiktok|youtube)$")
    file_url: str = Field(..., description="Video URL to analyze")
    mode: Optional[str] = Field("FULL", description="Analysis mode: FULL (fast mode removed)")
    title: Optional[str] = ""
    caption: Optional[str] = ""
    tags: Optional[List[str]] = []
    email: Optional[str] = None
    media_id: Optional[str] = None
    published_at: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    engagement_metrics: Optional[AudienceEngagementMetrics] = None
    include_comments: bool = False
    comment_threads: Optional[List[CommentThread]] = None
    scraper_cookies: Optional[Dict[str, str]] = None
    scraper_user_agent: Optional[str] = None
    max_comments: Optional[int] = Field(200, ge=1, le=500)
    scraper_cookie_path: Optional[str] = None

class AnalyzeResponse(BaseModel):
    score: float = Field(..., ge=0, le=100)
    verdict: str = Field(..., pattern="^(low|mid|high|error)$")
    suggestions: List[str]


def _json_default(value):
    """Redis cache için JSON serileştirici yardımcı."""

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

# Hata yakalama middleware - detayl hata mesajlar
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    error_detail = str(exc)
    error_traceback = traceback.format_exc()
    
    # Detayl hata dndr
    return {
        "error": f"Backend hatas: {error_detail}",
        "error_type": type(exc).__name__,
        "traceback": error_traceback[-1000:],  # Son 1000 karakter
        "duration_seconds": 0,
        "features": {},
        "scores": {},
        "verdict": "error",
        "viral": False,
        "mode": "ERROR",
        "analysis_complete": False,
        "suggestions": {
            "tips": [f"Sistem hatas: {error_detail}"]
        }
    }

@app.get("/healthz")
def health_check():
    """Railway salk kontrol endpoint'i"""
    return {"status": "ok"}

@app.get("/api/health")
def api_health_check():
    """Railway API salk kontrol endpoint'i"""
    return {"status": "ok"}

@app.get("/api/test")
def test_dependencies():
    """Dependencies test endpoint"""
    results = {}
    
    try:
        import tempfile
        results["tempfile"] = "OK"
    except Exception as e:
        results["tempfile"] = f"ERROR: {str(e)}"
    
    try:
        from utils import download_video
        results["utils"] = "OK"
    except Exception as e:
        results["utils"] = f"ERROR: {str(e)}"
    
    try:
        import cv2
        results["opencv"] = "OK"
    except Exception as e:
        results["opencv"] = f"ERROR: {str(e)}"
    
    try:
        import librosa
        results["librosa"] = "OK"
    except Exception as e:
        results["librosa"] = f"ERROR: {str(e)}"
    
    try:
        from faster_whisper import WhisperModel
        results["whisper"] = "OK"
    except Exception as e:
        results["whisper"] = f"ERROR: {str(e)}"
    
    try:
        import subprocess
        ffmpeg_result = subprocess.run(['ffmpeg', '-version'], 
                                     capture_output=True, text=True, timeout=5)
        if ffmpeg_result.returncode == 0:
            results["ffmpeg"] = "OK"
        else:
            results["ffmpeg"] = "NOT_FOUND"
    except Exception as e:
        results["ffmpeg"] = f"ERROR: {str(e)}"
    
    return {"dependencies": results}

@app.get("/api/ping")
def ping():
    """Keep-alive endpoint - Railway'i uyank tutmak iin"""
    return {"status": "awake", "timestamp": "2025-01-25"}

@app.get("/")
def root():
    """Ana sayfa"""
    return {"status": "ok", "name": "analyzeAPI", "version": "0.2.0"}

@app.post("/ingest_ig")
def ingest_instagram(data: InstagramIngest):
    """Instagram metriklerini al"""
    # Burada veritabanna kaydetme ilemi yaplabilir
    return {"status": "success", "media_id": data.media_id}

@app.post("/ingest_tt")
def ingest_tiktok(data: TikTokIngest):
    """TikTok metriklerini al"""
    # Burada veritabanna kaydetme ilemi yaplabilir
    return {"status": "success", "video_id": data.video_id}

@app.post("/api/analyze")
def analyze_performance_api(data: AnalyzeRequest):
    """Real video analysis - no mock responses"""
    
    try:
        safe_print(f"[DEBUG] Request received: mode={data.mode}")
        safe_print(f"[DEBUG] File URL: {data.file_url}")
        safe_print(f"[DEBUG] Platform: {data.platform}")
        
        redis_client = None
        cache_key = None

        if not data.include_comments and not data.comment_threads:
            try:
                redis_client = try_get_redis()
            except RedisNotConfigured:
                redis_client = None
            except Exception as redis_error:
                safe_print(f"[REDIS] Connection error: {redis_error}")
                redis_client = None

        if redis_client:
            signature = {
                "platform": data.platform,
                "file_url": data.file_url,
                "mode": data.mode or "FULL",
                "title": data.title or "",
                "caption": data.caption or "",
                "tags": data.tags or [],
                "metrics": data.metrics or {},
                "engagement_metrics": data.engagement_metrics.model_dump()
                if data.engagement_metrics
                else None,
            }
            serialized_signature = json.dumps(
                signature, sort_keys=True, separators=(",", ":"), default=_json_default
            )
            cache_key = f"analyze:{hashlib.sha256(serialized_signature.encode('utf-8')).hexdigest()}"
            try:
                cached_response = redis_client.get(cache_key)
                if cached_response:
                    safe_print(f"[REDIS] Cache hit for key {cache_key}")
                    return json.loads(cached_response)
            except Exception as redis_error:
                safe_print(f"[REDIS] Cache fetch error: {redis_error}")

        audience_summary: Optional[AudienceSentimentSummary] = None
        comment_threads: List[CommentThread] = []

        if data.include_comments:
            safe_print("[DEBUG] Audience comments requested")
            if data.comment_threads:
                safe_print(
                    f"[DEBUG] Using {len(data.comment_threads)} provided comment threads"
                )
                comment_threads = data.comment_threads
            else:
                safe_print("[DEBUG] Starting scraping pipeline for comments")
                default_cookie_paths = {
                    "instagram": os.getenv("INSTAGRAM_COOKIE_JAR"),
                    "tiktok": os.getenv("TIKTOK_COOKIE_JAR"),
                }
                config = ScraperConfig(
                    user_agent=data.scraper_user_agent
                    or ScraperConfig().user_agent,
                    cookies=data.scraper_cookies,
                    cookie_store_path=data.scraper_cookie_path
                    or default_cookie_paths.get(data.platform),
                )
                try:
                    if data.platform == "instagram":
                        scraper = InstagramScraper(config)
                        comment_threads = scraper.fetch_comments(
                            data.file_url, max_comments=data.max_comments or 200
                        )
                    elif data.platform == "tiktok":
                        scraper = TikTokScraper(config)
                        comment_threads = scraper.fetch_comments(
                            data.file_url, max_comments=data.max_comments or 200
                        )
                    else:
                        raise ScrapeError(
                            "Bu platform için scraper implementasyonu yok"
                        )
                except ScrapeError as scrape_error:
                    safe_print(f"[AUDIENCE] Comment scraping failed: {scrape_error}")
                    raise

            if comment_threads:
                safe_print(
                    f"[DEBUG] Running sentiment analysis on {sum(len(t.replies) + 1 for t in comment_threads)} comments"
                )
                sentiment_analyzer = AudienceSentimentAnalyzer()
                audience_summary = sentiment_analyzer.analyze(comment_threads)
            else:
                safe_print("[DEBUG] No comments available for sentiment analysis")

        safe_print("[DEBUG] Starting imports...")
        import tempfile
        import os
        from pathlib import Path
        safe_print("[DEBUG] Basic imports OK")
        
        from utils import download_video, extract_audio_via_ffmpeg, grab_frames, get_video_duration
        safe_print("[DEBUG] Utils import OK")
        
        from features import extract_features
        safe_print("[DEBUG] Features import OK")
        
        from scoring import score_features
        safe_print("[DEBUG] Scoring import OK")
        
        from suggest import generate_suggestions
        safe_print("[DEBUG] Suggest import OK")
        
        from advanced_analysis import analyze_hook, analyze_pacing_retention, analyze_cta_interaction, analyze_message_clarity, analyze_technical_quality, analyze_transition_quality, analyze_loop_final, analyze_text_readability, analyze_trend_originality, analyze_content_type_suitability, analyze_music_sync, analyze_accessibility
        safe_print("[DEBUG] Advanced analysis import OK")
        
    except Exception as e:
        safe_print(f"[ERROR] Import failed: {str(e)}")
        safe_print(f"[ERROR] Error type: {type(e)}")
        import traceback
        try:
            safe_print(f"[ERROR] Traceback: {traceback.format_exc()}")
        except UnicodeEncodeError:
            safe_print("[ERROR] Traceback: (Unicode error in traceback)")
        return {
            "error": f"Import hatas: {str(e)}",
            "duration_seconds": 0,
            "features": {},
            "scores": {},
            "verdict": "error",
            "viral": False,
            "mode": data.mode or "UNKNOWN",
            "analysis_complete": False,
            "suggestions": [f"Hata: {str(e)}"]
        }
    
    # Video indirme ve analiz
    try:
        safe_print("[DEBUG] Starting FULL analysis...")
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "video.mp4")
            audio_path = os.path.join(temp_dir, "audio.wav")
            frames_dir = os.path.join(temp_dir, "frames")
            safe_print(f"[DEBUG] Temp directory created: {temp_dir}")
            
            # Video indir - timeout ile
            safe_print(f"[DEBUG] Downloading video from: {data.file_url}")
            try:
                download_video(data.file_url, video_path, timeout=30)  # 30 saniye timeout
                safe_print(f"[DEBUG] Video downloaded to: {video_path}")
            except Exception as e:
                raise Exception(f"Video indirme baarsz (30s timeout): {e}")
            
            # Ses kar - hzl
            safe_print(" [DEBUG] Extracting audio...")
            audio_has_stream = True
            try:
                audio_has_stream = extract_audio_via_ffmpeg(video_path, audio_path, sample_rate=16000, mono=True)
                if audio_has_stream:
                    safe_print(f"[DEBUG] Audio extracted to: {audio_path}")
                else:
                    safe_print("[DEBUG] Video sessiz; placeholder audio üretildi.")
            except Exception as e:
                raise Exception(f"Ses karma baarsz: {e}")
            
            # Frame'ler kar - ok hzl
            safe_print(" [DEBUG] Extracting frames...")
            try:
                frames = grab_frames(video_path, frames_dir, max_frames=10)  # 10 frame - FULL analiz
                safe_print(f"[DEBUG] Frames extracted: {len(frames)} frames")
            except Exception as e:
                raise Exception(f"Frame karma baarsz: {e}")
            
            # Sre al
            safe_print(" [DEBUG] Getting video duration...")
            duration = get_video_duration(video_path)
            safe_print(f"[DEBUG] Duration: {duration} seconds")
            
            # zellikler kar
            safe_print(" [DEBUG] Extracting features...")
            features = extract_features(
                audio_path=audio_path,
                frames=frames,
                caption=data.caption or "",
                title=data.title or "",
                tags=data.tags or [],
                duration_seconds=duration,
                fast_mode=False  # Her zaman FULL mode - fast mode sikik
            )
            if not audio_has_stream:
                features.setdefault("issues", []).append("Video sessiz olduğu için ses analizi sınırlı yapılabildi.")
            safe_print("[DEBUG] Features extracted")
            
            # Eski skorlar hesapla
            safe_print("[DEBUG] Calculating legacy scores...")
            legacy_scores = score_features(features, data.platform, False)  # Her zaman FULL mode
            safe_print("[DEBUG] Legacy scores calculated")
            
            # YEN: Hook analizi (hzl versiyon - 3 frame)
            safe_print("[DEBUG] Analyzing hook (first 3 seconds)...")
            try:
                hook_result = analyze_hook(video_path, audio_path, frames[:3], features, duration)
                safe_print(f"[DEBUG] Hook analysis completed: {hook_result['score']}/18")
            except Exception as e:
                safe_print(f"[DEBUG] Hook analysis failed: {e}")
                raise Exception(f"Hook analizi baarsz: {e}")
            
            # YEN: Pacing analizi (hzl versiyon - 3 frame)
            safe_print(" [DEBUG] Analyzing pacing & retention...")
            try:
                pacing_result = analyze_pacing_retention(frames, features, duration)
                safe_print(f"[DEBUG] Pacing analysis completed: {pacing_result['score']}/12")
            except Exception as e:
                safe_print(f"[DEBUG] Pacing analysis failed: {e}")
                raise Exception(f"Pacing analizi baarsz: {e}")
            
            # YEN: CTA & Etkileim analizi
            safe_print(" [DEBUG] Analyzing CTA & interaction...")
            try:
                cta_result = analyze_cta_interaction(features, duration)
                safe_print(f"[DEBUG] CTA analysis completed: {cta_result['score']}/8")
            except Exception as e:
                safe_print(f"[DEBUG] CTA analysis failed: {e}")
                raise Exception(f"CTA analizi baarsz: {e}")
            
        # YEN: Mesaj netlii analizi
        safe_print("[DEBUG] Analyzing message clarity...")
        try:
            message_result = analyze_message_clarity(features, duration)
            safe_print(f"[DEBUG] Message analysis completed: {message_result['score']}/6")
        except Exception as e:
            safe_print(f"[DEBUG] Message analysis failed: {e}")
            raise Exception(f"Mesaj analizi baarsz: {e}")
        
        # YEN: Teknik kalite analizi
        safe_print("[DEBUG] Analyzing technical quality...")
        try:
            tech_result = analyze_technical_quality(features, duration)
            safe_print(f"[DEBUG] Technical analysis completed: {tech_result['score']}/15")
        except Exception as e:
            safe_print(f"[DEBUG] Technical analysis failed: {e}")
            raise Exception(f"Teknik analiz baarsz: {e}")
        
        # YEN: Gei kalitesi analizi
        safe_print("[DEBUG] Analyzing transition quality...")
        try:
            transition_result = analyze_transition_quality(features, duration)
            safe_print(f"[DEBUG] Transition analysis completed: {transition_result['score']}/8")
        except Exception as e:
            safe_print(f"[DEBUG] Transition analysis failed: {e}")
            raise Exception(f"Gei analizi baarsz: {e}")
        
        # YEN: Loop & final analizi
        safe_print("[DEBUG] Analyzing loop & final...")
        try:
            loop_result = analyze_loop_final(features, duration)
            safe_print(f"[DEBUG] Loop & final analysis completed: {loop_result['score']}/7")
        except Exception as e:
            safe_print(f"[DEBUG] Loop & final analysis failed: {e}")
            raise Exception(f"Loop & final analizi baarsz: {e}")
        
        # YEN: Metin okunabilirlik analizi
        safe_print("[DEBUG] Analyzing text readability...")
        try:
            text_result = analyze_text_readability(features, duration)
            safe_print(f"[DEBUG] Text readability analysis completed: {text_result['score']}/6")
        except Exception as e:
            safe_print(f"[DEBUG] Text readability analysis failed: {e}")
            raise Exception(f"Metin okunabilirlik analizi baarsz: {e}")
        
        # YEN: Trend & orijinallik analizi
        safe_print("[DEBUG] Analyzing trend & originality...")
        try:
            trend_result = analyze_trend_originality(features, duration)
            safe_print(f"[DEBUG] Trend & originality analysis completed: {trend_result['score']}/8")
        except Exception as e:
            safe_print(f"[DEBUG] Trend & originality analysis failed: {e}")
            raise Exception(f"Trend & orijinallik analizi baarsz: {e}")
        
        # YEN: erik tipi uygunluu analizi
        safe_print("[DEBUG] Analyzing content type suitability...")
        try:
            content_result = analyze_content_type_suitability(features, duration)
            safe_print(f"[DEBUG] Content type analysis completed: {content_result['score']}/6")
        except Exception as e:
            safe_print(f"[DEBUG] Content type analysis failed: {e}")
            raise Exception(f"erik tipi analizi baarsz: {e}")
        
        # YEN: Mzik & senkron analizi
        safe_print("[DEBUG] Analyzing music & sync...")
        try:
            music_result = analyze_music_sync(features, duration)
            safe_print(f"[DEBUG] Music & sync analysis completed: {music_result['score']}/5")
        except Exception as e:
            safe_print(f"[DEBUG] Music & sync analysis failed: {e}")
            raise Exception(f"Mzik & senkron analizi baarsz: {e}")
        
        # YEN: Eriilebilirlik analizi
        safe_print("[DEBUG] Analyzing accessibility...")
        try:
            access_result = analyze_accessibility(features, duration)
            safe_print(f"[DEBUG] Accessibility analysis completed: {access_result['score']}/4")
        except Exception as e:
            safe_print(f"[DEBUG] Accessibility analysis failed: {e}")
            raise Exception(f"Eriilebilirlik analizi baarsz: {e}")
        
        # Gelimi skorlar birletir
        advanced_scores = {
            "hook_score": hook_result["score"],
            "pacing_score": pacing_result["score"],
            "cta_score": cta_result["score"],
            "message_clarity_score": message_result["score"],
            "technical_quality_score": tech_result["score"],
            "transition_quality_score": transition_result["score"],
            "loop_final_score": loop_result["score"],
            "text_readability_score": text_result["score"],
            "trend_originality_score": trend_result["score"],
            "content_type_suitability_score": content_result["score"],
            "music_sync_score": music_result["score"],
            "accessibility_score": access_result["score"],
            **legacy_scores  # Eski skorlar da dahil
        }
        
        audience_result = {
            "score": 0.0,
            "max_score": 10.0,
            "recommendations": [],
            "findings": [],
            "summary": audience_summary.model_dump() if audience_summary else None,
            "engagement": data.engagement_metrics.model_dump()
            if data.engagement_metrics
            else None,
        }

        if audience_summary:
            sentiment_balance = audience_summary.positive_ratio - audience_summary.negative_ratio
            sentiment_score = 5.0 + sentiment_balance * 10.0
            sentiment_score = max(0.0, min(10.0, sentiment_score))
            if audience_summary.negative_ratio >= 0.3:
                audience_result["recommendations"].append(
                    "Yorumlarda belirgin negatif duygu var; krizin kaynak olduğu içeriklere hızlıca yanıt ver."
                )
            if audience_summary.positive_ratio >= 0.6:
                audience_result["findings"].append(
                    "Topluluk genel olarak içeriği olumlu karşılamış."
                )
        else:
            sentiment_score = 0.0

        engagement_component = 0.0
        engagement_data = data.engagement_metrics.model_dump() if data.engagement_metrics else {}
        denominator = (
            engagement_data.get("views")
            or engagement_data.get("plays")
            or engagement_data.get("impressions")
            or 0
        )
        if denominator and engagement_data.get("likes") is not None:
            like_rate = engagement_data["likes"] / max(1, denominator)
            engagement_component += min(1.0, like_rate * 5) * 4  # 0-4 puan
            if like_rate < 0.03:
                audience_result["recommendations"].append(
                    "Like oranı düşük; CTA veya thumbnail üzerinde optimizasyon yap."
                )
            elif like_rate > 0.1:
                audience_result["findings"].append(
                    "Like oranı güçlü; içerik hedef kitleyle rezonans yakalamış."
                )

        if denominator and engagement_data.get("comments") is not None:
            comment_rate = engagement_data["comments"] / max(1, denominator)
            engagement_component += min(1.0, comment_rate * 20) * 3  # 0-3 puan
            if comment_rate < 0.01:
                audience_result["recommendations"].append(
                    "Yorum etkileşimi sınırlı; videoda açık uçlu soru veya CTA ekle."
                )
            elif comment_rate > 0.03:
                audience_result["findings"].append(
                    "Yorum etkileşimi yüksek; topluluğun katılımı güçlü."
                )

        if denominator and engagement_data.get("shares") is not None:
            share_rate = engagement_data["shares"] / max(1, denominator)
            engagement_component += min(1.0, share_rate * 50) * 2  # 0-2 puan
            if share_rate < 0.005:
                audience_result["recommendations"].append(
                    "Paylaşım oranı düşük; viral potansiyeli artırmak için paylaşılabilir kancalar ekle."
                )
            elif share_rate > 0.02:
                audience_result["findings"].append(
                    "Paylaşım oranı yüksek; içerik organik yayılıyor."
                )

        if denominator and engagement_data.get("saves") is not None:
            save_rate = engagement_data["saves"] / max(1, denominator)
            engagement_component += min(1.0, save_rate * 20) * 1  # 0-1 puan
            if save_rate < 0.01:
                audience_result["recommendations"].append(
                    "Kaydetme oranı düşük; değerli ipuçlarını daha belirgin ver."
                )
            elif save_rate > 0.05:
                audience_result["findings"].append(
                    "Kaydetme oranı yüksek; içerik referans olarak tutuluyor."
                )

        engagement_component = min(5.0, engagement_component)

        audience_total = round(sentiment_score * 0.6 + engagement_component * 0.4, 2)
        audience_result["score"] = audience_total
        advanced_scores["audience_sentiment_score"] = audience_total

        # Toplam skor (tm analizler dahil)
        total_score = (hook_result["score"] + pacing_result["score"] + 
                      cta_result["score"] + message_result["score"] + 
                      tech_result["score"] + transition_result["score"] + 
                      loop_result["score"] + text_result["score"] + 
                      trend_result["score"] + content_result["score"] + 
                      music_result["score"] + access_result["score"] + 
                      audience_total +
                      legacy_scores.get("overall_score", 0))
        
        advanced_scores["total_score"] = total_score
        advanced_scores["overall_score"] = total_score  # Frontend iin
        
        # Legacy neriler
        safe_print("[DEBUG] Generating legacy suggestions...")
        legacy_suggestions = generate_suggestions(
            platform=data.platform,
            caption=data.caption or "",
            title=data.title or "",
            tags=data.tags or [],
            features=features,
            scores=legacy_scores
        )
        
        # Advanced nerileri birletir
        all_suggestions = []
        all_suggestions.extend(hook_result.get("recommendations", []))
        all_suggestions.extend(pacing_result.get("recommendations", []))
        all_suggestions.extend(cta_result.get("recommendations", []))
        all_suggestions.extend(message_result.get("recommendations", []))
        all_suggestions.extend(tech_result.get("recommendations", []))
        all_suggestions.extend(transition_result.get("recommendations", []))
        all_suggestions.extend(loop_result.get("recommendations", []))
        all_suggestions.extend(text_result.get("recommendations", []))
        all_suggestions.extend(trend_result.get("recommendations", []))
        all_suggestions.extend(content_result.get("recommendations", []))
        all_suggestions.extend(music_result.get("recommendations", []))
        all_suggestions.extend(access_result.get("recommendations", []))
        all_suggestions.extend(audience_result.get("recommendations", []))
        all_suggestions.extend(legacy_suggestions.get("tips", []))
        
        # Findings birletir
        all_findings = []
        all_findings.extend(hook_result.get("findings", []))
        all_findings.extend(pacing_result.get("findings", []))
        all_findings.extend(cta_result.get("findings", []))
        all_findings.extend(message_result.get("findings", []))
        all_findings.extend(tech_result.get("findings", []))
        all_findings.extend(transition_result.get("findings", []))
        all_findings.extend(loop_result.get("findings", []))
        all_findings.extend(text_result.get("findings", []))
        all_findings.extend(trend_result.get("findings", []))
        all_findings.extend(content_result.get("findings", []))
        all_findings.extend(music_result.get("findings", []))
        all_findings.extend(access_result.get("findings", []))
        all_findings.extend(audience_result.get("findings", []))
        
        safe_print("[DEBUG] Advanced suggestions generated")
        safe_print(f"[DEBUG] Total advanced score: {total_score}")
        
        safe_print("[DEBUG] Analysis completed successfully!")

        audience_payload = None
        if audience_summary:
            audience_payload = audience_summary.model_dump()
            audience_payload["comment_threads"] = [
                {
                    "parent": thread.parent.model_dump(),
                    "replies": [reply.model_dump() for reply in thread.replies],
                }
                for thread in comment_threads
            ]

        response_payload = {
            "duration_seconds": duration,
            "features": features,
            "scores": advanced_scores,
            "legacy_scores": legacy_scores,
            "verdict": "high" if total_score >= 70 else "mid" if total_score >= 40 else "low",
            "viral": total_score >= 70,
            "mode": data.mode,
            "analysis_complete": True,
            "suggestions": all_suggestions[:10],  # En nemli 10 neri
            "findings": all_findings,  # Zaman kodlu bulgular
            "hook_analysis": hook_result,  # Detayl hook analizi
            "pacing_analysis": pacing_result,  # Detayl pacing analizi
            "cta_analysis": cta_result,  # Detayl CTA analizi
            "message_analysis": message_result,  # Detayl mesaj analizi
            "technical_analysis": tech_result,  # Detayl teknik analiz
            "transition_analysis": transition_result,  # Detayl gei analizi
            "loop_final_analysis": loop_result,  # Detayl loop & final analizi
            "text_readability_analysis": text_result,  # Detayl metin okunabilirlik analizi
            "trend_originality_analysis": trend_result,  # Detayl trend & orijinallik analizi
            "content_type_suitability_analysis": content_result,  # Detayl ierik tipi uygunluu analizi
            "music_sync_analysis": music_result,  # Detayl mzik & senkron analizi
            "accessibility_analysis": access_result,  # Detayl eriilebilirlik analizi
            "nlp_analysis": message_result.get("raw", {}),  # NLP detaylar
            "emotion_analysis": message_result.get("raw", {}).get("emotions", {}),  # Duygu analizi
            "intent_analysis": message_result.get("raw", {}).get("intent", "unknown"),  # Ama analizi
            "readability_score": message_result.get("raw", {}).get("readability_score", 0.5),  # Okunabilirlik
            "engagement_potential": message_result.get("raw", {}).get("engagement_potential", 0.5),  # Etkileim potansiyeli
            "viral_keywords": message_result.get("raw", {}).get("viral_keywords", []),  # Viral kelimeler
            "language_quality": message_result.get("raw", {}).get("language_quality", 0.5),  # Dil kalitesi
            "score": total_score,  # Frontend iin direkt score
            "overall_score": total_score  # Frontend iin
            ,
            "audience_sentiment": audience_payload,
            "audience_insights": audience_result,
            "engagement_metrics": data.engagement_metrics.model_dump()
            if data.engagement_metrics
            else None,
        }
        if redis_client and cache_key:
            try:
                redis_client.setex(
                    cache_key, 300, json.dumps(response_payload, default=_json_default)
                )
                safe_print(f"[REDIS] Cache stored for key {cache_key}")
            except Exception as redis_error:
                safe_print(f"[REDIS] Cache store error: {redis_error}")
        return response_payload
    except Exception as e:
        safe_print(f"[ERROR] Analysis failed: {str(e)}")
        safe_print(f"[ERROR] Error type: {type(e)}")
        import traceback
        try:
            safe_print(f"[ERROR] Traceback: {traceback.format_exc()}")
        except UnicodeEncodeError:
            safe_print("[ERROR] Traceback: (Unicode error in traceback)")
        return {
            "error": f"Analiz hatas: {str(e)}",
            "duration_seconds": 0,
            "features": {},
            "scores": {},
            "verdict": "error", 
            "viral": False,
            "mode": data.mode or "UNKNOWN",
            "analysis_complete": False,
            "suggestions": [f"Video analiz hatas: {str(e)}"],
            "score": 0,  # Frontend iin
            "overall_score": 0  # Frontend iin
        }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_performance(data: AnalyzeRequest):
    """Legacy endpoint - redirects to /api/analyze"""
    # Gerek analiz iin /api/analyze'i ar
    result = analyze_performance_api(data)
    
    return AnalyzeResponse(
        score=result.get("scores", {}).get("overall_score", 0),
        verdict=result.get("verdict", "mid"),
        suggestions=result.get("suggestions", [])  # Artk direkt liste
    )

# main.py import'u kaldrld - circular import sorunu yaratyordu

if __name__ == "__main__":
    import uvicorn
    # Railway PORT environment variable'n kontrol et
    port_str = os.getenv("PORT")
    if port_str:
        try:
            port = int(port_str)
            safe_print(f"Using Railway PORT: {port}")
        except ValueError:
            port = 8000
            safe_print(f"Invalid PORT value: {port_str}, using default: {port}")
    else:
        port = 8000
        safe_print(f"No PORT env var, using default: {port}")
    
    safe_print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
