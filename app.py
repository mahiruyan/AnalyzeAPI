from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os

# Railway için ana uygulama dosyası
app = FastAPI(title="analyzeAPI", version="0.2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    platform: str = Field(..., pattern="^(instagram|tiktok)$")
    file_url: str = Field(..., description="Video URL to analyze")
    mode: Optional[str] = Field("FAST", description="Analysis mode: FAST or FULL")
    title: Optional[str] = ""
    caption: Optional[str] = ""
    tags: Optional[List[str]] = []
    email: Optional[str] = None
    media_id: Optional[str] = None
    published_at: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class AnalyzeResponse(BaseModel):
    score: float = Field(..., ge=0, le=100)
    verdict: str = Field(..., pattern="^(low|mid|high)$")
    suggestions: List[str]

# Hata yakalama middleware - detaylı hata mesajları
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    error_detail = str(exc)
    error_traceback = traceback.format_exc()
    
    # Detaylı hata döndür
    return {
        "error": f"Backend hatası: {error_detail}",
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
            "tips": [f"Sistem hatası: {error_detail}"]
        }
    }

@app.get("/healthz")
def health_check():
    """Railway sağlık kontrolü endpoint'i"""
    return {"status": "ok"}

@app.get("/api/health")
def api_health_check():
    """Railway API sağlık kontrolü endpoint'i"""
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
    """Keep-alive endpoint - Railway'i uyanık tutmak için"""
    return {"status": "awake", "timestamp": "2025-01-25"}

@app.get("/")
def root():
    """Ana sayfa"""
    return {"status": "ok", "name": "analyzeAPI", "version": "0.2.0"}

@app.post("/ingest_ig")
def ingest_instagram(data: InstagramIngest):
    """Instagram metriklerini al"""
    # Burada veritabanına kaydetme işlemi yapılabilir
    return {"status": "success", "media_id": data.media_id}

@app.post("/ingest_tt")
def ingest_tiktok(data: TikTokIngest):
    """TikTok metriklerini al"""
    # Burada veritabanına kaydetme işlemi yapılabilir
    return {"status": "success", "video_id": data.video_id}

@app.post("/api/analyze")
def analyze_performance_api(data: AnalyzeRequest):
    """Real video analysis - no mock responses"""
    
    try:
        print(f"[DEBUG] Request received: mode={data.mode}")
        print(f"[DEBUG] File URL: {data.file_url}")
        print(f"[DEBUG] Platform: {data.platform}")
        
        print("[DEBUG] Starting imports...")
        import tempfile
        import os
        from pathlib import Path
        print("[DEBUG] Basic imports OK")
        
        from utils import download_video, extract_audio_via_ffmpeg, grab_frames, get_video_duration
        print("[DEBUG] Utils import OK")
        
        from features import extract_features
        print("[DEBUG] Features import OK")
        
        from scoring import score_features
        print("[DEBUG] Scoring import OK")
        
        from suggest import generate_suggestions
        print("[DEBUG] Suggest import OK")
        
        from advanced_analysis import analyze_hook, analyze_pacing_retention, analyze_cta_interaction, analyze_message_clarity, analyze_technical_quality, analyze_transition_quality, analyze_loop_final, analyze_text_readability, analyze_trend_originality, analyze_content_type_suitability, analyze_music_sync, analyze_accessibility
        print("[DEBUG] Advanced analysis import OK")
        
    except Exception as e:
        print(f"[ERROR] Import failed: {str(e)}")
        print(f"[ERROR] Error type: {type(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return {
            "error": f"Import hatası: {str(e)}",
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
        print("[DEBUG] Starting FULL analysis...")
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "video.mp4")
            audio_path = os.path.join(temp_dir, "audio.wav")
            frames_dir = os.path.join(temp_dir, "frames")
            print(f"[DEBUG] Temp directory created: {temp_dir}")
            
            # Video indir - timeout ile
            print(f"🔽 [DEBUG] Downloading video from: {data.file_url}")
            try:
                download_video(data.file_url, video_path, timeout=30)  # 30 saniye timeout
                print(f"[DEBUG] Video downloaded to: {video_path}")
            except Exception as e:
                raise Exception(f"Video indirme başarısız (30s timeout): {e}")
            
            # Ses çıkar - hızlı
            print("🎵 [DEBUG] Extracting audio...")
            try:
                extract_audio_via_ffmpeg(video_path, audio_path, sample_rate=16000, mono=True)
                print(f"[DEBUG] Audio extracted to: {audio_path}")
            except Exception as e:
                raise Exception(f"Ses çıkarma başarısız: {e}")
            
            # Frame'ler çıkar - çok hızlı
            print("🖼️ [DEBUG] Extracting frames...")
            try:
                frames = grab_frames(video_path, frames_dir, max_frames=3)  # 3 frame'e düşürdük (hız için)
                print(f"[DEBUG] Frames extracted: {len(frames)} frames")
            except Exception as e:
                raise Exception(f"Frame çıkarma başarısız: {e}")
            
            # Süre al
            print("⏱️ [DEBUG] Getting video duration...")
            duration = get_video_duration(video_path)
            print(f"[DEBUG] Duration: {duration} seconds")
            
            # Özellikler çıkar
            print("🧠 [DEBUG] Extracting features...")
            features = extract_features(
                audio_path=audio_path,
                frames=frames,
                caption=data.caption or "",
                title=data.title or "",
                tags=data.tags or [],
                duration_seconds=duration,
                fast_mode=(data.mode == "FAST")
            )
            print("[DEBUG] Features extracted")
            
            # Eski skorlar hesapla
            print("[DEBUG] Calculating legacy scores...")
            legacy_scores = score_features(features, data.platform, (data.mode == "FAST"))
            print("[DEBUG] Legacy scores calculated")
            
            # YENİ: Hook analizi (hızlı versiyon - 3 frame)
            print("[DEBUG] Analyzing hook (first 3 seconds)...")
            try:
                hook_result = analyze_hook(video_path, audio_path, frames[:3], features, duration)
                print(f"[DEBUG] Hook analysis completed: {hook_result['score']}/18")
            except Exception as e:
                print(f"[DEBUG] Hook analysis failed: {e}")
                raise Exception(f"Hook analizi başarısız: {e}")
            
            # YENİ: Pacing analizi (hızlı versiyon - 3 frame)
            print("⚡ [DEBUG] Analyzing pacing & retention...")
            try:
                pacing_result = analyze_pacing_retention(frames, features, duration)
                print(f"[DEBUG] Pacing analysis completed: {pacing_result['score']}/12")
            except Exception as e:
                print(f"[DEBUG] Pacing analysis failed: {e}")
                raise Exception(f"Pacing analizi başarısız: {e}")
            
            # YENİ: CTA & Etkileşim analizi
            print("📢 [DEBUG] Analyzing CTA & interaction...")
            try:
                cta_result = analyze_cta_interaction(features, duration)
                print(f"[DEBUG] CTA analysis completed: {cta_result['score']}/8")
            except Exception as e:
                print(f"[DEBUG] CTA analysis failed: {e}")
                raise Exception(f"CTA analizi başarısız: {e}")
            
        # YENİ: Mesaj netliği analizi
        print("[DEBUG] Analyzing message clarity...")
        try:
            message_result = analyze_message_clarity(features, duration)
            print(f"[DEBUG] Message analysis completed: {message_result['score']}/6")
        except Exception as e:
            print(f"[DEBUG] Message analysis failed: {e}")
            raise Exception(f"Mesaj analizi başarısız: {e}")
        
        # YENİ: Teknik kalite analizi
        print("[DEBUG] Analyzing technical quality...")
        try:
            tech_result = analyze_technical_quality(features, duration)
            print(f"[DEBUG] Technical analysis completed: {tech_result['score']}/15")
        except Exception as e:
            print(f"[DEBUG] Technical analysis failed: {e}")
            raise Exception(f"Teknik analiz başarısız: {e}")
        
        # YENİ: Geçiş kalitesi analizi
        print("[DEBUG] Analyzing transition quality...")
        try:
            transition_result = analyze_transition_quality(features, duration)
            print(f"[DEBUG] Transition analysis completed: {transition_result['score']}/8")
        except Exception as e:
            print(f"[DEBUG] Transition analysis failed: {e}")
            raise Exception(f"Geçiş analizi başarısız: {e}")
        
        # YENİ: Loop & final analizi
        print("[DEBUG] Analyzing loop & final...")
        try:
            loop_result = analyze_loop_final(features, duration)
            print(f"[DEBUG] Loop & final analysis completed: {loop_result['score']}/7")
        except Exception as e:
            print(f"[DEBUG] Loop & final analysis failed: {e}")
            raise Exception(f"Loop & final analizi başarısız: {e}")
        
        # YENİ: Metin okunabilirlik analizi
        print("[DEBUG] Analyzing text readability...")
        try:
            text_result = analyze_text_readability(features, duration)
            print(f"[DEBUG] Text readability analysis completed: {text_result['score']}/6")
        except Exception as e:
            print(f"[DEBUG] Text readability analysis failed: {e}")
            raise Exception(f"Metin okunabilirlik analizi başarısız: {e}")
        
        # YENİ: Trend & orijinallik analizi
        print("[DEBUG] Analyzing trend & originality...")
        try:
            trend_result = analyze_trend_originality(features, duration)
            print(f"[DEBUG] Trend & originality analysis completed: {trend_result['score']}/8")
        except Exception as e:
            print(f"[DEBUG] Trend & originality analysis failed: {e}")
            raise Exception(f"Trend & orijinallik analizi başarısız: {e}")
        
        # YENİ: İçerik tipi uygunluğu analizi
        print("[DEBUG] Analyzing content type suitability...")
        try:
            content_result = analyze_content_type_suitability(features, duration)
            print(f"[DEBUG] Content type analysis completed: {content_result['score']}/6")
        except Exception as e:
            print(f"[DEBUG] Content type analysis failed: {e}")
            raise Exception(f"İçerik tipi analizi başarısız: {e}")
        
        # YENİ: Müzik & senkron analizi
        print("[DEBUG] Analyzing music & sync...")
        try:
            music_result = analyze_music_sync(features, duration)
            print(f"[DEBUG] Music & sync analysis completed: {music_result['score']}/5")
        except Exception as e:
            print(f"[DEBUG] Music & sync analysis failed: {e}")
            raise Exception(f"Müzik & senkron analizi başarısız: {e}")
        
        # YENİ: Erişilebilirlik analizi
        print("[DEBUG] Analyzing accessibility...")
        try:
            access_result = analyze_accessibility(features, duration)
            print(f"[DEBUG] Accessibility analysis completed: {access_result['score']}/4")
        except Exception as e:
            print(f"[DEBUG] Accessibility analysis failed: {e}")
            raise Exception(f"Erişilebilirlik analizi başarısız: {e}")
        
        # Gelişmiş skorları birleştir
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
        
        # Toplam skor (tüm analizler dahil)
        total_score = (hook_result["score"] + pacing_result["score"] + 
                      cta_result["score"] + message_result["score"] + 
                      tech_result["score"] + transition_result["score"] + 
                      loop_result["score"] + text_result["score"] + 
                      trend_result["score"] + content_result["score"] + 
                      music_result["score"] + access_result["score"] + 
                      legacy_scores.get("overall_score", 0))
        
        advanced_scores["total_score"] = total_score
        advanced_scores["overall_score"] = total_score  # Frontend için
        
        # Legacy öneriler
        print("[DEBUG] Generating legacy suggestions...")
        legacy_suggestions = generate_suggestions(
            platform=data.platform,
            caption=data.caption or "",
            title=data.title or "",
            tags=data.tags or [],
            features=features,
            scores=legacy_scores
        )
        
        # Advanced önerileri birleştir
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
        all_suggestions.extend(legacy_suggestions.get("tips", []))
        
        # Findings birleştir
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
        
        print("[DEBUG] Advanced suggestions generated")
        print(f"[DEBUG] Total advanced score: {total_score}")
        
        print("[DEBUG] Analysis completed successfully!")
        return {
            "duration_seconds": duration,
            "features": features,
            "scores": advanced_scores,
            "legacy_scores": legacy_scores,
            "verdict": "high" if total_score >= 70 else "mid" if total_score >= 40 else "low",
            "viral": total_score >= 70,
            "mode": data.mode,
            "analysis_complete": True,
            "suggestions": all_suggestions[:10],  # En önemli 10 öneri
            "findings": all_findings,  # Zaman kodlu bulgular
            "hook_analysis": hook_result,  # Detaylı hook analizi
            "pacing_analysis": pacing_result,  # Detaylı pacing analizi
            "cta_analysis": cta_result,  # Detaylı CTA analizi
            "message_analysis": message_result,  # Detaylı mesaj analizi
            "technical_analysis": tech_result,  # Detaylı teknik analiz
            "transition_analysis": transition_result,  # Detaylı geçiş analizi
            "loop_final_analysis": loop_result,  # Detaylı loop & final analizi
            "text_readability_analysis": text_result,  # Detaylı metin okunabilirlik analizi
            "trend_originality_analysis": trend_result,  # Detaylı trend & orijinallik analizi
            "content_type_suitability_analysis": content_result,  # Detaylı içerik tipi uygunluğu analizi
            "music_sync_analysis": music_result,  # Detaylı müzik & senkron analizi
            "accessibility_analysis": access_result,  # Detaylı erişilebilirlik analizi
            "nlp_analysis": message_result.get("raw", {}),  # NLP detayları
            "emotion_analysis": message_result.get("raw", {}).get("emotions", {}),  # Duygu analizi
            "intent_analysis": message_result.get("raw", {}).get("intent", "unknown"),  # Amaç analizi
            "readability_score": message_result.get("raw", {}).get("readability_score", 0.5),  # Okunabilirlik
            "engagement_potential": message_result.get("raw", {}).get("engagement_potential", 0.5),  # Etkileşim potansiyeli
            "viral_keywords": message_result.get("raw", {}).get("viral_keywords", []),  # Viral kelimeler
            "language_quality": message_result.get("raw", {}).get("language_quality", 0.5),  # Dil kalitesi
            "score": total_score,  # Frontend için direkt score
            "overall_score": total_score  # Frontend için
        }
    except Exception as e:
        print(f"[ERROR] Analysis failed: {str(e)}")
        print(f"[ERROR] Error type: {type(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return {
            "error": f"Analiz hatası: {str(e)}",
            "duration_seconds": 0,
            "features": {},
            "scores": {},
            "verdict": "error", 
            "viral": False,
            "mode": data.mode or "UNKNOWN",
            "analysis_complete": False,
            "suggestions": [f"Video analiz hatası: {str(e)}"],
            "score": 0,  # Frontend için
            "overall_score": 0  # Frontend için
        }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_performance(data: AnalyzeRequest):
    """Legacy endpoint - redirects to /api/analyze"""
    # Gerçek analiz için /api/analyze'i çağır
    result = analyze_performance_api(data)
    
    return AnalyzeResponse(
        score=result.get("scores", {}).get("overall_score", 0),
        verdict=result.get("verdict", "mid"),
        suggestions=result.get("suggestions", [])  # Artık direkt liste
    )

# main.py import'u kaldırıldı - circular import sorunu yaratıyordu

if __name__ == "__main__":
    import uvicorn
    # Railway PORT environment variable'ını kontrol et
    port_str = os.getenv("PORT")
    if port_str:
        try:
            port = int(port_str)
            print(f"Using Railway PORT: {port}")
        except ValueError:
            port = 8000
            print(f"Invalid PORT value: {port_str}, using default: {port}")
    else:
        port = 8000
        print(f"No PORT env var, using default: {port}")
    
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
