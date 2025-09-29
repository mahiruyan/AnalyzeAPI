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
        print(f"🔍 [DEBUG] Request received: mode={data.mode}")
        print(f"🔍 [DEBUG] File URL: {data.file_url}")
        print(f"🔍 [DEBUG] Platform: {data.platform}")
        
        print("🚀 [DEBUG] Starting imports...")
        import tempfile
        import os
        from pathlib import Path
        print("✅ [DEBUG] Basic imports OK")
        
        from utils import download_video, extract_audio_via_ffmpeg, grab_frames, get_video_duration
        print("✅ [DEBUG] Utils import OK")
        
        from features import extract_features
        print("✅ [DEBUG] Features import OK")
        
        from scoring import score_features
        print("✅ [DEBUG] Scoring import OK")
        
        from suggest import generate_suggestions
        print("✅ [DEBUG] Suggest import OK")
        
        from advanced_analysis import analyze_hook, analyze_pacing_retention
        print("✅ [DEBUG] Advanced analysis import OK")
        
    except Exception as e:
        print(f"❌ [ERROR] Import failed: {str(e)}")
        print(f"❌ [ERROR] Error type: {type(e)}")
        import traceback
        print(f"❌ [ERROR] Traceback: {traceback.format_exc()}")
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
        print("🚀 [DEBUG] Starting FULL analysis...")
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "video.mp4")
            audio_path = os.path.join(temp_dir, "audio.wav")
            frames_dir = os.path.join(temp_dir, "frames")
            print(f"✅ [DEBUG] Temp directory created: {temp_dir}")
            
            # Video indir - timeout ile
            print(f"🔽 [DEBUG] Downloading video from: {data.file_url}")
            try:
                download_video(data.file_url, video_path, timeout=30)  # 30 saniye timeout
                print(f"✅ [DEBUG] Video downloaded to: {video_path}")
            except Exception as e:
                raise Exception(f"Video indirme başarısız (30s timeout): {e}")
            
            # Ses çıkar - hızlı
            print("🎵 [DEBUG] Extracting audio...")
            try:
                extract_audio_via_ffmpeg(video_path, audio_path, sample_rate=16000, mono=True)
                print(f"✅ [DEBUG] Audio extracted to: {audio_path}")
            except Exception as e:
                raise Exception(f"Ses çıkarma başarısız: {e}")
            
            # Frame'ler çıkar - çok hızlı
            print("🖼️ [DEBUG] Extracting frames...")
            try:
                frames = grab_frames(video_path, frames_dir, max_frames=3)  # 3 frame'e düşürdük (hız için)
                print(f"✅ [DEBUG] Frames extracted: {len(frames)} frames")
            except Exception as e:
                raise Exception(f"Frame çıkarma başarısız: {e}")
            
            # Süre al
            print("⏱️ [DEBUG] Getting video duration...")
            duration = get_video_duration(video_path)
            print(f"✅ [DEBUG] Duration: {duration} seconds")
            
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
            print("✅ [DEBUG] Features extracted")
            
            # Eski skorlar hesapla
            print("📊 [DEBUG] Calculating legacy scores...")
            legacy_scores = score_features(features, data.platform, (data.mode == "FAST"))
            print("✅ [DEBUG] Legacy scores calculated")
            
            # YENİ: Hook analizi (hızlı versiyon - 3 frame)
            print("🎯 [DEBUG] Analyzing hook (first 3 seconds)...")
            try:
                hook_result = analyze_hook(video_path, audio_path, frames[:3], features, duration)
                print(f"✅ [DEBUG] Hook analysis completed: {hook_result['score']}/18")
            except Exception as e:
                print(f"⚠️ [DEBUG] Hook analysis failed: {e}, using fallback")
                hook_result = {"score": 8, "findings": [], "recommendations": [f"Hook analizi hatası: {e}"]}
            
            # YENİ: Pacing analizi (hızlı versiyon - 3 frame)
            print("⚡ [DEBUG] Analyzing pacing & retention...")
            try:
                pacing_result = analyze_pacing_retention(frames, features, duration)
                print(f"✅ [DEBUG] Pacing analysis completed: {pacing_result['score']}/12")
            except Exception as e:
                print(f"⚠️ [DEBUG] Pacing analysis failed: {e}, using fallback")
                pacing_result = {"score": 6, "findings": [], "recommendations": [f"Pacing analizi hatası: {e}"]}
            
            # Gelişmiş skorları birleştir
            advanced_scores = {
                "hook_score": hook_result["score"],
                "pacing_score": pacing_result["score"],
                **legacy_scores  # Eski skorlar da dahil
            }
            
            # Toplam skor (hook ve pacing dahil)
            total_score = (hook_result["score"] + pacing_result["score"] + 
                          legacy_scores.get("overall_score", 0))
            advanced_scores["total_score"] = total_score
            advanced_scores["overall_score"] = total_score  # Frontend için
            
            # Legacy öneriler
            print("💡 [DEBUG] Generating legacy suggestions...")
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
            all_suggestions.extend(legacy_suggestions.get("tips", []))
            
            # Findings birleştir
            all_findings = []
            all_findings.extend(hook_result.get("findings", []))
            all_findings.extend(pacing_result.get("findings", []))
            
            print("✅ [DEBUG] Advanced suggestions generated")
            print(f"🎯 [DEBUG] Total advanced score: {total_score}")
            
            print("✅ [DEBUG] Analysis completed successfully!")
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
                "score": total_score,  # Frontend için direkt score
                "overall_score": total_score  # Frontend için
            }
    except Exception as e:
        print(f"❌ [ERROR] Analysis failed: {str(e)}")
        print(f"❌ [ERROR] Error type: {type(e)}")
        import traceback
        print(f"❌ [ERROR] Traceback: {traceback.format_exc()}")
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
