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
    media_id: str
    published_at: str
    metrics: Dict[str, Any]

class AnalyzeResponse(BaseModel):
    score: float = Field(..., ge=0, le=100)
    verdict: str = Field(..., pattern="^(low|mid|high)$")
    suggestions: List[str]

# Hata yakalama middleware
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"error": "internal_error"}

@app.get("/healthz")
def health_check():
    """Railway sağlık kontrolü endpoint'i"""
    return {"status": "ok"}

@app.get("/api/health")
def api_health_check():
    """Railway API sağlık kontrolü endpoint'i"""
    return {"status": "ok"}

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
        # Her zaman gerçek video analizi yap
        import tempfile
        import os
        from pathlib import Path
        from utils import download_video, extract_audio_via_ffmpeg, grab_frames, get_video_duration
        from features import extract_features
        from scoring import score_features
        from suggest import generate_suggestions
    except Exception as e:
        return {
            "error": f"Import hatası: {str(e)}",
            "duration_seconds": 0,
            "features": {},
            "scores": {},
            "verdict": "error",
            "viral": False,
            "mode": data.mode or "UNKNOWN",
            "analysis_complete": False,
            "suggestions": {"tips": [f"Hata: {str(e)}"]}
        }
    
    # Video indirme ve analiz
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "video.mp4")
            audio_path = os.path.join(temp_dir, "audio.wav")
            frames_dir = os.path.join(temp_dir, "frames")
            
            # Video indir
            download_video(data.file_url, video_path)
            
            # Ses çıkar
            extract_audio_via_ffmpeg(video_path, audio_path)
            
            # Frame'ler çıkar
            frames = grab_frames(video_path, frames_dir, max_frames=10)
            
            # Süre al
            duration = get_video_duration(video_path)
            
            # Özellikler çıkar
            features = extract_features(
                audio_path=audio_path,
                frames=frames,
                caption=data.caption or "",
                title=data.title or "",
                tags=data.tags or [],
                duration_seconds=duration,
                fast_mode=(data.mode == "FAST")
            )
            
            # Skorlar hesapla
            scores = score_features(features, data.platform, (data.mode == "FAST"))
            
            # Spesifik öneriler oluştur
            suggestions = generate_suggestions(
                platform=data.platform,
                caption=data.caption or "",
                title=data.title or "",
                tags=data.tags or [],
                features=features,
                scores=scores
            )
            
            # Toplam skor
            total_score = scores.get("overall_score", 0)
            
            return {
                "duration_seconds": duration,
                "features": features,
                "scores": scores,
                "verdict": "high" if total_score >= 70 else "mid" if total_score >= 40 else "low",
                "viral": total_score >= 70,
                "mode": data.mode,
                "analysis_complete": True,
                "suggestions": suggestions
            }
    except Exception as e:
        return {
            "error": f"Analiz hatası: {str(e)}",
            "duration_seconds": 0,
            "features": {},
            "scores": {},
            "verdict": "error", 
            "viral": False,
            "mode": data.mode or "UNKNOWN",
            "analysis_complete": False,
            "suggestions": {"tips": [f"Video analiz hatası: {str(e)}"]}
        }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_performance(data: AnalyzeRequest):
    """Legacy endpoint - redirects to /api/analyze"""
    # Gerçek analiz için /api/analyze'i çağır
    result = analyze_performance_api(data)
    
    return AnalyzeResponse(
        score=result.get("scores", {}).get("overall_score", 0),
        verdict=result.get("verdict", "mid"),
        suggestions=result.get("suggestions", {}).get("tips", [])
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
