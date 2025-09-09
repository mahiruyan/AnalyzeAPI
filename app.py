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
    platform: str = Field(..., regex="^(instagram|tiktok)$")
    media_id: str
    published_at: str
    metrics: Dict[str, Any]

class AnalyzeResponse(BaseModel):
    score: float = Field(..., ge=0, le=100)
    verdict: str = Field(..., regex="^(low|mid|high)$")
    suggestions: List[str]

# Hata yakalama middleware
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"error": "internal_error"}

@app.get("/healthz")
def health_check():
    """Railway sağlık kontrolü endpoint'i"""
    return {"status": "ok"}

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

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_performance(data: AnalyzeRequest):
    """Normalize edilmiş payload ile skor hesapla"""
    metrics = data.metrics
    
    # Base skor (plays/views)
    base_metric = metrics.get("plays", metrics.get("views", 0))
    base_score = min(base_metric / 10000, 1.0) * 30  # Max 30 puan
    
    # Engagement skor
    likes = metrics.get("likes", 0)
    comments = metrics.get("comments", 0)
    shares = metrics.get("shares", 0)
    saves = metrics.get("saves", 0)
    
    engagement_total = likes + comments + shares + saves
    engagement_score = (engagement_total / max(base_metric, 1)) * 40  # Max 40 puan
    
    # Watch time bonus
    avg_watch_time = metrics.get("avg_watch_time_s", 0)
    completion_rate = metrics.get("completion_rate", 0)
    
    watch_bonus = 0
    if avg_watch_time:
        watch_bonus += min(avg_watch_time / 30, 1.0) * 15  # Max 15 puan
    if completion_rate:
        watch_bonus += completion_rate * 15  # Max 15 puan
    
    # Toplam skor
    total_score = min(base_score + engagement_score + watch_bonus, 100)
    
    # Verdict belirleme
    if total_score >= 70:
        verdict = "high"
        suggestions = ["Mükemmel performans! Bu içeriği daha fazla üretin.", "Benzer formatları tekrarlayın."]
    elif total_score >= 40:
        verdict = "mid"
        suggestions = ["Engagement'i artırmak için hook'u güçlendirin.", "İlk 3 saniyede dikkat çekici bir şey ekleyin."]
    else:
        verdict = "low"
        suggestions = ["İçerik kalitesini artırın.", "Daha kısa ve odaklı videolar yapın.", "Trending hashtag'leri kullanın."]
    
    return AnalyzeResponse(
        score=round(total_score, 1),
        verdict=verdict,
        suggestions=suggestions
    )

# Mevcut main.py'deki tüm endpoint'leri import et (video analizi için)
try:
    from main import app as main_app
    app.include_router(main_app.router)
except ImportError:
    pass  # main.py yoksa devam et

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
