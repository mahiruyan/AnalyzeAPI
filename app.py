
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

class AnalyzeResponse(BaseModel):
    score: float = Field(..., ge=0, le=100)
    verdict: str = Field(..., pattern="^(low|mid|high|error)$")
    suggestions: List[str]

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
            try:
                extract_audio_via_ffmpeg(video_path, audio_path, sample_rate=16000, mono=True)
                safe_print(f"[DEBUG] Audio extracted to: {audio_path}")
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
        
        # Toplam skor (tm analizler dahil)
        total_score = (hook_result["score"] + pacing_result["score"] + 
                      cta_result["score"] + message_result["score"] + 
                      tech_result["score"] + transition_result["score"] + 
                      loop_result["score"] + text_result["score"] + 
                      trend_result["score"] + content_result["score"] + 
                      music_result["score"] + access_result["score"] + 
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
        
        safe_print("[DEBUG] Advanced suggestions generated")
        safe_print(f"[DEBUG] Total advanced score: {total_score}")
        
        safe_print("[DEBUG] Analysis completed successfully!")
        return {
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
        }
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
