import subprocess
import shutil
import os
import math
import json
from pathlib import Path
from typing import List, Optional, Callable, Any
import tempfile
import re

import requests
import time
from typing import Callable

# yt-dlp için opsiyonel import
try:
    import yt_dlp
except ImportError:
    yt_dlp = None


def _ensure_ffmpeg_exists() -> None:
    for bin_name in ["ffmpeg", "ffprobe"]:
        if shutil.which(bin_name) is None:
            raise RuntimeError(f"{bin_name} bulunamadı. Lütfen FFmpeg kurun ve PATH'e ekleyin.")


def _is_social_media_url(url: str) -> bool:
    """Instagram, TikTok, YouTube gibi sosyal medya linklerini tespit eder"""
    social_patterns = [
        r'instagram\.com',
        r'tiktok\.com', 
        r'youtube\.com',
        r'youtu\.be',
        r'facebook\.com',
        r'twitter\.com',
        r'x\.com'
    ]
    return any(re.search(pattern, url.lower()) for pattern in social_patterns)


def _normalize_instagram_url(url: str) -> str:
    # instagram /reels/ → /reel/ formatlarını normalize et
    try:
        u = url.strip()
        u = u.replace("/reels/", "/reel/")
        # trailing slash standardize
        if not u.endswith("/"):
            u += "/"
        return u
    except Exception:
        return url


def download_video(url: str, dest_path: str, timeout: int = 60) -> None:
    """Video indirme - sosyal medya için yt-dlp, optimize edildi"""
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    
    if _is_social_media_url(url) and yt_dlp is not None:
        # Instagram URL normalize
        if "instagram.com" in url:
            url = _normalize_instagram_url(url)
        # Sosyal medya için yt-dlp kullan (hız için optimize edildi)
        ydl_opts = {
            'format': 'best[height<=480]/best',
            'outtmpl': dest_path,
            'quiet': False,  # Debug için açık
            'no_warnings': False,  # Debug için açık
            'extract_flat': False,
            'socket_timeout': timeout,
            'retries': 3,
            # Instagram için gerekli ayarlar
            'cookiesfrombrowser': None,
            'extractor_args': {
                'instagram': {
                    'api': 'web',
                    'web_api': True,
                    'include_stories': False,
                    'include_posts': True
                }
            },
            # User agent ve headers
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        }
        try:
            print(f"Downloading with yt-dlp: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Download completed: {dest_path}")
            return
        except Exception as e:
            print(f"yt-dlp failed: {e}")
            print(f"Error details: {str(e)}")
            # Instagram alternatif yöntemler
            if "instagram.com" in url:
                # 1. instagram.com.tr deneyelim
                alt1 = url.replace("instagram.com", "instagram.com.tr")
                try:
                    print(f"Retrying with instagram.com.tr: {alt1}")
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([alt1])
                    print(f"Download completed (instagram.com.tr): {dest_path}")
                    return
                except Exception as e2:
                    print(f"instagram.com.tr failed: {e2}")
                
                # 2. embed format deneyelim
                alt2 = url.replace("/reel/", "/embed/reel/")
                try:
                    print(f"Retrying with embed format: {alt2}")
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([alt2])
                    print(f"Download completed (embed): {dest_path}")
                    return
                except Exception as e3:
                    print(f"embed format failed: {e3}")
            
            raise Exception(f"Video indirme başarısız: {e}")
    
    # Fallback: normal HTTP indirme (sosyal medya değilse)
    try:
        print(f"Downloading with requests: {url}")
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"Download completed: {dest_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        raise Exception(f"Video indirme başarısız: {e}")


def extract_audio_via_ffmpeg(
    input_video: str,
    output_wav: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> None:
    _ensure_ffmpeg_exists()
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        args = [
            "ffmpeg",
            "-y",
            "-i",
            input_video,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1" if mono else "2",
            output_wav,
        ]
        result = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio extracted: {output_wav}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg audio extraction failed: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        
        raise Exception(f"Ses çıkarma başarısız: {e}")


def grab_frames(input_video: str, output_dir: str, max_frames: int = 10) -> List[str]:
    _ensure_ffmpeg_exists()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Video süresini al ve optimal frame interval hesapla
    try:
        duration = get_video_duration(input_video)
        if duration <= 0:
            return []
        
        # Frame interval hesapla (maksimum frame sayısına göre)
        interval = max(1, duration / max_frames)
        
        pattern = str(Path(output_dir) / "frame_%03d.jpg")
        args = [
            "ffmpeg",
            "-y",
            "-i", input_video,
            "-vf", f"fps=1/{interval}",  # Dinamik interval
            "-q:v", "5",  # Daha hızlı JPEG (kalite vs hız)
            "-threads", "2",  # Optimize thread sayısı
            "-frames:v", str(max_frames),  # Maksimum frame limiti
            pattern,
        ]
        
        subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        
    except subprocess.TimeoutExpired:
        print("️ Frame extraction timeout - using available frames")
    except subprocess.CalledProcessError as e:
        print(f"️ Frame extraction error: {e}")
        return []
    except Exception as e:
        print(f"️ Frame extraction failed: {e}")
        return []
    
    # Oluşturulan frame dosyalarını listele
    frames = sorted([str(p) for p in Path(output_dir).glob("frame_*.jpg")])
    print(f"Extracted {len(frames)} frames from {duration:.1f}s video")
    return frames


def get_video_duration(input_video: str) -> float:
    _ensure_ffmpeg_exists()
    args = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_video,
    ]
    proc = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = proc.stdout.decode().strip()
    try:
        return float(out)
    except Exception:
        return 0.0


# ---- Retry/backoff helpers ----

def with_retry(fn: Callable[[], Any], retries: int = 3, backoff_seconds: float = 0.8) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt == retries - 1:
                break
            time.sleep(backoff_seconds * (2 ** attempt))
    if last_exc:
        raise last_exc
    return None


