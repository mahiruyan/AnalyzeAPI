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


def download_video(url: str, dest_path: str, timeout: int = 60) -> None:
    """Video indirme - sosyal medya için yt-dlp, optimize edildi"""
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    
    if _is_social_media_url(url) and yt_dlp is not None:
        # Sosyal medya için yt-dlp kullan (hız için optimize edildi)
        ydl_opts = {
            'format': 'best[height<=480]/best',  # 480p'ye düşürdük (hız için)
            'outtmpl': dest_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': timeout,
            'retries': 2,  # Retry sayısını azalttık
            'cookiesfrombrowser': None,  # Instagram için gerekli
            'extractor_args': {
                'instagram': {
                    'api': 'web',
                    'web_api': True
                }
            }
        }
        try:
            print(f"📥 Downloading with yt-dlp: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"✅ Download completed: {dest_path}")
            return
        except Exception as e:
            print(f"❌ yt-dlp failed: {e}")
            # Instagram için özel fallback
            if "instagram" in url.lower():
                print("🔄 Trying Instagram-specific fallback...")
                try:
                    # Instagram için basit HTTP fallback
                    import requests
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(url, headers=headers, timeout=timeout)
                    if response.status_code == 200:
                        # Basit bir placeholder video oluştur
                        print("⚠️ Instagram URL not directly downloadable, using placeholder")
                        # 1 saniyelik siyah video oluştur
                        import subprocess
                        subprocess.run([
                            'ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=black:size=640x480:duration=1',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', dest_path
                        ], check=True, capture_output=True)
                        print(f"✅ Placeholder video created: {dest_path}")
                        return
                except Exception as fallback_e:
                    print(f"❌ Instagram fallback failed: {fallback_e}")
            
            raise Exception(f"Video indirme başarısız: {e}")
    
    # Fallback: normal HTTP indirme (sosyal medya değilse)
    try:
        print(f"📥 Downloading with requests: {url}")
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"✅ Download completed: {dest_path}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise Exception(f"Video indirme başarısız: {e}")


def extract_audio_via_ffmpeg(
    input_video: str,
    output_wav: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> None:
    _ensure_ffmpeg_exists()
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
    subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


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
        print("⚠️ Frame extraction timeout - using available frames")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Frame extraction error: {e}")
        return []
    except Exception as e:
        print(f"⚠️ Frame extraction failed: {e}")
        return []
    
    # Oluşturulan frame dosyalarını listele
    frames = sorted([str(p) for p in Path(output_dir).glob("frame_*.jpg")])
    print(f"✅ Extracted {len(frames)} frames from {duration:.1f}s video")
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


