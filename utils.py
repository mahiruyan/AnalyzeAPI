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


def download_video(url: str, dest_path: str, timeout: int = 120) -> None:
    """Video indirme - sosyal medya için yt-dlp, normal URL'ler için requests"""
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    
    if _is_social_media_url(url) and yt_dlp is not None:
        # Sosyal medya için yt-dlp kullan (daha hızlı ve güvenilir)
        ydl_opts = {
            'format': 'best[height<=720]/best',  # 720p altı, hızlı indirme
            'outtmpl': dest_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return
        except Exception as e:
            print(f"yt-dlp hatası: {e}, requests ile deneyeceğim...")
    
    # Fallback: normal HTTP indirme
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


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


def grab_frames(input_video: str, output_dir: str, fps: int = 1) -> List[str]:
    _ensure_ffmpeg_exists()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # ffmpeg ile belirli FPS'te kare çıkarma (hızlı mod)
    pattern = str(Path(output_dir) / "frame_%05d.jpg")
    args = [
        "ffmpeg",
        "-y",
        "-i",
        input_video,
        "-vf",
        f"fps={fps}",
        "-q:v", "2",  # Hızlı JPEG kalitesi
        "-threads", "4",  # Çoklu thread
        pattern,
    ]
    subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = sorted([str(p) for p in Path(output_dir).glob("frame_*.jpg")])
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


