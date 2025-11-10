
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

# yt-dlp iin opsiyonel import
try:
    import yt_dlp
except ImportError:
    yt_dlp = None


def _ensure_ffmpeg_exists() -> None:
    for bin_name in ["ffmpeg", "ffprobe"]:
        if shutil.which(bin_name) is None:
            raise RuntimeError(f"{bin_name} bulunamad. Ltfen FFmpeg kurun ve PATH'e ekleyin.")


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
    # instagram /reels/  /reel/ formatlarn normalize et
    try:
        u = url.strip()
        u = u.replace("/reels/", "/reel/")
        # trailing slash standardize
        if not u.endswith("/"):
            u += "/"
        return u
    except Exception:
        return url


def _resolve_scrapingbee_proxy() -> Optional[str]:
    proxy_url = os.getenv("SCRAPINGBEE_PROXY_URL")
    if proxy_url:
        return proxy_url.strip()

    proxy_host = os.getenv("SCRAPINGBEE_PROXY_HOST")
    proxy_port = os.getenv("SCRAPINGBEE_PROXY_PORT")
    if proxy_host and proxy_port:
        proxy_user = os.getenv("SCRAPINGBEE_PROXY_USER")
        proxy_pass = os.getenv("SCRAPINGBEE_PROXY_PASS") or os.getenv("SCRAPINGBEE_API_KEY")
        if proxy_user and proxy_pass:
            return f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
        return f"http://{proxy_host}:{proxy_port}"
    return None


def download_video(url: str, dest_path: str, timeout: int = 60) -> None:
    """Video indirme - sosyal medya iin yt-dlp, optimize edildi"""
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    proxy_url = _resolve_scrapingbee_proxy()
    proxy_host_display = None
    if proxy_url:
        proxy_host_display = proxy_url.split("@")[-1]
        proxy_host_display = proxy_host_display.replace("http://", "").replace("https://", "")
    
    if _is_social_media_url(url) and yt_dlp is not None:
        # Instagram URL normalize
        if "instagram.com" in url:
            url = _normalize_instagram_url(url)

        dest = Path(dest_path)
        base_no_ext = dest.with_suffix("")
        ydl_outtmpl = str(base_no_ext) + ".%(ext)s"

        # Sosyal medya iin yt-dlp kullan (hz iin optimize edildi)
        ydl_opts = {
            'format': 'bv[height<=480]+ba/b[height<=480]/best',
            'outtmpl': ydl_outtmpl,
            'quiet': False,  # Debug iin ak
            'no_warnings': False,  # Debug iin ak
            'extract_flat': False,
            'socket_timeout': timeout,
            'retries': 3,
            'noplaylist': True,
            # Instagram iin gerekli ayarlar
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
        if proxy_url:
            ydl_opts['proxy'] = proxy_url
            safe_print(f"[PROXY] yt-dlp via {proxy_host_display}")
        # mp4 merge garanti
        ydl_opts['merge_output_format'] = 'mp4'
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }]
        ydl_opts['postprocessor_args'] = ['-movflags', 'faststart']

        try:
            safe_print(f"Downloading with yt-dlp: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            # İndirilen gerçek dosyayı bul ve hedef adına taşı
            downloaded_candidates = sorted(dest.parent.glob(f"{base_no_ext.name}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
            selected_file = None
            for candidate in downloaded_candidates:
                if candidate.suffix.lower() in {'.mp4', '.mkv', '.mov', '.webm'}:
                    selected_file = candidate
                    break
            if selected_file is None:
                raise FileNotFoundError("yt-dlp uygun çıktı dosyası üretmedi")

            if selected_file != dest:
                if dest.exists():
                    dest.unlink()
                selected_file.rename(dest)

            file_size = dest.stat().st_size if dest.exists() else 0
            safe_print(f"Download completed: {dest_path} ({file_size/1024/1024:.2f} MB)")
            if file_size == 0:
                raise Exception("İndirilen video dosyası boş görünüyor")
            return
        except Exception as e:
            safe_print(f"yt-dlp failed: {e}")
            safe_print(f"Error details: {str(e)}")
            # Instagram alternatif yntemler
            if "instagram.com" in url:
                # 1. instagram.com.tr deneyelim
                alt1 = url.replace("instagram.com", "instagram.com.tr")
                try:
                    safe_print(f"Retrying with instagram.com.tr: {alt1}")
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([alt1])
                    safe_print(f"Download completed (instagram.com.tr): {dest_path}")
                    return
                except Exception as e2:
                    safe_print(f"instagram.com.tr failed: {e2}")
                
                # 2. embed format deneyelim
                alt2 = url.replace("/reel/", "/embed/reel/")
                try:
                    safe_print(f"Retrying with embed format: {alt2}")
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([alt2])
                    safe_print(f"Download completed (embed): {dest_path}")
                    return
                except Exception as e3:
                    safe_print(f"embed format failed: {e3}")
            
            raise Exception(f"Video indirme baarsz: {e}")
    
    # Fallback: normal HTTP indirme (sadece sosyal medya değilse)
    if not _is_social_media_url(url):
        try:
            safe_print(f"Downloading with requests: {url}")
            proxies = None
            if proxy_url:
                proxies = {"http": proxy_url, "https": proxy_url}
                safe_print(f"[PROXY] requests via {proxy_host_display}")
            resp = requests.get(url, stream=True, timeout=timeout, proxies=proxies)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            safe_print(f"Download completed: {dest_path}")
            return
        except Exception as e:
            safe_print(f"Download failed: {e}")
            raise Exception(f"Video indirme baarsz: {e}")
    else:
        # Sosyal medya URL'si için yt-dlp olmadan indirme yapamayız
        raise Exception(f"Sosyal medya videosu yt-dlp ile indirilemedi: {url}")


def extract_audio_via_ffmpeg(
    input_video: str,
    output_wav: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> None:
    _ensure_ffmpeg_exists()
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    
    # Check if video file exists
    if not Path(input_video).exists():
        raise Exception(f"Video file not found: {input_video}")
    
    # Simple FFmpeg command that works
    args = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1" if mono else "2",
        "-f", "wav",
        "-loglevel", "error",
        output_wav,
    ]
    
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        safe_print(f"Audio extracted: {output_wav}")
        
        # Verify the output file exists and has content
        if not Path(output_wav).exists() or Path(output_wav).stat().st_size == 0:
            raise Exception("Audio extraction produced empty file")
            
    except subprocess.CalledProcessError as e:
        safe_print(f"FFmpeg failed with exit code: {e.returncode}")
        safe_print(f"FFmpeg stderr: {e.stderr}")
        raise Exception(f"Ses karma baarsz: {e}")
    except Exception as e:
        safe_print(f"Audio extraction error: {e}")
        raise Exception(f"Ses karma baarsz: {e}")


def grab_frames(input_video: str, output_dir: str, max_frames: int = 10) -> List[str]:
    _ensure_ffmpeg_exists()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Video sresini al ve optimal frame interval hesapla
    try:
        duration = get_video_duration(input_video)
        if duration <= 0:
            return []
        
        # Frame interval hesapla (maksimum frame saysna gre)
        interval = max(1, duration / max_frames)
        
        pattern = str(Path(output_dir) / "frame_%03d.jpg")
        args = [
            "ffmpeg",
            "-y",
            "-i", input_video,
            "-vf", f"fps=1/{interval}",  # Dinamik interval
            "-q:v", "5",  # Daha hzl JPEG (kalite vs hz)
            "-threads", "2",  # Optimize thread says
            "-frames:v", str(max_frames),  # Maksimum frame limiti
            pattern,
        ]
        
        subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        
    except subprocess.TimeoutExpired:
        safe_print(" Frame extraction timeout - using available frames")
    except subprocess.CalledProcessError as e:
        safe_print(f" Frame extraction error: {e}")
        return []
    except Exception as e:
        safe_print(f" Frame extraction failed: {e}")
        return []
    
    # Oluturulan frame dosyalarn listele
    frames = sorted([str(p) for p in Path(output_dir).glob("frame_*.jpg")])
    safe_print(f"Extracted {len(frames)} frames from {duration:.1f}s video")
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


