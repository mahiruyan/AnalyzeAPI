"""
Build-time model preloader for AnalyzeAPI.

This script downloads the heavy ML assets (EasyOCR, Whisper) during the
Docker image build so that runtime requests do not have to wait for the
initial warm-up phase.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable
from pathlib import Path


def safe_print(*args: object) -> None:
    """Print helper that never explodes during Docker builds."""
    try:
        print(*args, flush=True)
    except UnicodeEncodeError:
        fallback: Iterable[str] = (
            str(arg).encode("ascii", "replace").decode("ascii") for arg in args
        )
        print(*fallback, flush=True)


def preload_easyocr() -> None:
    """EasyOCR modellerini Docker build sırasında indir."""
    safe_print("[PRELOAD] EasyOCR başlatılıyor...")
    try:
        import easyocr
        safe_print("[PRELOAD] EasyOCR Reader oluşturuluyor (en, tr, gpu=False)...")
        reader = easyocr.Reader(['en', 'tr'], gpu=False, verbose=True)
        safe_print("[PRELOAD] ✓ EasyOCR modelleri başarıyla indirildi ve cache'lendi.")
    except ImportError:
        safe_print("[PRELOAD] ⚠ easyocr paketi yüklü değil, atlanıyor.")
    except Exception as exc:
        safe_print(f"[PRELOAD] ✗ EasyOCR preload başarısız: {exc}")
        import traceback
        safe_print(traceback.format_exc())
        raise


def preload_whisper() -> None:
    """Whisper modelini Docker build sırasında indir."""
    safe_print("[PRELOAD] Whisper başlatılıyor...")
    try:
        from faster_whisper import WhisperModel
        safe_print("[PRELOAD] WhisperModel 'small' (compute_type=int8) indiriliyor...")
        model = WhisperModel("small", compute_type="int8")
        safe_print("[PRELOAD] ✓ Whisper modeli başarıyla indirildi ve cache'lendi.")
    except ImportError:
        safe_print("[PRELOAD] ⚠ faster-whisper paketi yüklü değil, atlanıyor.")
    except Exception as exc:
        safe_print(f"[PRELOAD] ✗ Whisper preload başarısız: {exc}")
        import traceback
        safe_print(traceback.format_exc())
        raise


def main() -> int:
    safe_print("=" * 60)
    safe_print("[PRELOAD] Docker build-time model preloading başlatılıyor...")
    safe_print("=" * 60)
    
    # PyTorch fork uyarılarını bastır
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    
    # EasyOCR ve Whisper'ı doğrudan indir
    try:
        preload_easyocr()
    except Exception as exc:
        safe_print(f"[PRELOAD] EasyOCR preload durdu, devam ediliyor: {exc}")
    
    try:
        preload_whisper()
    except Exception as exc:
        safe_print(f"[PRELOAD] Whisper preload durdu, devam ediliyor: {exc}")
    
    safe_print("=" * 60)
    safe_print("[PRELOAD] ✓ Tüm model preload işlemleri tamamlandı.")
    safe_print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

