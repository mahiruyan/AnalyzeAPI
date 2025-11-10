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


def safe_print(*args: object) -> None:
    """Print helper that never explodes during Docker builds."""
    try:
        print(*args)
    except UnicodeEncodeError:
        fallback: Iterable[str] = (
            str(arg).encode("ascii", "replace").decode("ascii") for arg in args
        )
        print(*fallback)


def preload_easyocr() -> None:
    try:
        safe_print("[PRELOAD] EasyOCR modelleri indiriliyor...")
        import easyocr  # type: ignore

        # gpu=False => CPU ortamlarına uygun; verbose=FALSE => build logunu sakin tut.
        easyocr.Reader(["en", "tr"], gpu=False, verbose=False)
        safe_print("[PRELOAD] EasyOCR modelleri hazır.")
    except ImportError:
        safe_print("[PRELOAD] easyocr paketi yüklü değil, atlanıyor.")
    except Exception as exc:
        safe_print(f"[PRELOAD] EasyOCR preload başarısız: {exc}")
        raise


def preload_whisper() -> None:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        safe_print("[PRELOAD] faster-whisper kurulmamış, ASR preload atlanacak.")
        return

    try:
        safe_print("[PRELOAD] Whisper 'small' modeli indiriliyor...")
        # compute_type=int8 => CPU uyumlu, gereksiz GPU arayışını engeller.
        WhisperModel("small", compute_type="int8")
        safe_print("[PRELOAD] Whisper modeli hazır.")
    except Exception as exc:
        safe_print(f"[PRELOAD] Whisper preload başarısız: {exc}")
        raise


def main() -> int:
    # Bazı PyTorch tabanlı paketler fork uyarısı basmasın diye.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    preload_easyocr()
    preload_whisper()

    safe_print("[PRELOAD] Tüm preload işlemleri tamamlandı.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

