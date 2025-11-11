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
        print(*args)
    except UnicodeEncodeError:
        fallback: Iterable[str] = (
            str(arg).encode("ascii", "replace").decode("ascii") for arg in args
        )
        print(*fallback)
def main() -> int:
    # Bazı PyTorch tabanlı paketler fork uyarısı basmasın diye.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from features import warmup_models
        safe_print("[PRELOAD] Model warmup başlatılıyor...")
        warmup_models()
        safe_print("[PRELOAD] Model warmup tamamlandı.")
    except Exception as exc:
        safe_print(f"[PRELOAD] Warmup script failed: {exc}")
        raise

    safe_print("[PRELOAD] Tüm preload işlemleri tamamlandı.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

