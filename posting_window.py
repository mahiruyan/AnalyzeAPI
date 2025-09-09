from datetime import datetime, timedelta
from typing import List, Tuple


def best_posting_windows(platform: str, tz_offset_hours: int = 3, country: str = "TR") -> List[Tuple[str, str]]:
    """Basit heuristik: platforma göre 2-3 zaman penceresi döndürür (yerel saat).
    tz_offset_hours: UTC'den ofset (Europe/Istanbul ~ +3)
    """
    base = [
        ("08:00", "10:00"),
        ("12:00", "14:00"),
        ("18:00", "21:00"),
    ]
    p = platform.lower()
    if "tiktok" in p:
        base = [("09:00", "11:00"), ("17:00", "20:00"), ("21:00", "23:00")]
    elif "instagram" in p or "reels" in p:
        base = [("08:00", "10:00"), ("12:00", "14:00"), ("19:00", "22:00")]
    elif "youtube" in p or "shorts" in p:
        base = [("10:00", "12:00"), ("16:00", "19:00")]
    return base


