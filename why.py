# why.py
from typing import Dict

def explain_why(summary: Dict, scores: Dict, threshold: float) -> str:
    s = summary
    parts = []

    # Hook / first 3s
    if not s.get("first3s_text"):
        parts.append("İlk 3 saniyede net bir hook cümlesi yok.")
    if not s.get("has_on_screen_text"):
        parts.append("Ekranda ilk saniyelerde okunur bir yazı kullanılmamış.")

    # Audio
    lufs = s.get("lufs")
    if lufs is not None:
        if abs(lufs + 14) > 4:
            parts.append(f"Ses seviyesi {lufs:.1f} LUFS, hedef -14±2 LUFS'tan uzak.")
    clip = s.get("clip_rate")
    if clip is not None and clip > 0.01:
        parts.append("Seste clipping tespit edildi, miksaj sıkıntılı.")

    # Motion / cuts
    flow = s.get("optical_flow", 0.0)
    scenes = s.get("scene_count", 0)
    dur = max(1.0, s.get("duration_s", 1.0))
    cuts_per_10s = scenes / (dur / 10.0)
    if flow < 0.2:
        parts.append("Görsel hareket düşük.")
    if cuts_per_10s < 0.4:
        parts.append("Kesme (cut) sayısı düşük, akış yavaş.")

    # Text length proxies
    if s.get("title_len", 0) < 8:
        parts.append("Başlık/kapak metni kısa veya zayıf.")
    if s.get("caption_len", 0) < 20:
        parts.append("Caption kısa; bağlamı desteklemiyor.")

    # If it looks good, flip tone positive
    if not parts:
        parts = ["İlk saniyelerde net mesaj, yeterli hareket ve kabul edilebilir ses seviyesi ile dikkat korunuyor."]

    # Final tone based on overall score
    vir = float(scores.get("overall_score", 0.0))
    if vir >= threshold:
        prefix = f"Video viral görünüyor (skor: {vir:.2f}). "
    else:
        prefix = f"Video viral değil (skor: {vir:.2f}). "
    return prefix + " ".join(parts)
