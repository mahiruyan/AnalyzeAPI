from typing import Dict, Any
import numpy as np


def _normalize(value: float, min_v: float, max_v: float) -> float:
    if max_v == min_v:
        return 0.0
    v = (value - min_v) / (max_v - min_v)
    return float(np.clip(v, 0.0, 1.0))


def score_features(features: Dict[str, Any], platform: str = "", fast_mode: bool = False) -> Dict[str, float]:
    audio = features.get("audio", {})
    visual = features.get("visual", {})
    textual = features.get("textual", {})
    duration = float(features.get("duration_seconds", 0.0))

    # Heuristik normalizasyon aralklar
    loudness = float(audio.get("loudness_lufs", 0.0))
    tempo = float(audio.get("tempo_bpm", 0.0))
    flow = float(visual.get("optical_flow_mean", 0.0))
    scene_count = float(visual.get("scene_count", 0.0))

    # Hook: ilk saniyelerde yksek hareket + gl metin ipucu
    text_len = len((textual.get("caption", "") + " " + textual.get("title", "") + " " + textual.get("asr_text", "")).strip())
    hook_score = 0.6 * _normalize(flow, 0.0, 3.0) + 0.4 * _normalize(text_len, 0.0, 240)

    # Flow: optik akn dengeli olmas ve sahne geileri
    flow_score = 0.7 * _normalize(flow, 0.2, 2.0) + 0.3 * _normalize(scene_count, 0.0, 12.0)

    # Audio quality: loudness (yaklak -23 ile -10 LUFS aras), tempo 70-160 bpm
    loudness_score = _normalize(-loudness, -(-23.0), -(-10.0))  # -23 .. -10 aralna yaknsa iyi
    tempo_score = _normalize(tempo, 70.0, 160.0)
    audio_quality_score = 0.6 * loudness_score + 0.4 * tempo_score

    # Content fit: etiketler ile metin uyumu
    tags = set([t.lower() for t in textual.get("tags", [])])
    if fast_mode:
        # FAST modda sadece caption ve title kullan
        text_blob = (textual.get("caption", "") + " " + textual.get("title", "")).lower()
    else:
        # FULL modda tm metinleri kullan
        text_blob = (textual.get("caption", "") + " " + textual.get("title", "") + " " + textual.get("ocr_text", "") + " " + textual.get("asr_text", "")).lower()
    tag_hits = sum(1 for t in tags if t in text_blob)
    content_fit_score = _normalize(tag_hits, 0.0, max(3.0, float(len(tags) or 1)))

    # Sre uygunluu (platforma gre kabaca)
    if platform.lower() in {"tiktok", "instagram", "reels", "shorts", "youtube"}:
        # 6 - 60 sn aras ideal varsaym
        duration_score = 1.0 - abs((duration - 30.0) / 30.0)
        duration_score = float(np.clip(duration_score, 0.0, 1.0))
    else:
        duration_score = 1.0

    overall = float(np.clip((hook_score + flow_score + audio_quality_score + content_fit_score + duration_score) / 5.0, 0.0, 1.0))

    return {
        "hook_score": round(hook_score, 3),
        "flow_score": round(flow_score, 3),
        "audio_quality_score": round(audio_quality_score, 3),
        "content_fit_score": round(content_fit_score, 3),
        "duration_fit_score": round(duration_score, 3),
        "overall_score": round(overall, 3),
    }


