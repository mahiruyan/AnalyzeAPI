import os
from typing import Dict, Any, List


def _rule_based_suggestions(
    platform: str,
    caption: str,
    title: str,
    tags: List[str],
    features: Dict[str, Any],
    scores: Dict[str, float],
) -> Dict[str, Any]:
    suggestions: List[str] = []
    
    # Gerçek video analizine dayalı spesifik öneriler
    audio = features.get("audio", {})
    visual = features.get("visual", {})
    textual = features.get("textual", {})
    duration = features.get("duration_seconds", 0)
    
    # Ses analizi önerileri
    loudness = audio.get("loudness_lufs", 0)
    tempo = audio.get("tempo_bpm", 0)
    
    if loudness < -20:
        suggestions.append(f"🎵 Ses seviyeniz çok düşük ({loudness:.1f} LUFS). -14 ile -10 LUFS arası ideal.")
    elif loudness > -8:
        suggestions.append(f"🎵 Ses seviyeniz çok yüksek ({loudness:.1f} LUFS). Dinleyiciler rahatsız olabilir.")
    
    if tempo < 70:
        suggestions.append(f"🎶 Müzik temponuz yavaş ({tempo:.0f} BPM). Instagram için 90-120 BPM daha etkili.")
    elif tempo > 160:
        suggestions.append(f"🎶 Müzik temponuz çok hızlı ({tempo:.0f} BPM). Daha sakin bir müzik deneyin.")
    
    # Görsel analiz önerileri
    flow = visual.get("optical_flow_mean", 0)
    if flow < 0.5:
        suggestions.append("📹 Videoda çok az hareket var. Daha dinamik sahneler ekleyin.")
    elif flow > 3.0:
        suggestions.append("📹 Çok hızlı geçişler var. Bazı sahneleri daha uzun tutun.")
    
    # Metin analizi önerileri
    ocr_text = textual.get("ocr_text", "")
    asr_text = textual.get("asr_text", "")
    
    if len(ocr_text) < 10:
        suggestions.append("📝 Videoda yazı/metin yok. Anahtar kelimeleri ekranda gösterin.")
    
    if len(asr_text) < 20:
        suggestions.append("🎤 Konuşma çok az. Sesli açıklama ekleyin.")
    
    # Süre önerileri
    if duration < 15:
        suggestions.append(f"⏱️ Video çok kısa ({duration:.1f}s). Instagram için 30-60 saniye ideal.")
    elif duration > 90:
        suggestions.append(f"⏱️ Video çok uzun ({duration:.1f}s). 60 saniyeden kısa tutun.")
    
    # Spesifik timing önerileri
    if scores.get("hook_score", 0) < 0.6:
        suggestions.append("🎯 İlk 3 saniyede dikkat çekici bir görsel veya ses efekti ekleyin.")
    
    if scores.get("flow_score", 0) < 0.6:
        suggestions.append("🔄 3-5 saniye aralıklarla sahne değiştirin.")
    
    # Platform spesifik öneriler
    if platform.lower() == "instagram":
        if not ocr_text:
            suggestions.append("📱 Instagram için alt yazı kullanın - sessiz izleme yaygın.")
        suggestions.append("🌈 Canlı renkler kullanın - Instagram'da daha çok dikkat çeker.")
    
    return {
        "tips": suggestions,
        "technical_details": {
            "audio_loudness": f"{loudness:.1f} LUFS",
            "tempo": f"{tempo:.0f} BPM", 
            "visual_flow": f"{flow:.2f}",
            "duration": f"{duration:.1f}s",
            "text_detected": len(ocr_text) > 0,
            "speech_detected": len(asr_text) > 20
        },
        "improvement_areas": [k for k, v in scores.items() if v < 0.6]
    }


def _openai_suggestions(
    platform: str,
    caption: str,
    title: str,
    tags: List[str],
    features: Dict[str, Any],
    scores: Dict[str, float],
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception:
        return _rule_based_suggestions(platform, caption, title, tags, features, scores)

    client = OpenAI()
    sys_prompt = (
        "Kısa video optimizasyon asistanısın. Hook, akış, ses kalitesi ve içerik uyumu için kısa ve uygulanabilir öneriler üret."
    )
    user_prompt = f"""
Platform: {platform}
Başlık: {title}
Altyazı: {caption}
Etiketler: {', '.join(tags)}
Özellikler: {features}
Skorlar: {scores}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        text = resp.choices[0].message.content or ""
    except Exception:
        return _rule_based_suggestions(platform, caption, title, tags, features, scores)

    # Basit ayıklama
    tips = [s.strip("- ").strip() for s in text.split("\n") if s.strip()][:6]
    return {
        "tips": tips or ["Hook'u güçlendir, mesajı sadeleştir, ritmi koru."],
        "alternative_hooks": tips[:3] or ["Saniyeler içinde şunu keşfet!"],
        "hashtags": [*(t.lower() for t in tags), "viral", "trend"][:8],
        "captions": [caption or "Mesajını tek cümlede ver."],
    }


def generate_suggestions(
    platform: str,
    caption: str,
    title: str,
    tags: List[str],
    features: Dict[str, Any],
    scores: Dict[str, float],
    fast_mode: bool = False,
) -> Dict[str, Any]:
    if os.getenv("OPENAI_API_KEY") and not fast_mode:
        return _openai_suggestions(platform, caption, title, tags, features, scores)
    return _rule_based_suggestions(platform, caption, title, tags, features, scores)


