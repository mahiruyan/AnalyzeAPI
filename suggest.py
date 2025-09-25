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
    
    # Gerçek video verilerini al
    audio = features.get("audio", {})
    visual = features.get("visual", {})
    textual = features.get("textual", {})
    duration = features.get("duration_seconds", 0)
    
    loudness = audio.get("loudness_lufs", 0)
    tempo = audio.get("tempo_bpm", 0)
    flow = visual.get("optical_flow_mean", 0)
    ocr_text = textual.get("ocr_text", "")
    asr_text = textual.get("asr_text", "")
    
    # BPM analizi - gerçek değerlere göre
    if tempo > 0:
        if tempo < 60:
            suggestions.append(f"Bu videonun BPM'i {tempo:.0f} - çok yavaş. Daha enerjik müzik kullanın.")
        elif tempo < 80:
            suggestions.append(f"BPM {tempo:.0f} - orta tempoda. Viral için 100-120 BPM daha etkili olur.")
        elif tempo < 100:
            suggestions.append(f"BPM {tempo:.0f} - iyi tempo. Bu hızı koruyun.")
        elif tempo < 140:
            suggestions.append(f"BPM {tempo:.0f} - hızlı tempo. Çok enerjik, dikkat çekici.")
        else:
            suggestions.append(f"BPM {tempo:.0f} - çok hızlı. Bazı izleyiciler için rahatsız edici olabilir.")
    
    # Ses seviyesi analizi - gerçek LUFS değerine göre
    if loudness != 0:
        if loudness < -25:
            suggestions.append(f"Ses seviyesi {loudness:.1f} LUFS - çok sessiz. Ses seviyesini artırın.")
        elif loudness < -18:
            suggestions.append(f"Ses seviyesi {loudness:.1f} LUFS - düşük. Daha yüksek ses kullanın.")
        elif loudness < -12:
            suggestions.append(f"Ses seviyesi {loudness:.1f} LUFS - ideal seviye.")
        elif loudness < -8:
            suggestions.append(f"Ses seviyesi {loudness:.1f} LUFS - yüksek. Biraz azaltın.")
        else:
            suggestions.append(f"Ses seviyesi {loudness:.1f} LUFS - çok yüksek. Dinleyiciler rahatsız olur.")
    
    # Görsel hareket analizi - gerçek flow değerine göre
    if flow > 0:
        if flow < 0.3:
            suggestions.append(f"Görsel hareket {flow:.2f} - statik video. Daha fazla kamera hareketi ekleyin.")
        elif flow < 0.8:
            suggestions.append(f"Görsel hareket {flow:.2f} - az hareket. Daha dinamik çekim yapın.")
        elif flow < 2.0:
            suggestions.append(f"Görsel hareket {flow:.2f} - dengeli hareket. Bu seviyeyi koruyun.")
        elif flow < 3.5:
            suggestions.append(f"Görsel hareket {flow:.2f} - yoğun hareket. Bazı sahneleri yavaşlatın.")
        else:
            suggestions.append(f"Görsel hareket {flow:.2f} - çok hızlı geçişler. Daha yavaş edit yapın.")
    
    # Metin tespiti - gerçek OCR sonuçlarına göre
    if len(ocr_text) > 0:
        suggestions.append(f"Videoda '{ocr_text[:50]}...' yazısı tespit edildi. Bu iyi bir başlangıç.")
        if len(ocr_text) < 30:
            suggestions.append("Daha fazla yazı ekleyin - metinler viral içerik için önemli.")
    else:
        suggestions.append("Videoda hiç yazı yok. Anahtar kelimeleri ekranda gösterin.")
    
    # Konuşma analizi - gerçek ASR sonuçlarına göre
    if len(asr_text) > 0:
        suggestions.append(f"Konuşma tespit edildi: '{asr_text[:40]}...'")
        if len(asr_text) < 50:
            suggestions.append("Daha uzun açıklama yapın. Detaylı konuşma viral için etkili.")
    else:
        suggestions.append("Hiç konuşma yok. Sesli açıklama ekleyin.")
    
    # Süre analizi - gerçek süreye göre
    if duration > 0:
        suggestions.append(f"Video süresi {duration:.1f} saniye.")
        if platform.lower() == "instagram":
            if duration < 20:
                suggestions.append("Instagram Reels için çok kısa. En az 30 saniye yapın.")
            elif duration > 90:
                suggestions.append("Instagram için çok uzun. 60 saniyeden kısa tutun.")
            else:
                suggestions.append("Instagram için ideal süre.")
    
    # Skorlara göre spesifik öneriler
    hook_score = scores.get("hook_score", 0)
    flow_score = scores.get("flow_score", 0)
    audio_score = scores.get("audio_quality_score", 0)
    
    if hook_score < 0.5:
        suggestions.append("İlk 3 saniye etkisiz. Daha güçlü bir açılış yapın.")
    elif hook_score < 0.7:
        suggestions.append("Açılış orta seviye. Daha dikkat çekici bir başlangıç deneyin.")
    
    if flow_score < 0.5:
        suggestions.append("Video akışı yavaş. Daha hızlı edit yapın.")
    elif flow_score < 0.7:
        suggestions.append("Akış dengeli ama daha ritmik olabilir.")
    
    if audio_score < 0.5:
        suggestions.append("Ses kalitesi düşük. Daha temiz ses kaydı yapın.")
    elif audio_score < 0.7:
        suggestions.append("Ses kalitesi orta. Biraz daha iyileştirin.")
    
    return {
        "tips": suggestions,
        "analysis_summary": {
            "actual_bpm": f"{tempo:.0f}",
            "actual_loudness": f"{loudness:.1f} LUFS",
            "actual_movement": f"{flow:.2f}",
            "actual_duration": f"{duration:.1f}s",
            "text_found": len(ocr_text) > 0,
            "speech_found": len(asr_text) > 0,
            "hook_strength": f"{hook_score:.1%}",
            "flow_quality": f"{flow_score:.1%}",
            "audio_quality": f"{audio_score:.1%}"
        }
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


