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
    
    # BPM analizi - doğal konuşma tarzında
    if tempo > 0:
        if tempo < 60:
            suggestions.append(f"Müziğiniz {tempo:.0f} BPM'de çok sakin. Biraz daha hareketli bir şarkı seçseniz izleyiciler daha çok dikkat eder.")
        elif tempo < 80:
            suggestions.append(f"{tempo:.0f} BPM güzel bir tempo ama viral olmak için 100-120 BPM arası daha etkili. Biraz daha hızlı müzik deneyin.")
        elif tempo < 100:
            suggestions.append(f"{tempo:.0f} BPM mükemmel! Bu tempoyu kesinlikle koruyun, çok dengeli.")
        elif tempo < 140:
            suggestions.append(f"{tempo:.0f} BPM çok enerjik! Bu hız viral içerik için harika, çok dikkat çekici.")
        else:
            suggestions.append(f"{tempo:.0f} BPM biraz fazla hızlı olmuş. Bazı izleyiciler rahatsız olabilir, biraz yavaşlatın.")
    
    # Ses seviyesi analizi - doğal konuşma tarzında
    if loudness != 0:
        if loudness < -25:
            suggestions.append(f" Sesiniz çok kısık ({loudness:.1f} LUFS). Biraz daha yükseltirseniz dinleyiciler daha rahat duyar.")
        elif loudness < -18:
            suggestions.append(f" Ses seviyeniz biraz düşük ({loudness:.1f} LUFS). Biraz daha açsanız çok daha iyi olur.")
        elif loudness < -12:
            suggestions.append(f" Ses seviyeniz harika ({loudness:.1f} LUFS)! Tam ideal seviyede, böyle devam edin.")
        elif loudness < -8:
            suggestions.append(f" Ses biraz yüksek ({loudness:.1f} LUFS). Hafifçe kısarsanız daha rahat dinlenir.")
        else:
            suggestions.append(f" Ses çok yüksek ({loudness:.1f} LUFS). Biraz kısın, dinleyiciler rahatsız olabilir.")
    
    # Görsel hareket analizi - doğal konuşma tarzında
    if flow > 0:
        if flow < 0.3:
            suggestions.append(f" Videonuz çok statik görünüyor. Biraz daha kamera hareketi veya zoom ekleseniz daha canlı olur.")
        elif flow < 0.8:
            suggestions.append(f" Hareket seviyeniz orta ({flow:.2f}). Biraz daha dinamik çekimler deneyin, daha dikkat çekici olur.")
        elif flow < 2.0:
            suggestions.append(f" Hareket dengeniz harika ({flow:.2f})! Bu seviyeyi kesinlikle koruyun, çok profesyonel.")
        elif flow < 3.5:
            suggestions.append(f" Çok yoğun hareket var ({flow:.2f}). Bazı sahneleri biraz yavaşlatırsanız daha rahat izlenir.")
        else:
            suggestions.append(f" Geçişler çok hızlı ({flow:.2f}). Biraz daha yavaş edit yaparsanız izleyiciler takip edebilir.")
    
    # Metin tespiti - doğal konuşma tarzında
    if len(ocr_text) > 0:
        suggestions.append(f"Videonuzda '{ocr_text[:50]}...' yazısı var. Bu güzel bir başlangıç, yazılar çok etkili!")
        if len(ocr_text) < 30:
            suggestions.append("Biraz daha yazı ekleseniz harika olur. Metinler viral içerik için süper etkili.")
    else:
        suggestions.append("Videonuzda hiç yazı göremiyorum. Anahtar kelimeleri ekranda gösterirseniz çok daha etkili olur.")
    
    # Konuşma analizi - doğal konuşma tarzında
    if len(asr_text) > 0:
        suggestions.append(f" Konuşmanızı duyuyorum: '{asr_text[:40]}...' - Bu çok güzel!")
        if len(asr_text) < 50:
            suggestions.append(" Biraz daha uzun konuşsanız harika olur. Detaylı açıklamalar viral için çok etkili.")
    else:
        suggestions.append(" Videonuzda konuşma duymuyorum. Sesli açıklama eklerseniz çok daha etkileşimli olur.")
    
    # Süre analizi - doğal konuşma tarzında
    if duration > 0:
        suggestions.append(f"Videonuz {duration:.1f} saniye sürüyor.")
        if platform.lower() == "instagram":
            if duration < 20:
                suggestions.append("Instagram Reels için biraz kısa. En az 30 saniye yaparsanız daha etkili olur.")
            elif duration > 90:
                suggestions.append("Instagram için biraz uzun olmuş. 60 saniyeden kısa tutarsanız daha iyi.")
            else:
                suggestions.append("Süre Instagram için mükemmel! Bu uzunlukta devam edin.")
    
    # Skorlara göre spesifik öneriler - doğal konuşma tarzında
    hook_score = scores.get("hook_score", 0)
    flow_score = scores.get("flow_score", 0)
    audio_score = scores.get("audio_quality_score", 0)
    
    if hook_score < 0.5:
        suggestions.append("İlk 3 saniyeniz biraz sönük. Daha güçlü bir açılış yaparsanız çok daha dikkat çeker.")
    elif hook_score < 0.7:
        suggestions.append("Açılışınız orta seviyede. Biraz daha dikkat çekici bir başlangıç denerseniz harika olur.")
    
    if flow_score < 0.5:
        suggestions.append("Video akışınız biraz yavaş. Daha hızlı edit yaparsanız daha canlı olur.")
    elif flow_score < 0.7:
        suggestions.append("Akışınız güzel ama biraz daha ritmik olabilir. Biraz daha hızlı geçişler deneyin.")
    
    if audio_score < 0.5:
        suggestions.append(" Ses kaliteniz biraz düşük. Daha temiz bir kayıt yaparsanız çok daha profesyonel olur.")
    elif audio_score < 0.7:
        suggestions.append(" Ses kaliteniz orta seviyede. Biraz daha iyileştirirseniz mükemmel olur.")
    
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


# Mock OpenAI fonksiyonu kaldırıldı - sadece gerçek analiz kullanılıyor


def generate_suggestions(
    platform: str,
    caption: str,
    title: str,
    tags: List[str],
    features: Dict[str, Any],
    scores: Dict[str, float],
    fast_mode: bool = False,
) -> Dict[str, Any]:
    # Sadece gerçek analiz kullan - mock OpenAI fonksiyonunu kullanma
    return _rule_based_suggestions(platform, caption, title, tags, features, scores)


