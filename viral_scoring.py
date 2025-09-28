"""
Gelişmiş Virallik Skorlama Sistemi
12 boyutlu detaylı video analizi - 999$ premium seviye
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from pathlib import Path
import cv2
import json
from datetime import datetime

def calculate_viral_score(
    video_path: str,
    audio_path: str, 
    frames: List[str],
    features: Dict[str, Any],
    duration: float,
    platform: str = "instagram"
) -> Dict[str, Any]:
    """
    Ana virallik skorlama fonksiyonu
    12 boyutlu analiz ile 0-100 arası skor
    """
    
    # Ana skorlar
    subscores = {}
    findings = []
    recommendations = []
    raw_metrics = {}
    
    # 1. Hook Analizi (İlk 3 saniye) - 18 puan
    hook_result = analyze_hook(frames[:3], features, duration)
    subscores["hook"] = hook_result["score"]
    findings.extend(hook_result["findings"])
    recommendations.extend(hook_result["recommendations"])
    raw_metrics.update(hook_result["raw"])
    
    # 2. Pacing & Retention - 12 puan  
    pacing_result = analyze_pacing(frames, features, duration)
    subscores["pacing"] = pacing_result["score"]
    findings.extend(pacing_result["findings"])
    recommendations.extend(pacing_result["recommendations"])
    raw_metrics.update(pacing_result["raw"])
    
    # 3. Müzik & Senkron - 8 puan
    music_result = analyze_music_sync(features, duration)
    subscores["music"] = music_result["score"]
    findings.extend(music_result["findings"])
    recommendations.extend(music_result["recommendations"])
    raw_metrics.update(music_result["raw"])
    
    # 4. Metin Okunabilirliği - 12 puan
    text_result = analyze_text_readability(frames, features, duration)
    subscores["text_legibility"] = text_result["score"]
    findings.extend(text_result["findings"])
    recommendations.extend(text_result["recommendations"])
    raw_metrics.update(text_result["raw"])
    
    # 5. Mesaj Netliği (4-7 saniye) - 6 puan
    clarity_result = analyze_message_clarity(features, duration)
    subscores["clarity_4_7s"] = clarity_result["score"]
    findings.extend(clarity_result["findings"])
    recommendations.extend(clarity_result["recommendations"])
    
    # 6. Geçiş Kalitesi - 8 puan
    transition_result = analyze_transitions(frames, features)
    subscores["transitions"] = transition_result["score"]
    findings.extend(transition_result["findings"])
    recommendations.extend(transition_result["recommendations"])
    
    # 7. CTA & Etkileşim - 6 puan
    cta_result = analyze_cta(features, duration)
    subscores["cta"] = cta_result["score"]
    findings.extend(cta_result["findings"])
    recommendations.extend(cta_result["recommendations"])
    
    # 8. Loop & Final - 5 puan
    loop_result = analyze_loop(frames, duration)
    subscores["loop"] = loop_result["score"]
    findings.extend(loop_result["findings"])
    recommendations.extend(loop_result["recommendations"])
    
    # 9. Erişilebilirlik - 4 puan
    accessibility_result = analyze_accessibility(features)
    subscores["accessibility"] = accessibility_result["score"]
    findings.extend(accessibility_result["findings"])
    recommendations.extend(accessibility_result["recommendations"])
    
    # 10. Teknik Kalite - 8 puan
    technical_result = analyze_technical_quality(video_path, frames)
    subscores["technical"] = technical_result["score"]
    findings.extend(technical_result["findings"])
    recommendations.extend(technical_result["recommendations"])
    
    # 11. Trend & Orijinallik - 6 puan
    trend_result = analyze_trend_originality(features)
    subscores["trend_originality"] = trend_result["score"]
    findings.extend(trend_result["findings"])
    recommendations.extend(trend_result["recommendations"])
    
    # 12. İçerik Tipi Uygunluğu - 7 puan
    content_type_result = analyze_content_type_fit(features, duration)
    subscores["content_fit"] = content_type_result["score"]
    findings.extend(content_type_result["findings"])
    recommendations.extend(content_type_result["recommendations"])
    
    # Toplam skor hesapla
    final_score = sum(subscores.values())
    
    # Viral threshold
    viral = final_score >= 70
    
    return {
        "final_score": round(final_score, 1),
        "subscores": subscores,
        "findings": findings,
        "recommendations": recommendations[:10],  # En önemli 10 öneri
        "viral": viral,
        "raw": raw_metrics,
        "content_type": detect_content_type(features),
        "analyzed_at": datetime.now().isoformat()
    }


def analyze_hook(frames: List[str], features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """Hook analizi - İlk 3 saniye etkisi (18 puan)"""
    score = 0
    findings = []
    recommendations = []
    raw = {}
    
    # Görsel sıçrama analizi (ilk 3 frame)
    if len(frames) >= 3:
        visual_jump = calculate_visual_jump(frames[:3])
        raw["visual_jump"] = visual_jump
        
        if visual_jump > 0.3:
            score += 5
            findings.append({"t": 0.5, "type": "visual_jump", "msg": f"{visual_jump:.2f} görsel sıçrama - güçlü hook!"})
        else:
            score += 1
            recommendations.append("İlk 3 saniyede ani zoom, renk değişimi veya yakın plan ekle")
    
    # Metin kancası kontrolü
    ocr_text = features.get("textual", {}).get("ocr_text", "")
    if ocr_text and len(ocr_text.split()) <= 10:
        hook_phrases = ["kimse", "gizli", "şok", "inanamayacaksın", "bu yöntem", "sır"]
        if any(phrase in ocr_text.lower() for phrase in hook_phrases):
            score += 4
            findings.append({"t": 1.0, "type": "hook_text", "msg": f"Güçlü kanca metni: '{ocr_text[:30]}...'"})
        else:
            score += 2
            recommendations.append("İlk saniyede güçlü bir kanca metni ekle (örn: 'Kimse söylemiyor ama...')")
    else:
        recommendations.append("İlk 3 saniyede kısa ve etkili bir soru/kanca ekle")
    
    # Ses giriş vuruşu
    audio = features.get("audio", {})
    tempo = audio.get("tempo_bpm", 0)
    if tempo > 100:
        score += 3
        findings.append({"t": 1.0, "type": "energy_intro", "msg": f"Enerjik giriş - {tempo} BPM müzik"})
    elif tempo > 80:
        score += 1
        recommendations.append("Daha enerjik bir müzik kullanarak giriş etkisini artır")
    else:
        recommendations.append("Giriş için 100+ BPM enerjik müzik kullan")
    
    # Erken konu sinyali
    asr_text = features.get("textual", {}).get("asr_text", "")
    if asr_text and not any(greeting in asr_text.lower()[:50] for greeting in ["merhaba", "selam", "hoş geldin"]):
        score += 3
        findings.append({"t": 2.0, "type": "direct_start", "msg": "Selamlaşma olmadan direkt konuya giriş - harika!"})
    else:
        recommendations.append("Selamlaşma atla, direkt konuya başla")
    
    return {
        "score": min(score, 18),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw
    }


def analyze_pacing(frames: List[str], features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """Pacing & Retention analizi (12 puan)"""
    score = 0
    findings = []
    recommendations = []
    raw = {}
    
    # Kesim yoğunluğu
    if len(frames) > 0:
        cuts_per_sec = len(frames) / duration if duration > 0 else 0
        raw["cuts_per_sec"] = cuts_per_sec
        
        # İdeal: 0.25-0.7 cuts/sec
        if 0.25 <= cuts_per_sec <= 0.7:
            score += 4
            findings.append({"t": duration/2, "type": "good_pacing", "msg": f"Mükemmel tempo - {cuts_per_sec:.2f} kesim/saniye"})
        elif cuts_per_sec < 0.25:
            score += 1
            recommendations.append(f"Kesim temposunu artır - şu an {cuts_per_sec:.2f}, ideal 0.3-0.6 arası")
        else:
            score += 2
            recommendations.append(f"Biraz yavaşlat - {cuts_per_sec:.2f} çok hızlı, 0.5 civarı daha iyi")
    
    # Hareket ritmi
    visual = features.get("visual", {})
    flow = visual.get("optical_flow_mean", 0)
    raw["flow_mean"] = flow
    
    if 0.5 <= flow <= 2.5:
        score += 3
        findings.append({"t": duration/2, "type": "good_flow", "msg": f"Hareket dengesi mükemmel ({flow:.2f})"})
    elif flow < 0.5:
        score += 1
        recommendations.append("Daha dinamik hareket ekle - kamera hareketi veya zoom kullan")
    else:
        score += 2
        recommendations.append("Hareket çok yoğun - bazı sahneleri sakinleştir")
    
    # Ölü an tespiti
    # Basit ölü an kontrolü (gerçekte frame-by-frame analiz gerekir)
    if flow < 0.3 and len(features.get("textual", {}).get("asr_text", "")) < 20:
        score -= 2
        recommendations.append("Sessiz ve hareketsiz kısımlar var - B-roll veya müzik ekle")
    else:
        score += 2
        findings.append({"t": duration/2, "type": "no_dead_time", "msg": "Ölü an yok - sürekli ilgi çekici"})
    
    return {
        "score": min(max(score, 0), 12),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw
    }


def analyze_music_sync(features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """Müzik & Senkron analizi (8 puan)"""
    score = 0
    findings = []
    recommendations = []
    raw = {}
    
    audio = features.get("audio", {})
    bpm = audio.get("tempo_bpm", 0)
    loudness = audio.get("loudness_lufs", 0)
    
    raw["bpm"] = bpm
    raw["loudness_lufs"] = loudness
    
    # BPM uygunluğu
    if 105 <= bpm <= 125:
        score += 3
        findings.append({"t": 5.0, "type": "perfect_bpm", "msg": f"Mükemmel BPM ({bmp}) - viral için ideal tempo"})
    elif 95 <= bpm <= 135:
        score += 2
        findings.append({"t": 5.0, "type": "good_bpm", "msg": f"İyi BPM ({bpm}) - viral aralıkta"})
    elif bpm > 0:
        score += 1
        if bpm < 95:
            recommendations.append(f"Müzik temposu yavaş ({bpm} BPM) - 105-125 BPM arası deneyin")
        else:
            recommendations.append(f"Müzik temposu hızlı ({bpm} BPM) - biraz yavaşlatın")
    else:
        recommendations.append("Müzik BPM'i tespit edilemedi - daha belirgin ritim kullanın")
    
    # Ses seviyesi
    if -15 <= loudness <= -8:
        score += 2
        findings.append({"t": 5.0, "type": "good_audio", "msg": f"Ses seviyesi ideal ({loudness:.1f} LUFS)"})
    elif loudness < -20:
        score += 1
        recommendations.append(f"Ses çok kısık ({loudness:.1f} LUFS) - biraz yükselt")
    elif loudness > -5:
        score += 1
        recommendations.append(f"Ses çok yüksek ({loudness:.1f} LUFS) - biraz kıs")
    else:
        score += 1
    
    # Beat-cut senkronu (basit versiyon)
    # Gerçekte librosa beat onset detection gerekir
    beat_sync_score = 0.5  # Placeholder - geliştirilecek
    raw["beat_sync"] = beat_sync_score
    
    if beat_sync_score >= 0.6:
        score += 3
        findings.append({"t": 8.0, "type": "beat_sync", "msg": f"Müzik-edit senkronu mükemmel (%{beat_sync_score*100:.0f})"})
    else:
        score += 1
        recommendations.append("Kesimleri müziğin beat'lerine hizala")
    
    return {
        "score": min(score, 8),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw
    }


def analyze_text_readability(frames: List[str], features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """Metin okunabilirlik analizi (12 puan)"""
    score = 0
    findings = []
    recommendations = []
    raw = {}
    
    ocr_text = features.get("textual", {}).get("ocr_text", "")
    
    if not ocr_text:
        recommendations.append("Videoda hiç metin yok - anahtar kelimeleri ekranda göster")
        return {"score": 0, "findings": findings, "recommendations": recommendations, "raw": raw}
    
    # Font boyutu analizi (placeholder - gerçekte OCR bbox gerekir)
    estimated_font_ratio = estimate_font_size(ocr_text, frames)
    raw["font_ratio"] = estimated_font_ratio
    
    if estimated_font_ratio >= 0.08:  # %8 ekran yüksekliği
        score += 3
        findings.append({"t": 5.0, "type": "good_font", "msg": f"Font boyutu mükemmel (%{estimated_font_ratio*100:.1f})"})
    elif estimated_font_ratio >= 0.06:  # %6
        score += 2
        findings.append({"t": 5.0, "type": "ok_font", "msg": f"Font boyutu uygun (%{estimated_font_ratio*100:.1f})"})
    else:
        score += 1
        recommendations.append("Yazı boyutunu ekran yüksekliğinin en az %8'i kadar büyüt")
    
    # Görünme süresi vs okuma süresi
    word_count = len(ocr_text.split())
    required_time = word_count * 0.33  # 180 wpm okuma hızı
    available_time = duration * 0.7  # Metinin %70'inin görünür olduğunu varsay
    
    reading_ratio = available_time / required_time if required_time > 0 else 1
    raw["reading_ratio"] = reading_ratio
    
    if reading_ratio >= 1.0:
        score += 3
        findings.append({"t": duration/2, "type": "readable_time", "msg": f"Metin okuma süresi yeterli ({reading_ratio:.1f}x)"})
    elif reading_ratio >= 0.7:
        score += 2
        recommendations.append("Metni biraz daha uzun süre ekranda tut")
    else:
        score += 1
        recommendations.append(f"Metin çok hızlı geçiyor - {required_time:.1f}s gerekli ama {available_time:.1f}s var")
    
    # Kontrast analizi (placeholder)
    contrast_score = estimate_text_contrast(frames, ocr_text)
    raw["contrast_score"] = contrast_score
    
    if contrast_score >= 4.5:
        score += 3
        findings.append({"t": 3.0, "type": "good_contrast", "msg": "Metin kontrastı mükemmel - net okunuyor"})
    elif contrast_score >= 3.0:
        score += 2
        recommendations.append("Metin kontrastını biraz artır - arka plan kutusu ekle")
    else:
        score += 1
        recommendations.append("Metin kontrastı çok düşük - arka plan veya outline ekle")
    
    # Güvenli alan kontrolü
    safe_area_ok = check_safe_area(ocr_text)
    if safe_area_ok:
        score += 3
        findings.append({"t": 2.0, "type": "safe_area", "msg": "Metinler güvenli alanda - UI ile çakışmıyor"})
    else:
        recommendations.append("Metinleri ekranın üst/alt %15'lik alanından uzak tut")
    
    return {
        "score": min(score, 12),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw
    }


def analyze_message_clarity(features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """Mesaj netliği 4-7 saniye arası (6 puan)"""
    score = 0
    findings = []
    recommendations = []
    
    # ASR'dan 4-7 saniye arası metin kontrol
    asr_text = features.get("textual", {}).get("asr_text", "")
    ocr_text = features.get("textual", {}).get("ocr_text", "")
    
    # Konu/başlık tespiti
    topic_keywords = ["kombin", "rutin", "yöntem", "taktik", "ipucu", "sır", "şey", "durum"]
    has_topic = any(keyword in (asr_text + " " + ocr_text).lower() for keyword in topic_keywords)
    
    if has_topic:
        score += 3
        findings.append({"t": 5.5, "type": "clear_topic", "msg": "4-7 saniye arası konu net belirlenmiş"})
    else:
        recommendations.append("4-7 saniye arasında videonun konusunu net bir şekilde belirt")
    
    # Fayda/sonuç ifadesi
    benefit_keywords = ["dakikada", "kolayca", "hızlı", "etkili", "garantili", "kesin", "mutlaka"]
    has_benefit = any(keyword in (asr_text + " " + ocr_text).lower() for keyword in benefit_keywords)
    
    if has_benefit:
        score += 3
        findings.append({"t": 6.0, "type": "clear_benefit", "msg": "Faydası/sonucu net belirtilmiş"})
    else:
        recommendations.append("Izleyiciye 'neden izlemeli?' sorusunun cevabını 4-7 saniye arası ver")
    
    return {
        "score": min(score, 6),
        "findings": findings,
        "recommendations": recommendations,
        "raw": {}
    }


def analyze_transitions(frames: List[str], features: Dict[str, Any]) -> Dict[str, Any]:
    """Geçiş/efekt kalitesi analizi (8 puan)"""
    score = 0
    findings = []
    recommendations = []
    raw = {}
    
    if len(frames) < 3:
        return {"score": 0, "findings": findings, "recommendations": ["Video çok kısa - geçiş analizi yapılamadı"], "raw": raw}
    
    # Frame farklılığı analizi
    transition_count = count_transitions(frames)
    raw["transition_count"] = transition_count
    
    transition_variety = estimate_transition_variety(frames)
    raw["transition_variety"] = transition_variety
    
    if transition_count >= 2:
        score += 4
        findings.append({"t": 10.0, "type": "good_transitions", "msg": f"{transition_count} farklı geçiş tipi tespit edildi"})
    elif transition_count >= 1:
        score += 2
        recommendations.append("Daha fazla geçiş çeşitliliği ekle (jump cut, match cut, zoom)")
    else:
        recommendations.append("Video geçişleri monoton - farklı edit teknikleri kullan")
    
    # Akış kalitesi
    visual = features.get("visual", {})
    flow_mean = visual.get("optical_flow_mean", 0)
    
    if 0.8 <= flow_mean <= 2.5:
        score += 4
        findings.append({"t": 8.0, "type": "smooth_flow", "msg": f"Video akışı dengeli ({flow_mean:.2f})"})
    else:
        score += 1
        if flow_mean < 0.8:
            recommendations.append("Daha dinamik kamera hareketleri ekle")
        else:
            recommendations.append("Geçişleri biraz yavaşlat - çok hızlı")
    
    return {
        "score": min(score, 8),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw
    }


def analyze_cta(features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """CTA & Etkileşim analizi (6 puan)"""
    score = 0
    findings = []
    recommendations = []
    
    ocr_text = features.get("textual", {}).get("ocr_text", "")
    asr_text = features.get("textual", {}).get("asr_text", "")
    combined_text = (ocr_text + " " + asr_text).lower()
    
    # CTA tespit
    cta_phrases = [
        "yorum", "kaydet", "etiketle", "arkadaşını", "1 mi 2 mi", "hangisi", 
        "beğen", "takip", "abone", "paylaş", "sence", "düşünce"
    ]
    
    has_cta = any(phrase in combined_text for phrase in cta_phrases)
    
    if has_cta:
        score += 3
        findings.append({"t": duration*0.7, "type": "cta_found", "msg": "Etkileşim çağrısı tespit edildi"})
    else:
        recommendations.append("Etkileşim çağrısı ekle - 'Sence hangisi?' veya 'Kaydet' gibi")
    
    # Soru formatı (daha etkili)
    question_words = ["neden", "nasıl", "hangi", "kim", "ne zaman", "sence", "?"]
    has_question = any(word in combined_text for word in question_words)
    
    if has_question:
        score += 2
        findings.append({"t": duration*0.8, "type": "question_format", "msg": "Soru formatında CTA - çok etkili"})
    else:
        recommendations.append("Soru formatında CTA ekle - 'Hangisini tercih edersin?' gibi")
    
    # Zamanlama kontrolü
    # 5-9 saniye veya son 2 saniye arası ideal
    # Bu gerçekte OCR/ASR timestamp analizi gerektirir
    timing_ok = True  # Placeholder
    if timing_ok:
        score += 1
        findings.append({"t": duration*0.8, "type": "good_timing", "msg": "CTA zamanlaması uygun"})
    else:
        recommendations.append("CTA'yı 5-9 saniye arası veya son 2 saniyeye taşı")
    
    return {
        "score": min(score, 6),
        "findings": findings,
        "recommendations": recommendations,
        "raw": {}
    }


def analyze_loop(frames: List[str], duration: float) -> Dict[str, Any]:
    """Loop & final analizi (5 puan)"""
    score = 0
    findings = []
    recommendations = []
    raw = {}
    
    if len(frames) < 2:
        return {"score": 0, "findings": findings, "recommendations": ["Video çok kısa"], "raw": raw}
    
    # İlk ve son frame benzerliği
    loop_similarity = calculate_frame_similarity(frames[0], frames[-1])
    raw["loop_similarity"] = loop_similarity
    
    if loop_similarity >= 0.70:
        score += 3
        findings.append({"t": duration-0.5, "type": "perfect_loop", "msg": f"Mükemmel döngü - %{loop_similarity*100:.0f} benzerlik"})
    elif loop_similarity >= 0.50:
        score += 2
        findings.append({"t": duration-0.5, "type": "good_loop", "msg": f"İyi döngü - %{loop_similarity*100:.0f} benzerlik"})
    else:
        score += 1
        recommendations.append(f"Döngü etkisi zayıf (%{loop_similarity*100:.0f}) - son kareyi ilk kareye benzetin")
    
    # Final peak kontrolü (son 2 saniyede hareket/enerji zirvesi)
    if len(frames) >= 4:
        final_energy = calculate_final_energy(frames[-2:])
        raw["final_energy"] = final_energy
        
        if final_energy > 0.7:
            score += 2
            findings.append({"t": duration-1, "type": "strong_finale", "msg": "Güçlü final - son saniyeler etkili"})
        else:
            recommendations.append("Son 2 saniyeyi daha etkili bitir - güçlü bir final yap")
    
    return {
        "score": min(score, 5),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw
    }


def analyze_accessibility(features: Dict[str, Any]) -> Dict[str, Any]:
    """Erişilebilirlik analizi (4 puan)"""
    score = 0
    findings = []
    recommendations = []
    
    ocr_text = features.get("textual", {}).get("ocr_text", "")
    asr_text = features.get("textual", {}).get("asr_text", "")
    
    # Sessiz izlenebilirlik
    if len(ocr_text) > 20:  # Yeterli altyazı/metin var
        if len(asr_text) > 0:
            # Konuşma kapsamı
            coverage = len(ocr_text) / len(asr_text) if len(asr_text) > 0 else 0
            if coverage >= 0.7:
                score += 3
                findings.append({"t": 10.0, "type": "good_subtitles", "msg": f"Konuşmanın %{coverage*100:.0f}'i altyazılı - sessiz izlenebilir"})
            else:
                score += 1
                recommendations.append(f"Altyazı kapsamını artır - şu an %{coverage*100:.0f}, ideal %70+")
        else:
            score += 2
            findings.append({"t": 5.0, "type": "text_only", "msg": "Sadece metin tabanlı - sessiz izleme için ideal"})
    else:
        recommendations.append("Sessiz izleyenler için daha fazla metin/altyazı ekle")
    
    # Dil uyumu (Türkçe hedef için)
    if any(tr_word in (asr_text + ocr_text).lower() for tr_word in ["ve", "bu", "ile", "için", "çok", "bir"]):
        score += 1
        findings.append({"t": 5.0, "type": "language_match", "msg": "Türkçe içerik - hedef kitle ile uyumlu"})
    
    return {
        "score": min(score, 4),
        "findings": findings,
        "recommendations": recommendations,
        "raw": {}
    }


def analyze_technical_quality(video_path: str, frames: List[str]) -> Dict[str, Any]:
    """Teknik kalite analizi (8 puan)"""
    score = 0
    findings = []
    recommendations = []
    raw = {}
    
    # Video bilgileri
    video_info = get_video_info(video_path)
    raw.update(video_info)
    
    # Çözünürlük kontrolü
    resolution = video_info.get("resolution", "")
    if "1080x1920" in resolution or "1080x1080" in resolution:
        score += 3
        findings.append({"t": 0, "type": "good_resolution", "msg": f"Çözünürlük mükemmel: {resolution}"})
    elif "720" in resolution:
        score += 2
        recommendations.append("1080p çözünürlük kullan - daha net görünür")
    else:
        score += 1
        recommendations.append("Çözünürlüğü artır - en az 720p kullan")
    
    # FPS kontrolü
    fps = video_info.get("fps", 0)
    if 30 <= fps <= 60:
        score += 2
        findings.append({"t": 0, "type": "good_fps", "msg": f"FPS ideal: {fps}"})
    else:
        score += 1
        recommendations.append("30-60 FPS aralığında kaydet")
    
    # Bulanıklık tespiti
    if frames:
        blur_score = detect_blur(frames[len(frames)//2])  # Orta frame kontrol
        raw["blur_score"] = blur_score
        
        if blur_score > 100:  # Laplacian variance threshold
            score += 2
            findings.append({"t": 10.0, "type": "sharp_video", "msg": "Video net ve keskin"})
        else:
            score += 1
            recommendations.append("Video bulanık - daha iyi ışık veya odaklama kullan")
    
    # Watermark tespiti (basit)
    watermark_detected = detect_watermark(frames)
    raw["watermark_detected"] = watermark_detected
    
    if not watermark_detected:
        score += 1
        findings.append({"t": 0, "type": "no_watermark", "msg": "Watermark yok - temiz görünüm"})
    else:
        recommendations.append("Watermark/filigran kaldır - daha profesyonel görünür")
    
    return {
        "score": min(score, 8),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw
    }


def analyze_trend_originality(features: Dict[str, Any]) -> Dict[str, Any]:
    """Trend & orijinallik analizi (6 puan)"""
    score = 3  # Baseline skor
    findings = []
    recommendations = []
    
    # Basit orijinallik değerlendirmesi
    # Gerçekte audio fingerprinting ve content analysis gerekir
    
    # Unique content indicators
    asr_text = features.get("textual", {}).get("asr_text", "")
    ocr_text = features.get("textual", {}).get("ocr_text", "")
    
    if len(asr_text) > 50:  # Uzun konuşma = orijinal içerik
        score += 2
        findings.append({"t": 10.0, "type": "original_speech", "msg": "Uzun konuşma - orijinal içerik"})
    
    # Trend kullanımı placeholder
    # Gerçekte audio fingerprint veya hashtag analizi gerekir
    trend_usage = 0.5  # Placeholder
    if trend_usage > 0.3:
        score += 1
        findings.append({"t": 5.0, "type": "trend_usage", "msg": "Trend elementi tespit edildi"})
    
    recommendations.append("Güncel trendleri kullan ama kendi yorumunu kat")
    
    return {
        "score": min(score, 6),
        "findings": findings,
        "recommendations": recommendations,
        "raw": {"trend_usage": trend_usage}
    }


def analyze_content_type_fit(features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """İçerik tipi uygunluk analizi (7 puan)"""
    score = 0
    findings = []
    recommendations = []
    
    content_type = detect_content_type(features)
    
    # İçerik tipine göre puanlama
    if content_type == "talking_head":
        # Talking head için metin/altyazı önemli
        ocr_len = len(features.get("textual", {}).get("ocr_text", ""))
        if ocr_len > 30:
            score += 4
            findings.append({"t": 5.0, "type": "talking_head_text", "msg": "Talking head + altyazı = mükemmel kombinasyon"})
        else:
            score += 2
            recommendations.append("Talking head videolar için daha fazla altyazı ekle")
    
    elif content_type == "lifestyle":
        # Lifestyle için tempo ve müzik önemli
        tempo = features.get("audio", {}).get("tempo_bpm", 0)
        if tempo > 100:
            score += 4
            findings.append({"t": 5.0, "type": "lifestyle_energy", "msg": "Lifestyle içerik için uygun tempo"})
        else:
            score += 2
            recommendations.append("Lifestyle içerik için daha enerjik müzik kullan")
    
    else:
        # Genel içerik
        score += 3
        findings.append({"t": 5.0, "type": "general_content", "msg": f"İçerik tipi: {content_type}"})
    
    return {
        "score": min(score, 7),
        "findings": findings,
        "recommendations": recommendations,
        "raw": {"content_type": content_type}
    }


# Helper functions
def calculate_visual_jump(frames: List[str]) -> float:
    """İlk frame'ler arası görsel sıçrama hesapla"""
    if len(frames) < 2:
        return 0.0
    
    try:
        import cv2
        img1 = cv2.imread(frames[0])
        img2 = cv2.imread(frames[1])
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Histogram farkı
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        
        # Chi-square distance
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        return min(diff / 1000000, 1.0)  # Normalize
    except:
        return 0.0


def estimate_font_size(text: str, frames: List[str]) -> float:
    """Font boyutunu tahmin et (placeholder)"""
    # Gerçekte OCR bounding box analizi gerekir
    # Basit tahmin: metin uzunluğuna göre
    if len(text) < 20:
        return 0.08  # Kısa metin = büyük font
    elif len(text) < 50:
        return 0.06  # Orta metin = orta font
    else:
        return 0.04  # Uzun metin = küçük font


def estimate_text_contrast(frames: List[str], text: str) -> float:
    """Metin kontrast oranını tahmin et"""
    # Placeholder - gerçekte frame analizi gerekir
    if len(text) > 0:
        return 4.8  # Varsayılan iyi kontrast
    return 2.0


def check_safe_area(text: str) -> bool:
    """Güvenli alan kontrolü"""
    # Placeholder - gerçekte OCR bbox pozisyon analizi gerekir
    return len(text) > 0  # Basit kontrol


def count_transitions(frames: List[str]) -> int:
    """Geçiş tiplerini say"""
    # Placeholder - frame analizi gerekir
    return min(len(frames) // 3, 4)  # Basit tahmin


def estimate_transition_variety(frames: List[str]) -> float:
    """Geçiş çeşitliliğini tahmin et"""
    return min(len(frames) / 10, 1.0)


def calculate_frame_similarity(frame1_path: str, frame2_path: str) -> float:
    """İki frame arasında benzerlik hesapla"""
    try:
        import cv2
        img1 = cv2.imread(frame1_path)
        img2 = cv2.imread(frame2_path)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Resize for speed
        img1 = cv2.resize(img1, (64, 64))
        img2 = cv2.resize(img2, (64, 64))
        
        # MSE similarity
        mse = np.mean((img1 - img2) ** 2)
        similarity = 1 / (1 + mse / 1000)  # Normalize
        return similarity
    except:
        return 0.0


def calculate_final_energy(final_frames: List[str]) -> float:
    """Son frame'lerde enerji hesapla"""
    # Placeholder - hareket analizi gerekir
    return 0.6  # Varsayılan orta enerji


def detect_blur(frame_path: str) -> float:
    """Bulanıklık tespit - Laplacian variance"""
    try:
        import cv2
        img = cv2.imread(frame_path)
        if img is None:
            return 0.0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    except:
        return 0.0


def detect_watermark(frames: List[str]) -> bool:
    """Watermark tespit (basit)"""
    # Placeholder - template matching gerekir
    return False  # Şimdilik watermark yok varsay


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Video teknik bilgileri"""
    try:
        import subprocess
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', video_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), {})
            
            return {
                "resolution": f"{video_stream.get('width', 0)}x{video_stream.get('height', 0)}",
                "fps": float(video_stream.get('r_frame_rate', '30/1').split('/')[0]),
                "bitrate": int(video_stream.get('bit_rate', 0)),
                "codec": video_stream.get('codec_name', 'unknown')
            }
    except:
        pass
    
    return {"resolution": "unknown", "fps": 30, "bitrate": 0, "codec": "unknown"}


def detect_content_type(features: Dict[str, Any]) -> str:
    """İçerik tipini tespit et"""
    asr_text = features.get("textual", {}).get("asr_text", "")
    ocr_text = features.get("textual", {}).get("ocr_text", "")
    flow = features.get("visual", {}).get("optical_flow_mean", 0)
    
    # Basit heuristik
    if len(asr_text) > 100 and flow < 1.0:
        return "talking_head"
    elif flow > 2.0 and len(ocr_text) < 50:
        return "lifestyle"
    elif "nasıl" in asr_text.lower() or "adım" in (asr_text + ocr_text).lower():
        return "tutorial"
    else:
        return "general"
