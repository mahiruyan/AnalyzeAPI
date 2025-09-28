"""
Advanced Video Analysis - Gerçek implementasyonlar
12 metrik detaylı analiz sistemi
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
import json
from pathlib import Path
import subprocess

def analyze_hook(video_path: str, frames: List[str], features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """
    Hook Analizi - İlk 3 saniye detaylı analiz (18 puan)
    Gerçek OpenCV ve frame analizi ile
    """
    score = 0
    findings = []
    recommendations = []
    raw_metrics = {}
    
    print("🎯 [HOOK] Starting hook analysis...")
    
    if len(frames) < 3 or duration < 3:
        return {
            "score": 0,
            "findings": [{"t": 0, "msg": "Video 3 saniyeden kısa - hook analizi yapılamadı"}],
            "recommendations": ["Video en az 3 saniye olmalı"],
            "raw": {}
        }
    
    # 1. Görsel Sıçrama Analizi (Frame 0 vs Frame 1-2)
    visual_jump_score = analyze_visual_jump(frames[:3])
    raw_metrics["visual_jump"] = visual_jump_score
    
    if visual_jump_score > 0.4:
        score += 5
        findings.append({
            "t": 0.5, 
            "type": "strong_visual_interrupt",
            "msg": f"Güçlü görsel sıçrama ({visual_jump_score:.2f}) - dikkat çekici pattern interrupt!"
        })
    elif visual_jump_score > 0.2:
        score += 3
        findings.append({
            "t": 0.5,
            "type": "moderate_visual_change", 
            "msg": f"Orta seviye görsel değişim ({visual_jump_score:.2f})"
        })
        recommendations.append("İlk saniyede daha keskin görsel değişim ekle (zoom, renk, açı)")
    else:
        score += 1
        recommendations.append("İlk 3 saniyede ani zoom, renk değişimi veya yakın plan ekle")
    
    # 2. Metin Kancası Analizi (OCR)
    hook_text_score = analyze_hook_text(features)
    raw_metrics["hook_text_strength"] = hook_text_score
    
    if hook_text_score > 0.7:
        score += 4
        findings.append({
            "t": 1.0,
            "type": "powerful_hook_text",
            "msg": "Güçlü kanca metni tespit edildi - merak uyandırıcı!"
        })
    elif hook_text_score > 0.4:
        score += 2
        recommendations.append("Kanca metnini daha güçlü yap - 'Kimse söylemiyor ama...' tarzı")
    else:
        recommendations.append("İlk 3 saniyede güçlü bir kanca metni ekle")
    
    # 3. Ses/Beat Giriş Analizi
    audio_intro_score = analyze_audio_intro(features)
    raw_metrics["audio_intro_energy"] = audio_intro_score
    
    if audio_intro_score > 0.6:
        score += 3
        findings.append({
            "t": 1.5,
            "type": "energetic_intro",
            "msg": "Enerjik ses girişi - beat drop veya enerji artışı var"
        })
    elif audio_intro_score > 0.3:
        score += 1
        recommendations.append("Giriş için daha enerjik müzik veya beat drop kullan")
    else:
        recommendations.append("İlk 2 saniyede belirgin bir beat/enerji artışı ekle")
    
    # 4. Hemen Konu Girişi (Anti-greeting)
    direct_start_score = analyze_direct_start(features)
    raw_metrics["direct_start"] = direct_start_score
    
    if direct_start_score > 0.8:
        score += 3
        findings.append({
            "t": 2.0,
            "type": "direct_topic_start",
            "msg": "Mükemmel! Selamlaşma olmadan direkt konuya giriş"
        })
    elif direct_start_score > 0.5:
        score += 2
        recommendations.append("Selamlaşmayı kısalt, daha hızlı konuya geç")
    else:
        score += 1
        recommendations.append("'Merhaba' yerine direkt konuyla başla")
    
    # 5. Erken Foreshadowing (Gelecek vaat)
    foreshadow_score = analyze_foreshadowing(features)
    raw_metrics["foreshadowing"] = foreshadow_score
    
    if foreshadow_score > 0.6:
        score += 3
        findings.append({
            "t": 2.5,
            "type": "good_foreshadowing",
            "msg": "İyi foreshadowing - 'Sonunda göreceksin' tarzı vaat var"
        })
    else:
        recommendations.append("0-3 saniye arası 'Sonunda X'i göreceksin' tarzı vaat ekle")
    
    print(f"🎯 [HOOK] Hook score: {score}/18")
    
    return {
        "score": min(score, 18),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw_metrics
    }


def analyze_visual_jump(frames: List[str]) -> float:
    """
    Görsel sıçrama analizi - gerçek OpenCV implementation
    İlk frame'ler arası histogram ve edge farkı
    """
    if len(frames) < 2:
        return 0.0
    
    try:
        # Frame'leri oku
        img1 = cv2.imread(frames[0])
        img2 = cv2.imread(frames[1])
        
        if img1 is None or img2 is None:
            print("⚠️ [HOOK] Frame okuma hatası")
            return 0.0
        
        # Resize for faster processing
        h, w = img1.shape[:2]
        target_size = (w//4, h//4)  # 1/4 boyut
        img1_small = cv2.resize(img1, target_size)
        img2_small = cv2.resize(img2, target_size)
        
        # 1. Histogram farkı (renk değişimi)
        hist1 = cv2.calcHist([img1_small], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2_small], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        
        # Chi-square distance
        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        hist_score = min(hist_diff / 50000, 1.0)  # Normalize
        
        # 2. Edge farkı (şekil/kompozisyon değişimi)
        gray1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)
        
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        
        edge_diff = np.mean(np.abs(edges1.astype(float) - edges2.astype(float))) / 255.0
        
        # 3. Optical Flow (hareket şiddeti)
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, 
                                       corners=cv2.goodFeaturesToTrack(gray1, maxCorners=50, qualityLevel=0.3, minDistance=7),
                                       nextPts=None)[0]
        
        flow_magnitude = 0.0
        if flow is not None and len(flow) > 0:
            # Hareket büyüklüğü
            flow_magnitude = np.mean(np.sqrt(np.sum(flow**2, axis=1)))
            flow_magnitude = min(flow_magnitude / 20.0, 1.0)  # Normalize
        
        # Birleşik skor
        combined_score = (hist_score * 0.4) + (edge_diff * 0.3) + (flow_magnitude * 0.3)
        
        print(f"🎯 [HOOK] Visual jump - hist: {hist_score:.2f}, edge: {edge_diff:.2f}, flow: {flow_magnitude:.2f}, combined: {combined_score:.2f}")
        
        return combined_score
        
    except Exception as e:
        print(f"❌ [HOOK] Visual jump analysis failed: {e}")
        return 0.0


def analyze_hook_text(features: Dict[str, Any]) -> float:
    """
    Kanca metni analizi - gerçek OCR text analysis
    """
    ocr_text = features.get("textual", {}).get("ocr_text", "").lower()
    
    if not ocr_text:
        return 0.0
    
    # Güçlü kanca kelimeleri (ağırlıklı)
    hook_keywords = {
        "kimse": 0.9,
        "gizli": 0.8, 
        "sır": 0.8,
        "şok": 0.7,
        "inanamayacaksın": 0.9,
        "bu yöntem": 0.6,
        "hiç kimse": 0.9,
        "asla": 0.7,
        "kesinlikle": 0.6,
        "muhtemelen": 0.5,
        "bilmiyor": 0.7,
        "fark": 0.6,
        "özel": 0.5,
        "yeni": 0.4,
        "ilk kez": 0.7,
        "sadece": 0.6
    }
    
    # Soru formatları (çok etkili)
    question_patterns = [
        "neden", "nasıl", "ne zaman", "hangi", "kim", 
        "niçin", "nereden", "ne kadar", "kaç"
    ]
    
    # Kanca skoru hesapla
    score = 0.0
    
    # Keyword matching
    for keyword, weight in hook_keywords.items():
        if keyword in ocr_text:
            score += weight
            print(f"🎯 [HOOK] Hook keyword found: '{keyword}' (+{weight})")
    
    # Soru formatı bonus
    question_bonus = 0.0
    for pattern in question_patterns:
        if pattern in ocr_text and "?" in ocr_text:
            question_bonus = 0.3
            print(f"🎯 [HOOK] Question format found: '{pattern}' (+0.3)")
            break
    
    score += question_bonus
    
    # Kelime sayısı kontrolü (ideal: 3-10 kelime)
    word_count = len(ocr_text.split())
    if 3 <= word_count <= 10:
        score += 0.2
        print(f"🎯 [HOOK] Good word count: {word_count} words (+0.2)")
    elif word_count > 15:
        score -= 0.3
        print(f"🎯 [HOOK] Too many words: {word_count} (penalty)")
    
    # Normalize (0-1)
    final_score = min(score / 1.5, 1.0)  # Max 1.5 puan normalize to 1.0
    
    print(f"🎯 [HOOK] Hook text analysis: '{ocr_text[:50]}...' = {final_score:.2f}")
    
    return final_score


def analyze_audio_intro(features: Dict[str, Any]) -> float:
    """
    Ses giriş analizi - BPM ve enerji
    """
    audio = features.get("audio", {})
    bpm = audio.get("tempo_bpm", 0)
    loudness = audio.get("loudness_lufs", 0)
    
    score = 0.0
    
    # BPM enerji değerlendirmesi
    if bpm > 120:
        score += 0.4  # Yüksek enerji
        print(f"🎯 [HOOK] High energy BPM: {bmp}")
    elif bpm > 100:
        score += 0.3  # Orta-yüksek enerji
    elif bpm > 80:
        score += 0.2  # Orta enerji
    else:
        print(f"🎯 [HOOK] Low energy BPM: {bpm}")
    
    # Ses seviyesi (intro için önemli)
    if -15 <= loudness <= -8:
        score += 0.3  # İdeal seviye
        print(f"🎯 [HOOK] Good intro volume: {loudness:.1f} LUFS")
    elif loudness < -20:
        score += 0.1  # Çok kısık
        print(f"🎯 [HOOK] Intro too quiet: {loudness:.1f} LUFS")
    elif loudness > -5:
        score += 0.1  # Çok yüksek
        print(f"🎯 [HOOK] Intro too loud: {loudness:.1f} LUFS")
    else:
        score += 0.2
    
    print(f"🎯 [HOOK] Audio intro score: {score:.2f}")
    return min(score, 1.0)


def analyze_direct_start(features: Dict[str, Any]) -> float:
    """
    Direkt başlangıç analizi - selamlaşma vs konu
    """
    asr_text = features.get("textual", {}).get("asr_text", "").lower()
    
    if not asr_text:
        return 0.5  # Konuşma yok, nötr
    
    # İlk 50 karakter (yaklaşık ilk 3 saniye konuşma)
    first_part = asr_text[:50]
    
    # Selamlaşma kalıpları (negatif)
    greetings = [
        "merhaba", "selam", "hoş geldin", "kanalıma", "ben", "adım",
        "bugün", "video", "hepinize", "arkadaşlar", "izleyiciler"
    ]
    
    # Direkt konu kalıpları (pozitif)
    direct_patterns = [
        "şunu", "bunu", "bu yöntem", "bu taktik", "bu durum",
        "geçen", "dün", "şimdi", "hemen", "direkt"
    ]
    
    greeting_count = sum(1 for g in greetings if g in first_part)
    direct_count = sum(1 for d in direct_patterns if d in first_part)
    
    # Skor hesapla
    if direct_count > 0 and greeting_count == 0:
        score = 1.0  # Mükemmel direkt başlangıç
        print(f"🎯 [HOOK] Perfect direct start - no greetings, direct topic")
    elif direct_count > greeting_count:
        score = 0.7  # İyi, daha çok konu odaklı
        print(f"🎯 [HOOK] Good direct start - more topic than greeting")
    elif greeting_count <= 1:
        score = 0.6  # Kısa selamlaşma, kabul edilebilir
        print(f"🎯 [HOOK] Short greeting, acceptable")
    else:
        score = 0.3  # Uzun selamlaşma
        print(f"🎯 [HOOK] Too much greeting - {greeting_count} greeting words")
    
    return score


def analyze_foreshadowing(features: Dict[str, Any]) -> float:
    """
    Foreshadowing analizi - gelecek vaat
    """
    ocr_text = features.get("textual", {}).get("ocr_text", "").lower()
    asr_text = features.get("textual", {}).get("asr_text", "").lower()
    combined_text = ocr_text + " " + asr_text
    
    # Foreshadowing kalıpları
    foreshadow_patterns = [
        "sonunda", "göreceksin", "öğreneceksin", "anlayacaksın",
        "şaşıracaksın", "inanamayacaksın", "keşfedeceksin",
        "final", "sonda", "en son", "surprise", "plot twist"
    ]
    
    # Vaat kalıpları
    promise_patterns = [
        "garanti", "kesin", "mutlaka", "illa", "확실히", 
        "works", "çalışır", "effect", "sonuç"
    ]
    
    score = 0.0
    
    # Foreshadowing tespit
    for pattern in foreshadow_patterns:
        if pattern in combined_text:
            score += 0.4
            print(f"🎯 [HOOK] Foreshadowing found: '{pattern}'")
            break
    
    # Vaat tespit
    for pattern in promise_patterns:
        if pattern in combined_text:
            score += 0.3
            print(f"🎯 [HOOK] Promise found: '{pattern}'")
            break
    
    print(f"🎯 [HOOK] Foreshadowing score: {score:.2f}")
    return min(score, 1.0)


def analyze_pacing_retention(frames: List[str], features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """
    Pacing & Retention Analizi (12 puan)
    Gerçek cut detection ve hareket analizi
    """
    score = 0
    findings = []
    recommendations = []
    raw_metrics = {}
    
    print("⚡ [PACING] Starting pacing analysis...")
    
    # 1. Kesim Yoğunluğu Analizi
    cuts_per_sec = detect_cuts_per_second(frames, duration)
    raw_metrics["cuts_per_sec"] = cuts_per_sec
    
    # İdeal aralık: 0.25-0.7 cuts/sec (lifestyle için)
    if 0.25 <= cuts_per_sec <= 0.7:
        score += 4
        findings.append({
            "t": duration/2,
            "type": "perfect_pacing",
            "msg": f"Mükemmel kesim temposu: {cuts_per_sec:.2f} kesim/saniye"
        })
    elif 0.15 <= cuts_per_sec < 0.25:
        score += 2
        recommendations.append(f"Kesim temposunu artır - şu an {cuts_per_sec:.2f}, ideal 0.3-0.6 arası")
    elif 0.7 < cuts_per_sec <= 1.0:
        score += 3
        recommendations.append(f"Biraz yavaşlat - {cuts_per_sec:.2f} çok hızlı, 0.5 civarı daha iyi")
    else:
        score += 1
        if cuts_per_sec < 0.15:
            recommendations.append("Çok yavaş edit - daha fazla kesim ekle")
        else:
            recommendations.append("Çok hızlı edit - izleyici kaybedebilir")
    
    # 2. Hareket Ritmi Analizi
    movement_rhythm = analyze_movement_rhythm(frames)
    raw_metrics["movement_rhythm"] = movement_rhythm
    
    if movement_rhythm > 0.7:
        score += 3
        findings.append({
            "t": duration*0.6,
            "type": "rhythmic_movement",
            "msg": f"Ritmik hareket paterni mükemmel ({movement_rhythm:.2f})"
        })
    elif movement_rhythm > 0.4:
        score += 2
        recommendations.append("Hareket ritmini biraz daha düzenli yap")
    else:
        score += 1
        recommendations.append("Daha ritmik hareket paterni oluştur - düzenli aralıklarla hareket")
    
    # 3. Ölü An Tespiti
    dead_time_penalty = detect_dead_time(frames, features, duration)
    raw_metrics["dead_time_penalty"] = dead_time_penalty
    
    if dead_time_penalty == 0:
        score += 3
        findings.append({
            "t": duration*0.7,
            "type": "no_dead_time",
            "msg": "Hiç ölü an yok - sürekli ilgi çekici content"
        })
    else:
        score -= dead_time_penalty
        recommendations.append(f"Ölü anları ({dead_time_penalty} tespit) B-roll veya müzikle doldur")
    
    # 4. Retention Curve Tahmini
    retention_estimate = estimate_retention_curve(frames, features)
    raw_metrics["retention_estimate"] = retention_estimate
    
    if retention_estimate > 0.75:
        score += 2
        findings.append({
            "t": duration*0.8,
            "type": "high_retention",
            "msg": f"Yüksek retention tahmini (%{retention_estimate*100:.0f})"
        })
    elif retention_estimate > 0.5:
        score += 1
    else:
        recommendations.append("Retention'ı artır - daha sık hareket ve metin kullan")
    
    print(f"⚡ [PACING] Pacing score: {score}/12")
    
    return {
        "score": min(max(score, 0), 12),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw_metrics
    }


def detect_cuts_per_second(frames: List[str], duration: float) -> float:
    """
    Gerçek cut detection - histogram based
    """
    if len(frames) <= 1 or duration <= 0:
        return 0.0
    
    try:
        cuts = 0
        threshold = 0.3  # Histogram difference threshold
        
        for i in range(len(frames) - 1):
            img1 = cv2.imread(frames[i])
            img2 = cv2.imread(frames[i + 1])
            
            if img1 is None or img2 is None:
                continue
            
            # Histogram farkı
            hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            # Chi-square distance
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            normalized_diff = diff / 10000.0  # Normalize
            
            if normalized_diff > threshold:
                cuts += 1
                print(f"⚡ [PACING] Cut detected at frame {i}: diff={normalized_diff:.3f}")
        
        cuts_per_sec = cuts / duration
        print(f"⚡ [PACING] Total cuts: {cuts}, Duration: {duration:.1f}s, Rate: {cuts_per_sec:.3f}")
        
        return cuts_per_sec
        
    except Exception as e:
        print(f"❌ [PACING] Cut detection failed: {e}")
        return len(frames) / duration  # Fallback: frame rate as cut rate


def analyze_movement_rhythm(frames: List[str]) -> float:
    """
    Hareket ritmi analizi - optical flow patterns
    """
    if len(frames) < 4:
        return 0.0
    
    try:
        movement_values = []
        
        for i in range(len(frames) - 1):
            img1 = cv2.imread(frames[i])
            img2 = cv2.imread(frames[i + 1])
            
            if img1 is None or img2 is None:
                movement_values.append(0.0)
                continue
            
            # Grayscale ve resize
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.resize(gray1, (64, 64))
            gray2 = cv2.resize(gray2, (64, 64))
            
            # Optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                gray1, gray2, 
                corners=cv2.goodFeaturesToTrack(gray1, maxCorners=20, qualityLevel=0.3, minDistance=7),
                nextPts=None
            )[0]
            
            # Hareket büyüklüğü
            if flow is not None and len(flow) > 0:
                movement = np.mean(np.sqrt(np.sum(flow**2, axis=1)))
                movement_values.append(movement)
            else:
                movement_values.append(0.0)
        
        if not movement_values:
            return 0.0
        
        # Ritim analizi - variance ve peaks
        movement_array = np.array(movement_values)
        movement_std = np.std(movement_array)
        movement_mean = np.mean(movement_array)
        
        # Ritmik varsa std yüksek ama çok wild değil
        rhythm_score = 0.0
        if movement_mean > 0.1:  # En az biraz hareket var
            if 0.3 <= movement_std/movement_mean <= 1.2:  # Coefficient of variation
                rhythm_score = 0.8  # İyi ritim
                print(f"⚡ [PACING] Good movement rhythm: std/mean = {movement_std/movement_mean:.2f}")
            else:
                rhythm_score = 0.4  # Düzensiz
                print(f"⚡ [PACING] Irregular movement: std/mean = {movement_std/movement_mean:.2f}")
        else:
            rhythm_score = 0.1  # Çok az hareket
            print(f"⚡ [PACING] Very low movement: mean = {movement_mean:.3f}")
        
        return rhythm_score
        
    except Exception as e:
        print(f"❌ [PACING] Movement rhythm analysis failed: {e}")
        return 0.0


def detect_dead_time(frames: List[str], features: Dict[str, Any], duration: float) -> int:
    """
    Ölü an tespiti - hareket yok + ses yok + metin yok
    """
    # Basit versiyon - geliştirilecek
    visual = features.get("visual", {})
    flow = visual.get("optical_flow_mean", 0)
    
    asr_text = features.get("textual", {}).get("asr_text", "")
    ocr_text = features.get("textual", {}).get("ocr_text", "")
    
    penalty = 0
    
    # Çok düşük hareket + az metin = ölü an
    if flow < 0.2 and len(asr_text) < 20 and len(ocr_text) < 10:
        penalty = 2
        print(f"⚡ [PACING] Dead time detected - low flow ({flow:.2f}), minimal text")
    elif flow < 0.4 and len(asr_text + ocr_text) < 30:
        penalty = 1
        print(f"⚡ [PACING] Some boring moments detected")
    
    return penalty


def estimate_retention_curve(frames: List[str], features: Dict[str, Any]) -> float:
    """
    Retention curve tahmini
    """
    # Hareket, metin ve ses verilerini birleştir
    visual = features.get("visual", {})
    flow = visual.get("optical_flow_mean", 0)
    
    text_length = len(features.get("textual", {}).get("ocr_text", "") + 
                     features.get("textual", {}).get("asr_text", ""))
    
    # Basit retention tahmini
    base_retention = 0.5
    
    # Hareket bonusu
    if flow > 1.0:
        base_retention += 0.2
    elif flow > 0.5:
        base_retention += 0.1
    
    # Metin bonusu
    if text_length > 50:
        base_retention += 0.2
    elif text_length > 20:
        base_retention += 0.1
    
    # BPM bonusu
    bpm = features.get("audio", {}).get("tempo_bpm", 0)
    if bpm > 100:
        base_retention += 0.1
    
    return min(base_retention, 1.0)
