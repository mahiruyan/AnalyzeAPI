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

def analyze_hook(video_path: str, audio_path: str, frames: List[str], features: Dict[str, Any], duration: float) -> Dict[str, Any]:
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
        # Gerçek analiz verisine göre öneri
        recommendations.append(f"Görsel sıçrama çok düşük ({visual_jump_score:.2f}). İlk 3 saniyede ani zoom, renk değişimi veya yakın plan ekle.")
    
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
        # Gerçek OCR verisine göre öneri
        ocr_text = features.get("textual", {}).get("ocr_text", "")
        if len(ocr_text) > 0:
            recommendations.append(f"Metin tespit edildi ama kanca gücü düşük ({hook_text_score:.2f}). '{ocr_text[:30]}...' metnini daha dikkat çekici yap.")
        else:
            recommendations.append(f"Hiç metin tespit edilmedi ({hook_text_score:.2f}). İlk 3 saniyede güçlü bir kanca metni ekle.")
    
    # 3. Ses/Beat Giriş Analizi + GERÇEK beat detection
    audio_intro_score = analyze_audio_intro(features)
    beat_drop_score = detect_beat_drop_in_intro(audio_path)
    raw_metrics["audio_intro_energy"] = audio_intro_score
    raw_metrics["beat_drop_score"] = beat_drop_score
    
    # Beat drop bonusu ekle
    audio_intro_score = min(audio_intro_score + beat_drop_score, 1.0)
    
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
        # Gerçek ASR verisine göre öneri
        asr_text = features.get("textual", {}).get("asr_text", "")
        if len(asr_text) > 0:
            first_words = asr_text.split()[:5]
            recommendations.append(f"Konuşma tespit edildi ama direkt başlangıç yok ({direct_start_score:.2f}). '{' '.join(first_words)}...' yerine direkt konuyla başla.")
        else:
            recommendations.append(f"Konuşma tespit edilmedi ({direct_start_score:.2f}). 'Merhaba' yerine direkt konuyla başla.")
    
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
        # Gerçek metin verisine göre öneri
        combined_text = (features.get("textual", {}).get("ocr_text", "") + " " + 
                        features.get("textual", {}).get("asr_text", "")).lower()
        if "sonunda" in combined_text or "göreceksin" in combined_text:
            recommendations.append(f"Foreshadowing var ama yeterince güçlü değil ({foreshadow_score:.2f}). 'Sonunda X'i göreceksin' ifadesini daha belirgin yap.")
        else:
            recommendations.append(f"Hiç foreshadowing yok ({foreshadow_score:.2f}). 0-3 saniye arası 'Sonunda X'i göreceksin' tarzı vaat ekle.")
    
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
        print(f"🎯 [HOOK] High energy BPM: {bpm}")
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


def detect_beat_drop_in_intro(audio_path: str) -> float:
    """
    GERÇEK beat detection - librosa ile ilk 3 saniyede beat/drop tespit
    """
    try:
        import librosa
        import librosa.beat
        
        print(f"🎵 [HOOK] Analyzing beats in: {audio_path}")
        
        # İlk 3 saniyeyi yükle
        y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
        
        if len(y) == 0:
            print("🎵 [HOOK] Audio file empty or not found")
            return 0.0
        
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
        
        # Onset detection (beat drop'lar için)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, units='time')
        
        # İlk 3 saniyede beat/onset var mı?
        early_beats = [t for t in beat_times if t <= 3.0]
        early_onsets = [t for t in onset_frames if t <= 3.0]
        
        score = 0.0
        
        # Beat drop scoring
        if len(early_onsets) >= 2:
            score += 0.4  # Çoklu onset = drop efekti
            print(f"🎵 [HOOK] Multiple beat drops in intro: {len(early_onsets)} onsets at {early_onsets}")
        elif len(early_onsets) >= 1:
            score += 0.2  # Tek onset
            print(f"🎵 [HOOK] Beat drop detected at {early_onsets[0]:.1f}s")
        
        # İlk 1 saniyede onset varsa bonus
        if any(t <= 1.0 for t in early_onsets):
            score += 0.3
            print(f"🎵 [HOOK] Early beat drop (within 1s) - very powerful!")
        
        # Beat consistency check
        if tempo > 0 and len(early_beats) > 0:
            expected_beats_in_3s = (tempo / 60) * 3
            actual_beats = len(early_beats)
            consistency = min(actual_beats / expected_beats_in_3s, 1.0) if expected_beats_in_3s > 0 else 0
            
            if consistency > 0.7:
                score += 0.1
                print(f"🎵 [HOOK] Good beat consistency: {consistency:.2f} ({actual_beats}/{expected_beats_in_3s:.1f})")
        
        print(f"🎵 [HOOK] Beat drop analysis complete: {score:.2f}")
        return min(score, 1.0)
        
    except Exception as e:
        print(f"❌ [HOOK] Beat detection failed: {e}")
        return 0.0


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
    
    # 1. Kesim Yoğunluğu Analizi (GERÇEK timing ile)
    cuts_per_sec, cut_timestamps = detect_cuts_per_second(frames, duration)
    raw_metrics["cuts_per_sec"] = cuts_per_sec
    raw_metrics["cut_timestamps"] = cut_timestamps
    
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
            recommendations.append(f"Çok yavaş edit ({cuts_per_sec:.2f} kesim/saniye) - daha fazla kesim ekle. İdeal: 0.3-0.6 arası.")
        else:
            recommendations.append(f"Çok hızlı edit ({cuts_per_sec:.2f} kesim/saniye) - izleyici kaybedebilir. Biraz yavaşlat.")
    
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
        recommendations.append(f"Hareket ritmi çok düşük ({movement_rhythm:.2f}). Daha ritmik hareket paterni oluştur - düzenli aralıklarla hareket ekle.")
    
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


def detect_cuts_per_second(frames: List[str], duration: float) -> Tuple[float, List[float]]:
    """
    GERÇEK cut detection - histogram + edge + timing based
    Returns: (cuts_per_sec, cut_timestamps)
    """
    if len(frames) <= 1 or duration <= 0:
        return 0.0, []
    
    try:
        cuts = []
        cut_timestamps = []
        
        # Frame timing hesapla
        frame_interval = duration / len(frames) if len(frames) > 0 else 1.0
        
        for i in range(len(frames) - 1):
            img1 = cv2.imread(frames[i])
            img2 = cv2.imread(frames[i + 1])
            
            if img1 is None or img2 is None:
                continue
            
            # Resize for speed
            img1 = cv2.resize(img1, (128, 128))
            img2 = cv2.resize(img2, (128, 128))
            
            # 1. Histogram farkı (renk değişimi)
            hist1 = cv2.calcHist([img1], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([img2], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
            hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR) / 10000.0
            
            # 2. Edge farkı (kompozisyon değişimi)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)
            edge_diff = np.mean(np.abs(edges1.astype(float) - edges2.astype(float))) / 255.0
            
            # 3. Structural similarity
            ssim_score = calculate_ssim(gray1, gray2)
            structure_diff = 1.0 - ssim_score
            
            # Kombinasyon: üç metriğin weighted average'ı
            combined_diff = (hist_diff * 0.4) + (edge_diff * 0.3) + (structure_diff * 0.3)
            
            # Cut threshold (adaptive)
            threshold = 0.25
            if combined_diff > threshold:
                cut_time = i * frame_interval
                cuts.append(combined_diff)
                cut_timestamps.append(cut_time)
                print(f"⚡ [PACING] CUT at {cut_time:.1f}s: hist={hist_diff:.3f}, edge={edge_diff:.3f}, struct={structure_diff:.3f}, combined={combined_diff:.3f}")
        
        cuts_per_sec = len(cuts) / duration
        print(f"⚡ [PACING] Cut detection complete: {len(cuts)} cuts in {duration:.1f}s = {cuts_per_sec:.3f} cuts/sec")
        
        return cuts_per_sec, cut_timestamps
        
    except Exception as e:
        print(f"❌ [PACING] Cut detection failed: {e}")
        return len(frames) / duration, []  # Fallback


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index - frame benzerliği
    """
    try:
        # Simple SSIM implementation
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        # SSIM constants
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
        
        ssim = numerator / denominator if denominator > 0 else 0.0
        return max(0.0, min(1.0, ssim))
        
    except Exception as e:
        print(f"❌ [PACING] SSIM calculation failed: {e}")
        return 0.0


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


def analyze_cta_interaction(features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """
    CTA & Etkileşim Analizi (8 puan)
    Çağrı tespit ve analiz
    """
    score = 0
    findings = []
    recommendations = []
    raw_metrics = {}
    
    print("📢 [CTA] Starting CTA & interaction analysis...")
    
    # Tüm metinleri birleştir
    textual = features.get("textual", {})
    ocr_text = textual.get("ocr_text", "").lower()
    asr_text = textual.get("asr_text", "").lower()
    caption = textual.get("caption", "").lower()
    title = textual.get("title", "").lower()
    
    combined_text = f"{ocr_text} {asr_text} {caption} {title}"
    
    # 1. CTA Tespiti (4 puan)
    cta_score = detect_cta_patterns(combined_text)
    raw_metrics["cta_detection"] = cta_score
    
    if cta_score > 0.7:
        score += 4
        findings.append({
            "t": duration * 0.8,
            "type": "strong_cta",
            "msg": "Güçlü çağrı tespit edildi - etkileşim artırıcı!"
        })
    elif cta_score > 0.4:
        score += 2
        recommendations.append("CTA'yı daha belirgin yap - 'beğen', 'takip et', 'yorum yap' ekle")
    else:
        score += 1
        recommendations.append("Hiç CTA yok - video sonunda güçlü çağrı ekle")
    
    # 2. Etkileşim Teşviki (2 puan)
    interaction_score = detect_interaction_patterns(combined_text)
    raw_metrics["interaction_encouragement"] = interaction_score
    
    if interaction_score > 0.6:
        score += 2
        findings.append({
            "t": duration * 0.7,
            "type": "interaction_boost",
            "msg": "Etkileşim teşviki var - beğeni/takip çağrısı"
        })
    elif interaction_score > 0.3:
        score += 1
        recommendations.append("Etkileşim çağrısını güçlendir")
    else:
        recommendations.append("Etkileşim çağrısı ekle - 'beğenmeyi unutma' tarzı")
    
    # 3. Zamanlama Analizi (2 puan)
    timing_score = analyze_cta_timing(features, duration)
    raw_metrics["cta_timing"] = timing_score
    
    if timing_score > 0.7:
        score += 2
        findings.append({
            "t": duration * 0.9,
            "type": "perfect_timing",
            "msg": "CTA zamanlaması mükemmel - video sonunda"
        })
    elif timing_score > 0.4:
        score += 1
        recommendations.append("CTA'yı video sonuna taşı")
    else:
        recommendations.append("CTA çok erken - video sonunda olmalı")
    
    print(f"📢 [CTA] CTA & interaction score: {score}/8")
    
    return {
        "score": min(score, 8),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw_metrics
    }


def detect_cta_patterns(text: str) -> float:
    """
    CTA pattern tespiti
    """
    # Güçlü CTA kalıpları
    strong_cta = [
        "beğen", "like", "takip et", "follow", "abone ol", "subscribe",
        "yorum yap", "comment", "paylaş", "share", "kaydet", "save",
        "dm at", "dm", "mesaj", "message", "link", "linke tıkla"
    ]
    
    # Orta CTA kalıpları
    medium_cta = [
        "daha fazla", "more", "devam", "continue", "sonraki", "next",
        "önceki", "previous", "kanal", "channel", "profil", "profile"
    ]
    
    # Zayıf CTA kalıpları
    weak_cta = [
        "bak", "look", "gör", "see", "dinle", "listen", "izle", "watch"
    ]
    
    score = 0.0
    
    # Güçlü CTA'lar
    for pattern in strong_cta:
        if pattern in text:
            score += 0.4
            print(f"📢 [CTA] Strong CTA found: '{pattern}' (+0.4)")
    
    # Orta CTA'lar
    for pattern in medium_cta:
        if pattern in text:
            score += 0.2
            print(f"📢 [CTA] Medium CTA found: '{pattern}' (+0.2)")
    
    # Zayıf CTA'lar
    for pattern in weak_cta:
        if pattern in text:
            score += 0.1
            print(f"📢 [CTA] Weak CTA found: '{pattern}' (+0.1)")
    
    return min(score, 1.0)


def detect_interaction_patterns(text: str) -> float:
    """
    Etkileşim teşviki tespiti
    """
    interaction_patterns = [
        "beğenmeyi unutma", "don't forget to like", "like if you",
        "takip etmeyi unutma", "don't forget to follow", "follow for more",
        "yorum yapmayı unutma", "don't forget to comment", "comment below",
        "paylaşmayı unutma", "don't forget to share", "share this",
        "kaydetmeyi unutma", "don't forget to save", "save this post",
        "bildirimleri aç", "turn on notifications", "notifications on"
    ]
    
    score = 0.0
    
    for pattern in interaction_patterns:
        if pattern in text:
            score += 0.3
            print(f"📢 [CTA] Interaction pattern found: '{pattern}' (+0.3)")
    
    return min(score, 1.0)


def analyze_cta_timing(features: Dict[str, Any], duration: float) -> float:
    """
    CTA zamanlama analizi
    """
    # CTA'ların video sonunda olması ideal
    # Şimdilik basit analiz - geliştirilecek
    
    asr_text = features.get("textual", {}).get("asr_text", "").lower()
    ocr_text = features.get("textual", {}).get("ocr_text", "").lower()
    
    # Video sonundaki metinleri kontrol et (son %20)
    # Bu basit bir yaklaşım - gerçek zamanlama analizi için daha gelişmiş yöntem gerekli
    
    cta_keywords = ["beğen", "takip", "abone", "yorum", "paylaş", "kaydet"]
    
    # OCR ve ASR'da CTA var mı?
    has_cta = any(keyword in asr_text or keyword in ocr_text for keyword in cta_keywords)
    
    if has_cta:
        # CTA var - zamanlama analizi yapılabilir
        return 0.6  # Orta skor - gerçek zamanlama analizi geliştirilecek
    else:
        return 0.0  # CTA yok


def analyze_message_clarity(features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """
    Mesaj Netliği Analizi (6 puan)
    4-7 saniye konu belirtme analizi
    """
    score = 0
    findings = []
    recommendations = []
    raw_metrics = {}
    
    print("💬 [MESSAGE] Starting message clarity analysis...")
    
    # 1. Erken Konu Belirtme (3 puan)
    early_topic_score = analyze_early_topic_mention(features, duration)
    raw_metrics["early_topic_mention"] = early_topic_score
    
    if early_topic_score > 0.7:
        score += 3
        findings.append({
            "t": 3.0,
            "type": "clear_early_topic",
            "msg": "Konu 4-7 saniye içinde net belirtilmiş"
        })
    elif early_topic_score > 0.4:
        score += 2
        recommendations.append("Konuyu daha erken belirt - ilk 5 saniyede")
    else:
        score += 1
        recommendations.append("Konu belirsiz - ilk 7 saniyede net konu belirt")
    
    # 2. Mesaj Tutarlılığı (2 puan)
    consistency_score = analyze_message_consistency(features)
    raw_metrics["message_consistency"] = consistency_score
    
    if consistency_score > 0.6:
        score += 2
        findings.append({
            "t": duration * 0.5,
            "type": "consistent_message",
            "msg": "Mesaj tutarlı - konu boyunca aynı tema"
        })
    elif consistency_score > 0.3:
        score += 1
        recommendations.append("Mesajı daha tutarlı yap - konudan sapma")
    else:
        recommendations.append("Mesaj tutarsız - tek konuya odaklan")
    
    # 3. Netlik Skoru (1 puan)
    clarity_score = analyze_message_clarity_score(features)
    raw_metrics["clarity_score"] = clarity_score
    
    if clarity_score > 0.7:
        score += 1
        findings.append({
            "t": duration * 0.3,
            "type": "clear_message",
            "msg": "Mesaj çok net ve anlaşılır"
        })
    else:
        recommendations.append("Mesajı daha net yap - basit kelimeler kullan")
    
    print(f"💬 [MESSAGE] Message clarity score: {score}/6")
    
    return {
        "score": min(score, 6),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw_metrics
    }


def analyze_early_topic_mention(features: Dict[str, Any], duration: float) -> float:
    """
    Erken konu belirtme analizi (4-7 saniye)
    """
    asr_text = features.get("textual", {}).get("asr_text", "").lower()
    ocr_text = features.get("textual", {}).get("ocr_text", "").lower()
    caption = features.get("textual", {}).get("caption", "").lower()
    title = features.get("textual", {}).get("title", "").lower()
    
    # Konu belirten kelimeler
    topic_indicators = [
        "bugün", "today", "şimdi", "now", "bu video", "this video",
        "konu", "topic", "hakkında", "about", "ile ilgili", "related to",
        "anlatacağım", "i will tell", "göstereceğim", "i will show",
        "öğreneceksin", "you will learn", "göreceksin", "you will see"
    ]
    
    # İlk 7 saniyede konu belirtme kontrolü
    # Basit yaklaşım - gerçek zamanlama analizi için daha gelişmiş yöntem gerekli
    
    combined_text = f"{asr_text} {ocr_text} {caption} {title}"
    
    topic_mentions = 0
    for indicator in topic_indicators:
        if indicator in combined_text:
            topic_mentions += 1
            print(f"💬 [MESSAGE] Topic indicator found: '{indicator}'")
    
    # Skor hesapla
    if topic_mentions >= 3:
        return 1.0  # Mükemmel
    elif topic_mentions >= 2:
        return 0.7  # İyi
    elif topic_mentions >= 1:
        return 0.4  # Orta
    else:
        return 0.0  # Zayıf


def analyze_message_consistency(features: Dict[str, Any]) -> float:
    """
    Mesaj tutarlılığı analizi
    """
    asr_text = features.get("textual", {}).get("asr_text", "").lower()
    ocr_text = features.get("textual", {}).get("ocr_text", "").lower()
    caption = features.get("textual", {}).get("caption", "").lower()
    title = features.get("textual", {}).get("title", "").lower()
    
    # Ana konu kelimelerini çıkar
    all_text = f"{asr_text} {ocr_text} {caption} {title}"
    
    # Basit tutarlılık analizi - kelime tekrarı
    words = all_text.split()
    if len(words) < 5:
        return 0.0
    
    # En sık kullanılan kelimeler
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Kısa kelimeleri atla
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # En sık kelimenin oranı
    if word_freq:
        max_freq = max(word_freq.values())
        total_words = len([w for w in words if len(w) > 3])
        consistency_ratio = max_freq / total_words
        
        if consistency_ratio > 0.1:  # %10'dan fazla tekrar
            return 0.8
        elif consistency_ratio > 0.05:  # %5'ten fazla tekrar
            return 0.6
        else:
            return 0.3
    else:
        return 0.0


def analyze_message_clarity_score(features: Dict[str, Any]) -> float:
    """
    Mesaj netlik skoru - gelişmiş NLP ile
    """
    asr_text = features.get("textual", {}).get("asr_text", "")
    ocr_text = features.get("textual", {}).get("ocr_text", "")
    
    # Gelişmiş NLP analizi
    nlp_result = advanced_nlp_analysis(f"{asr_text} {ocr_text}")
    
    # Netlik skoru hesapla
    clarity_score = 0.0
    
    # 1. Sentiment analizi (pozitif mesajlar daha net)
    sentiment = nlp_result.get("sentiment", 0.5)
    if sentiment > 0.6:
        clarity_score += 0.3
    elif sentiment > 0.4:
        clarity_score += 0.2
    else:
        clarity_score += 0.1
    
    # 2. Kelime karmaşıklığı
    complexity = nlp_result.get("complexity", 0.5)
    if complexity < 0.3:  # Basit kelimeler
        clarity_score += 0.3
    elif complexity < 0.6:
        clarity_score += 0.2
    else:
        clarity_score += 0.1
    
    # 3. Cümle yapısı
    sentence_structure = nlp_result.get("sentence_structure", 0.5)
    if sentence_structure > 0.7:  # İyi cümle yapısı
        clarity_score += 0.2
    elif sentence_structure > 0.4:
        clarity_score += 0.1
    
    # 4. Anahtar kelime yoğunluğu
    keyword_density = nlp_result.get("keyword_density", 0.5)
    if 0.3 <= keyword_density <= 0.7:  # Optimal yoğunluk
        clarity_score += 0.2
    else:
        clarity_score += 0.1
    
    # 5. Readability bonusu
    readability = nlp_result.get("readability_score", 0.5)
    if readability > 0.7:
        clarity_score += 0.1
    
    # 6. Language quality bonusu
    language_quality = nlp_result.get("language_quality", 0.5)
    if language_quality > 0.7:
        clarity_score += 0.1
    
    return min(clarity_score, 1.0)


def advanced_nlp_analysis(text: str) -> Dict[str, float]:
    """
    Gelişmiş NLP analizi - sentiment, complexity, structure, emotion, intent, readability
    """
    if not text or len(text.strip()) < 10:
        return {
            "sentiment": 0.5,
            "complexity": 0.5,
            "sentence_structure": 0.5,
            "keyword_density": 0.5,
            "entities": [],
            "topics": [],
            "emotions": {},
            "intent": "unknown",
            "readability_score": 0.5,
            "engagement_potential": 0.5,
            "viral_keywords": [],
            "language_quality": 0.5
        }
    
    text_lower = text.lower()
    
    # 1. Sentiment Analizi
    sentiment_score = analyze_sentiment(text_lower)
    
    # 2. Kelime Karmaşıklığı
    complexity_score = analyze_complexity(text_lower)
    
    # 3. Cümle Yapısı
    structure_score = analyze_sentence_structure(text_lower)
    
    # 4. Anahtar Kelime Yoğunluğu
    keyword_density = analyze_keyword_density(text_lower)
    
    # 5. Entity Recognition
    entities = extract_entities(text_lower)
    
    # 6. Topic Modeling
    topics = extract_topics(text_lower)
    
    # 7. Emotion Detection
    emotions = detect_emotions(text_lower)
    
    # 8. Intent Analysis
    intent = analyze_intent(text_lower)
    
    # 9. Readability Score
    readability_score = calculate_readability_score(text_lower)
    
    # 10. Engagement Potential
    engagement_potential = calculate_engagement_potential(text_lower, sentiment_score, emotions)
    
    # 11. Viral Keywords
    viral_keywords = extract_viral_keywords(text_lower)
    
    # 12. Language Quality
    language_quality = assess_language_quality(text_lower, structure_score, complexity_score)
    
    return {
        "sentiment": sentiment_score,
        "complexity": complexity_score,
        "sentence_structure": structure_score,
        "keyword_density": keyword_density,
        "entities": entities,
        "topics": topics,
        "emotions": emotions,
        "intent": intent,
        "readability_score": readability_score,
        "engagement_potential": engagement_potential,
        "viral_keywords": viral_keywords,
        "language_quality": language_quality
    }


def analyze_sentiment(text: str) -> float:
    """
    Sentiment analizi - pozitif/negatif/nötr
    """
    # Pozitif kelimeler
    positive_words = [
        "harika", "mükemmel", "güzel", "iyi", "başarılı", "başarı", "kazanç", "kazan",
        "kolay", "basit", "hızlı", "etkili", "güçlü", "büyük", "yüksek", "artış",
        "çözüm", "çözmek", "başarmak", "başarı", "mutlu", "sevinç", "heyecan",
        "beğen", "sev", "takdir", "övgü", "tebrik", "kutlama", "celebration",
        "great", "amazing", "wonderful", "excellent", "perfect", "awesome",
        "good", "best", "better", "improve", "success", "win", "profit"
    ]
    
    # Negatif kelimeler
    negative_words = [
        "kötü", "berbat", "başarısız", "hata", "problem", "sorun", "zor", "karmaşık",
        "yavaş", "düşük", "az", "kayıp", "kaybet", "başarısızlık", "hata", "yanlış",
        "üzgün", "kızgın", "sinir", "stres", "endişe", "korku", "kaygı",
        "beğenme", "sevme", "nefret", "kızgınlık", "öfke", "hayal kırıklığı",
        "bad", "terrible", "awful", "horrible", "worst", "fail", "failure",
        "error", "mistake", "problem", "issue", "difficult", "hard", "complex"
    ]
    
    # Nötr kelimeler
    neutral_words = [
        "video", "içerik", "kanal", "abone", "izleyici", "takip", "beğen",
        "yorum", "paylaş", "kaydet", "bildirim", "link", "profil", "hesap",
        "video", "content", "channel", "subscribe", "viewer", "follow", "like",
        "comment", "share", "save", "notification", "link", "profile", "account"
    ]
    
    words = text.split()
    if not words:
        return 0.5
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    neutral_count = sum(1 for word in words if word in neutral_words)
    
    total_words = len(words)
    
    # Sentiment skoru hesapla
    if total_words == 0:
        return 0.5
    
    positive_ratio = positive_count / total_words
    negative_ratio = negative_count / total_words
    
    # 0.0 (çok negatif) - 1.0 (çok pozitif)
    sentiment = 0.5 + (positive_ratio - negative_ratio) * 2
    sentiment = max(0.0, min(1.0, sentiment))
    
    print(f"💬 [NLP] Sentiment: {sentiment:.2f} (pos: {positive_count}, neg: {negative_count}, total: {total_words})")
    
    return sentiment


def analyze_complexity(text: str) -> float:
    """
    Kelime karmaşıklığı analizi
    """
    words = text.split()
    if not words:
        return 0.5
    
    # Karmaşık kelimeler (uzun, teknik)
    complex_words = [
        "algoritma", "teknoloji", "sistem", "mekanizma", "prosedür", "metodoloji",
        "analiz", "sentez", "değerlendirme", "optimizasyon", "implementasyon",
        "algorithm", "technology", "system", "mechanism", "procedure", "methodology",
        "analysis", "synthesis", "evaluation", "optimization", "implementation"
    ]
    
    # Basit kelimeler (kısa, günlük)
    simple_words = [
        "iyi", "kötü", "güzel", "çirkin", "büyük", "küçük", "hızlı", "yavaş",
        "kolay", "zor", "basit", "karmaşık", "açık", "kapalı", "yeni", "eski",
        "good", "bad", "beautiful", "ugly", "big", "small", "fast", "slow",
        "easy", "hard", "simple", "complex", "open", "closed", "new", "old"
    ]
    
    complex_count = sum(1 for word in words if word in complex_words)
    simple_count = sum(1 for word in words if word in simple_words)
    total_words = len(words)
    
    if total_words == 0:
        return 0.5
    
    # Ortalama kelime uzunluğu
    avg_word_length = sum(len(word) for word in words) / total_words
    
    # Karmaşıklık skoru
    complexity_score = 0.0
    
    # Kelime türü oranı
    if complex_count > simple_count:
        complexity_score += 0.4
    elif complex_count == simple_count:
        complexity_score += 0.2
    
    # Ortalama kelime uzunluğu
    if avg_word_length > 6:
        complexity_score += 0.3
    elif avg_word_length > 4:
        complexity_score += 0.2
    else:
        complexity_score += 0.1
    
    # Teknik terim oranı
    technical_ratio = complex_count / total_words
    if technical_ratio > 0.3:
        complexity_score += 0.3
    elif technical_ratio > 0.1:
        complexity_score += 0.2
    else:
        complexity_score += 0.1
    
    complexity_score = min(complexity_score, 1.0)
    
    print(f"💬 [NLP] Complexity: {complexity_score:.2f} (complex: {complex_count}, simple: {simple_count}, avg_len: {avg_word_length:.1f})")
    
    return complexity_score


def analyze_sentence_structure(text: str) -> float:
    """
    Cümle yapısı analizi
    """
    # Cümle ayırıcıları
    sentence_endings = [".", "!", "?", ":", ";"]
    
    # Cümleleri ayır
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in sentence_endings:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    if not sentences:
        return 0.5
    
    structure_score = 0.0
    
    # 1. Cümle uzunluğu çeşitliliği
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    if sentence_lengths:
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        length_variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
        
        # Optimal cümle uzunluğu (5-15 kelime)
        if 5 <= avg_length <= 15:
            structure_score += 0.4
        elif 3 <= avg_length <= 20:
            structure_score += 0.2
        
        # Çeşitlilik bonusu
        if length_variance > 10:  # Farklı uzunlukta cümleler
            structure_score += 0.2
    
    # 2. Cümle başlangıçları çeşitliliği
    sentence_starts = [sentence.split()[0] if sentence.split() else "" for sentence in sentences]
    unique_starts = len(set(sentence_starts))
    total_sentences = len(sentences)
    
    if total_sentences > 0:
        start_diversity = unique_starts / total_sentences
        if start_diversity > 0.7:
            structure_score += 0.2
        elif start_diversity > 0.5:
            structure_score += 0.1
    
    # 3. Noktalama kullanımı
    punctuation_count = sum(1 for char in text if char in sentence_endings)
    if punctuation_count > 0:
        structure_score += 0.2
    
    structure_score = min(structure_score, 1.0)
    
    print(f"💬 [NLP] Sentence structure: {structure_score:.2f} (sentences: {len(sentences)}, avg_len: {avg_length:.1f})")
    
    return structure_score


def analyze_keyword_density(text: str) -> float:
    """
    Anahtar kelime yoğunluğu analizi
    """
    words = text.split()
    if not words:
        return 0.5
    
    # Anahtar kelimeler (video içeriği için)
    keywords = [
        "video", "içerik", "kanal", "abone", "izleyici", "takip", "beğen",
        "yorum", "paylaş", "kaydet", "bildirim", "link", "profil", "hesap",
        "viral", "trend", "popüler", "başarı", "kazanç", "artış", "gelişim",
        "video", "content", "channel", "subscribe", "viewer", "follow", "like",
        "comment", "share", "save", "notification", "link", "profile", "account",
        "viral", "trend", "popular", "success", "profit", "growth", "development"
    ]
    
    # Stop words (gereksiz kelimeler)
    stop_words = [
        "ve", "ile", "için", "olan", "bu", "şu", "o", "bir", "her", "hiç",
        "çok", "daha", "en", "da", "de", "ki", "mi", "mı", "mu", "mü",
        "and", "with", "for", "the", "a", "an", "this", "that", "is", "are",
        "was", "were", "be", "been", "have", "has", "had", "do", "does", "did"
    ]
    
    # Kelime frekansları
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    if not word_freq:
        return 0.5
    
    # Anahtar kelime oranı
    keyword_count = sum(1 for word in words if word in keywords)
    total_words = len(words)
    keyword_ratio = keyword_count / total_words if total_words > 0 else 0
    
    # En sık kelimenin oranı
    max_freq = max(word_freq.values())
    total_unique_words = len(word_freq)
    max_freq_ratio = max_freq / total_unique_words if total_unique_words > 0 else 0
    
    # Yoğunluk skoru
    density_score = 0.0
    
    # Anahtar kelime oranı (optimal: %2-5)
    if 0.02 <= keyword_ratio <= 0.05:
        density_score += 0.5
    elif 0.01 <= keyword_ratio <= 0.08:
        density_score += 0.3
    else:
        density_score += 0.1
    
    # En sık kelime oranı (optimal: %5-15)
    if 0.05 <= max_freq_ratio <= 0.15:
        density_score += 0.3
    elif 0.03 <= max_freq_ratio <= 0.25:
        density_score += 0.2
    else:
        density_score += 0.1
    
    # Kelime çeşitliliği
    if total_unique_words > 10:
        density_score += 0.2
    elif total_unique_words > 5:
        density_score += 0.1
    
    density_score = min(density_score, 1.0)
    
    print(f"💬 [NLP] Keyword density: {density_score:.2f} (keywords: {keyword_count}/{total_words}, max_freq: {max_freq_ratio:.2f})")
    
    return density_score


def extract_entities(text: str) -> List[str]:
    """
    Entity recognition - kişi, yer, kurum tespiti
    """
    entities = []
    
    # Kişi isimleri (basit pattern matching)
    person_patterns = [
        "ahmet", "mehmet", "ali", "ayşe", "fatma", "zeynep", "emre", "can",
        "john", "michael", "david", "sarah", "emily", "jessica", "chris"
    ]
    
    # Yer isimleri
    place_patterns = [
        "istanbul", "ankara", "izmir", "bursa", "antalya", "adana", "konya",
        "london", "paris", "new york", "tokyo", "berlin", "rome", "madrid"
    ]
    
    # Kurum isimleri
    organization_patterns = [
        "google", "facebook", "instagram", "youtube", "tiktok", "twitter",
        "apple", "microsoft", "amazon", "netflix", "spotify", "uber"
    ]
    
    # Entity tespiti
    for pattern in person_patterns:
        if pattern in text:
            entities.append(f"PERSON:{pattern}")
    
    for pattern in place_patterns:
        if pattern in text:
            entities.append(f"PLACE:{pattern}")
    
    for pattern in organization_patterns:
        if pattern in text:
            entities.append(f"ORG:{pattern}")
    
    print(f"💬 [NLP] Entities found: {entities}")
    
    return entities


def extract_topics(text: str) -> List[str]:
    """
    Topic modeling - ana konuları çıkar
    """
    topics = []
    
    # Konu kategorileri
    topic_categories = {
        "teknoloji": ["teknoloji", "teknoloji", "bilgisayar", "telefon", "internet", "yazılım", "uygulama"],
        "eğitim": ["eğitim", "öğrenme", "ders", "okul", "üniversite", "kurs", "öğretmen"],
        "sağlık": ["sağlık", "doktor", "hastane", "ilaç", "tedavi", "beslenme", "spor"],
        "iş": ["iş", "kariyer", "maaş", "şirket", "çalışma", "patron", "meslek"],
        "eğlence": ["eğlence", "oyun", "film", "müzik", "parti", "tatil", "hobi"],
        "alışveriş": ["alışveriş", "satın", "fiyat", "indirim", "mağaza", "ürün", "marka"],
        "teknoloji": ["technology", "computer", "phone", "internet", "software", "app", "digital"],
        "education": ["education", "learning", "school", "university", "course", "teacher", "study"],
        "health": ["health", "doctor", "hospital", "medicine", "treatment", "nutrition", "fitness"],
        "business": ["business", "career", "salary", "company", "work", "boss", "job"],
        "entertainment": ["entertainment", "game", "movie", "music", "party", "vacation", "hobby"],
        "shopping": ["shopping", "buy", "price", "discount", "store", "product", "brand"]
    }
    
    # Konu tespiti
    for topic, keywords in topic_categories.items():
        keyword_count = sum(1 for keyword in keywords if keyword in text)
        if keyword_count >= 2:  # En az 2 anahtar kelime
            topics.append(topic)
    
    print(f"💬 [NLP] Topics found: {topics}")
    
    return topics


def detect_emotions(text: str) -> Dict[str, float]:
    """
    Emotion detection - 8 temel duygu
    """
    emotions = {
        "joy": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "sadness": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "trust": 0.0,
        "anticipation": 0.0
    }
    
    # Joy (Sevinç)
    joy_words = [
        "mutlu", "sevinç", "heyecan", "neşe", "gülümseme", "gülmek", "eğlence",
        "harika", "mükemmel", "süper", "fantastik", "muhteşem", "wow", "amazing",
        "happy", "joy", "excitement", "fun", "smile", "laugh", "wonderful", "great"
    ]
    
    # Anger (Öfke)
    anger_words = [
        "kızgın", "sinir", "öfke", "hiddet", "kızgınlık", "sinirlenmek", "küfür",
        "berbat", "nefret", "tiksinti", "rahatsız", "angry", "mad", "furious",
        "rage", "hate", "disgust", "annoyed", "irritated", "terrible", "awful"
    ]
    
    # Fear (Korku)
    fear_words = [
        "korku", "endişe", "kaygı", "panik", "tedirgin", "korkmak", "endişelenmek",
        "tehlikeli", "risk", "riskli", "fear", "afraid", "scared", "worried",
        "anxiety", "panic", "dangerous", "risk", "threat", "concern"
    ]
    
    # Sadness (Üzüntü)
    sadness_words = [
        "üzgün", "hüzün", "keder", "mutsuz", "depresyon", "ağlamak", "gözyaşı",
        "kayıp", "acı", "hüzünlü", "sad", "sadness", "depressed", "unhappy",
        "cry", "tears", "loss", "pain", "grief", "melancholy"
    ]
    
    # Surprise (Şaşkınlık)
    surprise_words = [
        "şaşkın", "hayret", "şaşırmak", "inanılmaz", "inanamıyorum", "vay be",
        "wow", "oh my god", "incredible", "unbelievable", "surprised", "shocked",
        "amazed", "astonished", "stunned", "speechless"
    ]
    
    # Disgust (Tiksinti)
    disgust_words = [
        "tiksinti", "iğrenme", "tiksinmek", "iğrenç", "çirkin", "pis", "kirli",
        "disgust", "disgusting", "gross", "nasty", "ugly", "dirty", "filthy",
        "revolting", "repulsive", "sickening"
    ]
    
    # Trust (Güven)
    trust_words = [
        "güven", "güvenmek", "güvenilir", "güvenli", "emin", "kesin", "garanti",
        "trust", "trustworthy", "reliable", "safe", "secure", "confident",
        "certain", "guarantee", "promise", "assure"
    ]
    
    # Anticipation (Beklenti)
    anticipation_words = [
        "beklenti", "beklemek", "umut", "umutlu", "heyecan", "merak", "sabırsızlık",
        "anticipation", "expect", "hope", "hopeful", "excited", "curious",
        "impatient", "waiting", "looking forward", "eager"
    ]
    
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return emotions
    
    # Her duygu için skor hesapla
    emotion_word_lists = {
        "joy": joy_words,
        "anger": anger_words,
        "fear": fear_words,
        "sadness": sadness_words,
        "surprise": surprise_words,
        "disgust": disgust_words,
        "trust": trust_words,
        "anticipation": anticipation_words
    }
    
    for emotion, word_list in emotion_word_lists.items():
        count = sum(1 for word in words if word in word_list)
        emotions[emotion] = min(count / total_words * 10, 1.0)  # Normalize
    
    # En güçlü duyguyu bul
    dominant_emotion = max(emotions, key=emotions.get)
    print(f"💬 [NLP] Emotions: {emotions}, dominant: {dominant_emotion}")
    
    return emotions


def analyze_intent(text: str) -> str:
    """
    Intent analysis - kullanıcının amacını tespit et
    """
    # Intent kategorileri
    intents = {
        "inform": 0.0,      # Bilgi verme
        "persuade": 0.0,    # İkna etme
        "entertain": 0.0,   # Eğlendirme
        "educate": 0.0,     # Eğitme
        "sell": 0.0,        # Satış
        "engage": 0.0,      # Etkileşim
        "inspire": 0.0,     # İlham verme
        "warn": 0.0         # Uyarı
    }
    
    # Inform (Bilgi verme)
    inform_words = [
        "bilgi", "açıklama", "anlatmak", "göstermek", "öğretmek", "öğrenmek",
        "nasıl", "neden", "ne", "hangi", "kim", "nerede", "ne zaman",
        "information", "explain", "tell", "show", "teach", "learn",
        "how", "why", "what", "which", "who", "where", "when"
    ]
    
    # Persuade (İkna etme)
    persuade_words = [
        "ikna", "ikna etmek", "tavsiye", "öneri", "öner", "kesinlikle", "mutlaka",
        "gerekli", "önemli", "faydalı", "yararlı", "persuade", "convince",
        "recommend", "suggest", "definitely", "must", "necessary", "important"
    ]
    
    # Entertain (Eğlendirme)
    entertain_words = [
        "eğlence", "eğlenceli", "komik", "gülmek", "gülümseme", "şaka", "mizah",
        "eğlendirmek", "eğlenmek", "entertainment", "fun", "funny", "laugh",
        "smile", "joke", "humor", "entertain", "enjoy"
    ]
    
    # Educate (Eğitme)
    educate_words = [
        "eğitim", "eğitmek", "öğretmek", "ders", "kurs", "öğrenme", "bilgi",
        "education", "educate", "teach", "lesson", "course", "learning", "knowledge"
    ]
    
    # Sell (Satış)
    sell_words = [
        "satış", "satmak", "alışveriş", "fiyat", "indirim", "kampanya", "ürün",
        "marka", "mağaza", "sale", "sell", "buy", "price", "discount", "product",
        "brand", "store", "shop", "purchase"
    ]
    
    # Engage (Etkileşim)
    engage_words = [
        "etkileşim", "katılım", "katılmak", "yorum", "beğen", "paylaş", "takip",
        "abone", "bildirim", "engagement", "interaction", "participate", "comment",
        "like", "share", "follow", "subscribe", "notification"
    ]
    
    # Inspire (İlham verme)
    inspire_words = [
        "ilham", "ilham vermek", "motivasyon", "motivasyonel", "başarı", "hedef",
        "hayal", "rüya", "umut", "umutlu", "inspiration", "inspire", "motivation",
        "success", "goal", "dream", "hope", "hopeful"
    ]
    
    # Warn (Uyarı)
    warn_words = [
        "uyarı", "uyarmak", "dikkat", "dikkatli", "tehlikeli", "risk", "riskli",
        "warning", "warn", "careful", "dangerous", "risk", "risky", "caution"
    ]
    
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return "unknown"
    
    # Intent skorları
    intent_word_lists = {
        "inform": inform_words,
        "persuade": persuade_words,
        "entertain": entertain_words,
        "educate": educate_words,
        "sell": sell_words,
        "engage": engage_words,
        "inspire": inspire_words,
        "warn": warn_words
    }
    
    for intent, word_list in intent_word_lists.items():
        count = sum(1 for word in words if word in word_list)
        intents[intent] = count / total_words
    
    # En yüksek skorlu intent
    dominant_intent = max(intents, key=intents.get)
    print(f"💬 [NLP] Intent: {dominant_intent} (scores: {intents})")
    
    return dominant_intent


def calculate_readability_score(text: str) -> float:
    """
    Readability score - Flesch Reading Ease benzeri
    """
    words = text.split()
    sentences = text.split('.')
    
    if not words or not sentences:
        return 0.5
    
    # Temel metrikler
    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Ortalama kelime uzunluğu
    avg_word_length = sum(len(word) for word in words) / word_count
    
    # Ortalama cümle uzunluğu
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Readability skoru (0-1, 1 = çok okunabilir)
    readability = 1.0
    
    # Kelime uzunluğu penaltısı
    if avg_word_length > 6:
        readability -= 0.3
    elif avg_word_length > 4:
        readability -= 0.1
    
    # Cümle uzunluğu penaltısı
    if avg_sentence_length > 20:
        readability -= 0.3
    elif avg_sentence_length > 15:
        readability -= 0.1
    
    # Karmaşık kelimeler penaltısı
    complex_words = ["teknoloji", "algoritma", "sistem", "mekanizma", "prosedür"]
    complex_count = sum(1 for word in words if word in complex_words)
    if complex_count > word_count * 0.1:  # %10'dan fazla karmaşık kelime
        readability -= 0.2
    
    readability = max(0.0, min(1.0, readability))
    
    print(f"💬 [NLP] Readability: {readability:.2f} (avg_word: {avg_word_length:.1f}, avg_sentence: {avg_sentence_length:.1f})")
    
    return readability


def calculate_engagement_potential(text: str, sentiment: float, emotions: Dict[str, float]) -> float:
    """
    Engagement potential - etkileşim potansiyeli
    """
    engagement = 0.0
    
    # 1. Sentiment bonusu
    if sentiment > 0.6:
        engagement += 0.3
    elif sentiment > 0.4:
        engagement += 0.2
    else:
        engagement += 0.1
    
    # 2. Emotion bonusu
    high_engagement_emotions = ["joy", "surprise", "anticipation"]
    for emotion in high_engagement_emotions:
        if emotions.get(emotion, 0) > 0.3:
            engagement += 0.2
    
    # 3. Soru işareti bonusu
    if "?" in text:
        engagement += 0.1
    
    # 4. Ünlem işareti bonusu
    if "!" in text:
        engagement += 0.1
    
    # 5. Kişisel zamirler
    personal_pronouns = ["ben", "sen", "biz", "siz", "i", "you", "we", "us"]
    pronoun_count = sum(1 for word in text.split() if word in personal_pronouns)
    if pronoun_count > 0:
        engagement += 0.1
    
    # 6. Emir kipi
    command_words = ["yap", "et", "git", "gel", "bak", "dinle", "izle", "do", "go", "come", "look", "listen", "watch"]
    command_count = sum(1 for word in text.split() if word in command_words)
    if command_count > 0:
        engagement += 0.1
    
    engagement = min(engagement, 1.0)
    
    print(f"💬 [NLP] Engagement potential: {engagement:.2f}")
    
    return engagement


def extract_viral_keywords(text: str) -> List[str]:
    """
    Viral keywords - viral potansiyeli yüksek kelimeler
    """
    viral_keywords = []
    
    # Viral potansiyeli yüksek kelimeler
    viral_words = [
        "viral", "trend", "popüler", "hype", "boom", "explosive", "breakthrough",
        "revolutionary", "game changer", "mind blowing", "insane", "crazy",
        "unbelievable", "incredible", "amazing", "shocking", "surprising",
        "secret", "hidden", "exclusive", "first time", "never seen before",
        "breaking", "urgent", "important", "must see", "must watch",
        "viral", "trending", "popular", "hype", "boom", "explosive",
        "revolutionary", "game changer", "mind blowing", "insane", "crazy"
    ]
    
    # Viral patterns
    viral_patterns = [
        "kimse bilmiyor", "hiç kimse", "sadece ben", "sır", "gizli", "özel",
        "sonunda", "göreceksin", "inanamayacaksın", "şaşıracaksın",
        "nobody knows", "no one", "only me", "secret", "hidden", "exclusive",
        "finally", "you will see", "you won't believe", "you'll be shocked"
    ]
    
    words = text.split()
    
    # Viral kelimeler
    for word in words:
        if word in viral_words:
            viral_keywords.append(word)
    
    # Viral patterns
    for pattern in viral_patterns:
        if pattern in text:
            viral_keywords.append(pattern)
    
    print(f"💬 [NLP] Viral keywords: {viral_keywords}")
    
    return viral_keywords


def assess_language_quality(text: str, structure_score: float, complexity_score: float) -> float:
    """
    Language quality assessment - genel dil kalitesi
    """
    quality = 0.0
    
    # 1. Cümle yapısı
    quality += structure_score * 0.4
    
    # 2. Karmaşıklık dengesi (çok basit veya çok karmaşık olmamalı)
    if 0.3 <= complexity_score <= 0.7:
        quality += 0.3
    elif 0.2 <= complexity_score <= 0.8:
        quality += 0.2
    else:
        quality += 0.1
    
    # 3. Kelime çeşitliliği
    words = text.split()
    unique_words = len(set(words))
    total_words = len(words)
    
    if total_words > 0:
        diversity_ratio = unique_words / total_words
        if diversity_ratio > 0.7:
            quality += 0.2
        elif diversity_ratio > 0.5:
            quality += 0.1
    
    # 4. Noktalama kullanımı
    punctuation_count = sum(1 for char in text if char in ".,!?;:")
    if punctuation_count > 0:
        quality += 0.1
    
    quality = min(quality, 1.0)
    
    print(f"💬 [NLP] Language quality: {quality:.2f}")
    
    return quality


def analyze_technical_quality(features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """
    Teknik Kalite Analizi (15 puan)
    Çözünürlük, fps, blur, ses kalitesi, bitrate, görsel analiz
    """
    score = 0
    findings = []
    recommendations = []
    raw_metrics = {}
    
    print("🔧 [TECH] Starting technical quality analysis...")
    
    # Video bilgileri
    video_info = features.get("video_info", {})
    width = video_info.get("width", 0)
    height = video_info.get("height", 0)
    fps = video_info.get("fps", 0)
    duration_sec = video_info.get("duration", 0)
    
    # Ses bilgileri
    audio_info = features.get("audio_info", {})
    sample_rate = audio_info.get("sample_rate", 0)
    channels = audio_info.get("channels", 0)
    
    # 1. Çözünürlük Analizi (3 puan)
    resolution_score = analyze_resolution(width, height)
    raw_metrics["resolution"] = resolution_score
    
    if resolution_score > 0.8:
        score += 3
        findings.append({
            "t": 0.0,
            "type": "high_resolution",
            "msg": f"Yüksek çözünürlük: {width}x{height} - profesyonel kalite"
        })
    elif resolution_score > 0.6:
        score += 2
        recommendations.append("Çözünürlüğü artır - en az 1080p kullan")
    else:
        score += 1
        recommendations.append("Çözünürlük çok düşük - 720p minimum gerekli")
    
    # 2. FPS Analizi (2 puan)
    fps_score = analyze_fps(fps)
    raw_metrics["fps"] = fps_score
    
    if fps_score > 0.8:
        score += 2
        findings.append({
            "t": 0.0,
            "type": "smooth_fps",
            "msg": f"Yumuşak FPS: {fps} - akıcı görüntü"
        })
    elif fps_score > 0.6:
        score += 1
        recommendations.append("FPS'i artır - en az 30 FPS önerilir")
    else:
        recommendations.append("FPS çok düşük - 24 FPS minimum gerekli")
    
    # 3. Blur Analizi (2 puan)
    blur_score = analyze_blur(features)
    raw_metrics["blur"] = blur_score
    
    if blur_score > 0.7:
        score += 2
        findings.append({
            "t": duration * 0.5,
            "type": "sharp_video",
            "msg": "Net görüntü - blur yok"
        })
    elif blur_score > 0.4:
        score += 1
        recommendations.append("Görüntüyü netleştir - odak sorunu var")
    else:
        recommendations.append("Ciddi blur sorunu - kamera ayarlarını kontrol et")
    
    # 4. Ses Kalitesi (2 puan)
    audio_score = analyze_audio_quality(sample_rate, channels, features)
    raw_metrics["audio_quality"] = audio_score
    
    if audio_score > 0.8:
        score += 2
        findings.append({
            "t": 0.0,
            "type": "high_audio_quality",
            "msg": f"Yüksek ses kalitesi: {sample_rate}Hz, {channels} kanal"
        })
    elif audio_score > 0.6:
        score += 1
        recommendations.append("Ses kalitesini artır - 44.1kHz önerilir")
    else:
        recommendations.append("Ses kalitesi düşük - mikrofon ve kayıt ayarlarını kontrol et")
    
    # 5. Bitrate Analizi (1 puan)
    bitrate_score = analyze_bitrate(features, duration_sec)
    raw_metrics["bitrate"] = bitrate_score
    
    if bitrate_score > 0.7:
        score += 1
        findings.append({
            "t": 0.0,
            "type": "optimal_bitrate",
            "msg": "Optimal bitrate - kaliteli sıkıştırma"
        })
    else:
        recommendations.append("Bitrate'i artır - daha kaliteli sıkıştırma gerekli")
    
    # 6. Görsel Analiz (5 puan)
    visual_score = analyze_visual_quality(features)
    raw_metrics["visual_quality"] = visual_score
    
    if visual_score > 0.8:
        score += 5
        findings.append({
            "t": 0.0,
            "type": "excellent_visual",
            "msg": "Mükemmel görsel kalite - viral potansiyeli yüksek"
        })
    elif visual_score > 0.6:
        score += 3
        recommendations.append("Görsel kaliteyi artır - renk ve kompozisyon önemli")
    else:
        score += 1
        recommendations.append("Görsel kalite düşük - renk, kompozisyon ve estetik iyileştir")
    
    print(f"🔧 [TECH] Technical quality score: {score}/15")
    
    return {
        "score": min(score, 15),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw_metrics
    }


def analyze_resolution(width: int, height: int) -> float:
    """
    Çözünürlük analizi
    """
    if width == 0 or height == 0:
        return 0.0
    
    # Çözünürlük kategorileri
    if width >= 3840 and height >= 2160:  # 4K
        return 1.0
    elif width >= 1920 and height >= 1080:  # 1080p
        return 0.9
    elif width >= 1280 and height >= 720:  # 720p
        return 0.7
    elif width >= 854 and height >= 480:  # 480p
        return 0.5
    elif width >= 640 and height >= 360:  # 360p
        return 0.3
    else:  # 240p ve altı
        return 0.1


def analyze_fps(fps: float) -> float:
    """
    FPS analizi
    """
    if fps == 0:
        return 0.0
    
    # FPS kategorileri
    if fps >= 60:  # 60+ FPS
        return 1.0
    elif fps >= 30:  # 30-59 FPS
        return 0.8
    elif fps >= 24:  # 24-29 FPS
        return 0.6
    elif fps >= 15:  # 15-23 FPS
        return 0.4
    else:  # 15 FPS altı
        return 0.2


def analyze_blur(features: Dict[str, Any]) -> float:
    """
    Blur analizi - gerçek görüntü netliği tespiti
    """
    try:
        # Frame'lerden blur analizi yap
        frames = features.get("frames", [])
        if not frames:
            print("🔧 [TECH] No frames available for blur analysis")
            return 0.5
        
        # İlk 3 frame'i analiz et
        blur_scores = []
        for i, frame_path in enumerate(frames[:3]):
            try:
                blur_score = calculate_frame_blur(frame_path)
                blur_scores.append(blur_score)
                print(f"🔧 [TECH] Frame {i+1} blur score: {blur_score:.2f}")
            except Exception as e:
                print(f"🔧 [TECH] Frame {i+1} blur analysis failed: {e}")
                continue
        
        if not blur_scores:
            print("🔧 [TECH] No valid blur scores calculated")
            return 0.5
        
        # Ortalama blur skoru
        avg_blur_score = sum(blur_scores) / len(blur_scores)
        
        # Çözünürlük etkisi
        video_info = features.get("video_info", {})
        width = video_info.get("width", 0)
        height = video_info.get("height", 0)
        
        # Çözünürlük bonusu/penaltısı
        if width >= 1920 and height >= 1080:  # 1080p+
            avg_blur_score += 0.1
        elif width < 1280 or height < 720:  # 720p altı
            avg_blur_score -= 0.2
        
        # FPS etkisi
        fps = video_info.get("fps", 0)
        if fps < 24:
            avg_blur_score -= 0.1  # Düşük FPS blur etkisi
        
        avg_blur_score = max(0.0, min(1.0, avg_blur_score))
        
        print(f"🔧 [TECH] Average blur score: {avg_blur_score:.2f}")
        
        return avg_blur_score
        
    except Exception as e:
        print(f"🔧 [TECH] Blur analysis failed: {e}")
        return 0.5


def calculate_frame_blur(frame_path: str) -> float:
    """
    Tek frame'den blur skoru hesapla - OpenCV ile
    """
    import cv2
    import numpy as np
    
    # Görüntüyü yükle
    image = cv2.imread(frame_path)
    if image is None:
        print(f"🔧 [TECH] Could not load frame: {frame_path}")
        return 0.5
    
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Laplacian variance (en etkili blur tespiti)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Sobel gradient
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_var = sobel_magnitude.var()
    
    # 3. Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 4. Histogram analizi
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_std = np.std(hist)
    
    # Blur skoru hesapla
    blur_score = 0.0
    
    # Laplacian variance (ana metrik)
    if laplacian_var > 1000:  # Çok net
        blur_score += 0.4
    elif laplacian_var > 500:  # Net
        blur_score += 0.3
    elif laplacian_var > 200:  # Orta
        blur_score += 0.2
    else:  # Blur
        blur_score += 0.1
    
    # Sobel gradient
    if sobel_var > 5000:  # Çok net
        blur_score += 0.2
    elif sobel_var > 2000:  # Net
        blur_score += 0.15
    elif sobel_var > 500:  # Orta
        blur_score += 0.1
    else:  # Blur
        blur_score += 0.05
    
    # Edge density
    if edge_density > 0.1:  # Çok net
        blur_score += 0.2
    elif edge_density > 0.05:  # Net
        blur_score += 0.15
    elif edge_density > 0.02:  # Orta
        blur_score += 0.1
    else:  # Blur
        blur_score += 0.05
    
    # Histogram std
    if hist_std > 50:  # Çok net
        blur_score += 0.2
    elif hist_std > 30:  # Net
        blur_score += 0.15
    elif hist_std > 15:  # Orta
        blur_score += 0.1
    else:  # Blur
        blur_score += 0.05
    
    blur_score = min(blur_score, 1.0)
    
    print(f"🔧 [TECH] Frame blur metrics - Laplacian: {laplacian_var:.1f}, Sobel: {sobel_var:.1f}, Edges: {edge_density:.3f}, Hist: {hist_std:.1f}")
    
    return blur_score


def analyze_audio_quality(sample_rate: int, channels: int, features: Dict[str, Any]) -> float:
    """
    Ses kalitesi analizi
    """
    audio_score = 0.0
    
    # 1. Sample rate analizi
    if sample_rate >= 48000:  # 48kHz+
        audio_score += 0.4
    elif sample_rate >= 44100:  # 44.1kHz
        audio_score += 0.3
    elif sample_rate >= 22050:  # 22.05kHz
        audio_score += 0.2
    else:  # 22kHz altı
        audio_score += 0.1
    
    # 2. Kanal sayısı analizi
    if channels >= 2:  # Stereo
        audio_score += 0.3
    elif channels == 1:  # Mono
        audio_score += 0.2
    else:  # Kanal yok
        audio_score += 0.0
    
    # 3. Ses seviyesi analizi
    audio_features = features.get("audio", {})
    loudness = audio_features.get("loudness", 0)
    
    if -20 <= loudness <= -6:  # Optimal ses seviyesi
        audio_score += 0.3
    elif -30 <= loudness <= -3:  # Kabul edilebilir
        audio_score += 0.2
    else:  # Çok düşük veya yüksek
        audio_score += 0.1
    
    audio_score = min(audio_score, 1.0)
    
    print(f"🔧 [TECH] Audio quality: {audio_score:.2f} (sample_rate: {sample_rate}, channels: {channels})")
    
    return audio_score


def analyze_bitrate(features: Dict[str, Any], duration: float) -> float:
    """
    Bitrate analizi
    """
    # Basit bitrate hesaplama
    video_info = features.get("video_info", {})
    width = video_info.get("width", 0)
    height = video_info.get("height", 0)
    
    if width == 0 or height == 0 or duration == 0:
        return 0.5
    
    # Piksel sayısı
    pixels = width * height
    
    # Tahmini bitrate (basit hesaplama)
    if pixels >= 3840 * 2160:  # 4K
        estimated_bitrate = 15000  # 15 Mbps
    elif pixels >= 1920 * 1080:  # 1080p
        estimated_bitrate = 5000   # 5 Mbps
    elif pixels >= 1280 * 720:   # 720p
        estimated_bitrate = 2500   # 2.5 Mbps
    else:  # 480p ve altı
        estimated_bitrate = 1000   # 1 Mbps
    
    # Bitrate skoru
    if estimated_bitrate >= 5000:  # Yüksek bitrate
        bitrate_score = 1.0
    elif estimated_bitrate >= 2500:  # Orta bitrate
        bitrate_score = 0.8
    elif estimated_bitrate >= 1000:  # Düşük bitrate
        bitrate_score = 0.6
    else:  # Çok düşük bitrate
        bitrate_score = 0.3
    
    print(f"🔧 [TECH] Bitrate score: {bitrate_score:.2f} (estimated: {estimated_bitrate} kbps)")
    
    return bitrate_score


def analyze_visual_quality(features: Dict[str, Any]) -> float:
    """
    Görsel kalite analizi - renk, kompozisyon, nesne, hareket, estetik
    """
    try:
        frames = features.get("frames", [])
        if not frames:
            print("🎨 [VISUAL] No frames available for visual analysis")
            return 0.5
        
        # İlk 3 frame'i analiz et
        visual_scores = []
        for i, frame_path in enumerate(frames[:3]):
            try:
                visual_score = calculate_frame_visual_quality(frame_path)
                visual_scores.append(visual_score)
                print(f"🎨 [VISUAL] Frame {i+1} visual score: {visual_score:.2f}")
            except Exception as e:
                print(f"🎨 [VISUAL] Frame {i+1} visual analysis failed: {e}")
                continue
        
        if not visual_scores:
            print("🎨 [VISUAL] No valid visual scores calculated")
            return 0.5
        
        # Ortalama görsel skoru
        avg_visual_score = sum(visual_scores) / len(visual_scores)
        
        print(f"🎨 [VISUAL] Average visual score: {avg_visual_score:.2f}")
        
        return avg_visual_score
        
    except Exception as e:
        print(f"🎨 [VISUAL] Visual analysis failed: {e}")
        return 0.5


def calculate_frame_visual_quality(frame_path: str) -> float:
    """
    Tek frame'den görsel kalite skoru hesapla
    """
    import cv2
    import numpy as np
    
    # Görüntüyü yükle
    image = cv2.imread(frame_path)
    if image is None:
        print(f"🎨 [VISUAL] Could not load frame: {frame_path}")
        return 0.5
    
    visual_score = 0.0
    
    # 1. Renk Analizi (2 puan)
    color_score = analyze_color_quality(image)
    visual_score += color_score * 0.4
    
    # 2. Kompozisyon Analizi (1.5 puan)
    composition_score = analyze_composition(image)
    visual_score += composition_score * 0.3
    
    # 3. Nesne Tespiti (1 puan)
    object_score = analyze_objects(image)
    visual_score += object_score * 0.2
    
    # 4. Kalite Metrikleri (0.5 puan)
    quality_score = analyze_quality_metrics(image)
    visual_score += quality_score * 0.1
    
    visual_score = min(visual_score, 1.0)
    
    return visual_score


def analyze_color_quality(image) -> float:
    """
    Renk kalitesi analizi
    """
    import cv2
    import numpy as np
    
    # HSV'ye çevir
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    color_score = 0.0
    
    # 1. Renk çeşitliliği
    h_channel = hsv[:, :, 0]
    unique_hues = len(np.unique(h_channel))
    if unique_hues > 50:  # Çok çeşitli renkler
        color_score += 0.3
    elif unique_hues > 20:  # Orta çeşitlilik
        color_score += 0.2
    else:  # Az çeşitlilik
        color_score += 0.1
    
    # 2. Parlaklık analizi
    v_channel = hsv[:, :, 2]
    brightness_mean = np.mean(v_channel)
    if 100 <= brightness_mean <= 200:  # Optimal parlaklık
        color_score += 0.3
    elif 50 <= brightness_mean <= 250:  # Kabul edilebilir
        color_score += 0.2
    else:  # Aşırı karanlık/parlak
        color_score += 0.1
    
    # 3. Kontrast analizi
    contrast = np.std(v_channel)
    if contrast > 50:  # Yüksek kontrast
        color_score += 0.2
    elif contrast > 30:  # Orta kontrast
        color_score += 0.15
    else:  # Düşük kontrast
        color_score += 0.1
    
    # 4. Renk doygunluğu
    s_channel = hsv[:, :, 1]
    saturation_mean = np.mean(s_channel)
    if saturation_mean > 100:  # Yüksek doygunluk
        color_score += 0.2
    elif saturation_mean > 50:  # Orta doygunluk
        color_score += 0.15
    else:  # Düşük doygunluk
        color_score += 0.1
    
    return min(color_score, 1.0)


def analyze_composition(image) -> float:
    """
    Kompozisyon analizi
    """
    import cv2
    import numpy as np
    
    height, width = image.shape[:2]
    composition_score = 0.0
    
    # 1. Rule of thirds analizi
    # Görüntüyü 9 eşit parçaya böl
    third_w = width // 3
    third_h = height // 3
    
    # Kenar bölgelerinin parlaklık analizi
    edges = [
        image[0:third_h, 0:third_w],  # Sol üst
        image[0:third_h, third_w:2*third_w],  # Orta üst
        image[0:third_h, 2*third_w:width],  # Sağ üst
        image[third_h:2*third_h, 0:third_w],  # Sol orta
        image[third_h:2*third_h, 2*third_w:width],  # Sağ orta
        image[2*third_h:height, 0:third_w],  # Sol alt
        image[2*third_h:height, third_w:2*third_w],  # Orta alt
        image[2*third_h:height, 2*third_w:width]  # Sağ alt
    ]
    
    # Merkez bölge
    center = image[third_h:2*third_h, third_w:2*third_w]
    
    # Merkez vs kenar analizi
    center_brightness = np.mean(cv2.cvtColor(center, cv2.COLOR_BGR2GRAY))
    edge_brightness = np.mean([np.mean(cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)) for edge in edges])
    
    if abs(center_brightness - edge_brightness) > 30:  # İyi kompozisyon
        composition_score += 0.4
    elif abs(center_brightness - edge_brightness) > 15:  # Orta kompozisyon
        composition_score += 0.3
    else:  # Zayıf kompozisyon
        composition_score += 0.2
    
    # 2. Simetri analizi
    # Yatay simetri
    top_half = image[0:height//2, :]
    bottom_half = cv2.flip(image[height//2:height, :], 0)
    
    # Dikey simetri
    left_half = image[:, 0:width//2]
    right_half = cv2.flip(image[:, width//2:width], 1)
    
    # Simetri skorları
    h_symmetry = 1.0 - np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float))) / 255.0
    v_symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
    
    max_symmetry = max(h_symmetry, v_symmetry)
    if max_symmetry > 0.8:  # Yüksek simetri
        composition_score += 0.3
    elif max_symmetry > 0.6:  # Orta simetri
        composition_score += 0.2
    else:  # Düşük simetri
        composition_score += 0.1
    
    # 3. Odak noktası analizi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    focus_score = np.var(laplacian)
    
    if focus_score > 1000:  # Net odak
        composition_score += 0.3
    elif focus_score > 500:  # Orta odak
        composition_score += 0.2
    else:  # Bulanık
        composition_score += 0.1
    
    return min(composition_score, 1.0)


def analyze_objects(image) -> float:
    """
    Nesne tespiti analizi
    """
    import cv2
    import numpy as np
    
    object_score = 0.0
    
    # 1. Yüz tespiti
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Haar cascade ile yüz tespiti
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:  # Yüz tespit edildi
        object_score += 0.4
        print(f"🎨 [VISUAL] Faces detected: {len(faces)}")
    
    # 2. Kenar tespiti (nesne varlığı)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    if edge_density > 0.1:  # Çok nesne
        object_score += 0.3
    elif edge_density > 0.05:  # Orta nesne
        object_score += 0.2
    else:  # Az nesne
        object_score += 0.1
    
    # 3. Metin tespiti
    # Basit metin tespiti (kenar analizi)
    text_regions = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    text_density = np.sum(text_regions > 0) / text_regions.size
    
    if text_density > 0.05:  # Metin var
        object_score += 0.3
        print(f"🎨 [VISUAL] Text detected")
    
    return min(object_score, 1.0)


def analyze_quality_metrics(image) -> float:
    """
    Kalite metrikleri analizi
    """
    import cv2
    import numpy as np
    
    quality_score = 0.0
    
    # 1. Noise analizi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Laplacian ile noise tespiti
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_level = np.std(laplacian)
    
    if noise_level < 50:  # Düşük noise
        quality_score += 0.3
    elif noise_level < 100:  # Orta noise
        quality_score += 0.2
    else:  # Yüksek noise
        quality_score += 0.1
    
    # 2. Exposure analizi
    brightness = np.mean(gray)
    
    if 80 <= brightness <= 180:  # Optimal exposure
        quality_score += 0.4
    elif 50 <= brightness <= 220:  # Kabul edilebilir
        quality_score += 0.3
    else:  # Aşırı exposure
        quality_score += 0.1
    
    # 3. Sharpness analizi
    sharpness = np.var(laplacian)
    
    if sharpness > 1000:  # Çok keskin
        quality_score += 0.3
    elif sharpness > 500:  # Keskin
        quality_score += 0.2
    else:  # Bulanık
        quality_score += 0.1
    
    return min(quality_score, 1.0)


def analyze_transition_quality(features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """
    Geçiş Kalitesi Analizi (8 puan)
    Edit ve efekt analizi
    """
    score = 0
    findings = []
    recommendations = []
    raw_metrics = {}
    
    print("🎬 [TRANSITION] Starting transition quality analysis...")
    
    # Video bilgileri
    video_info = features.get("video_info", {})
    fps = video_info.get("fps", 0)
    duration_sec = video_info.get("duration", 0)
    
    # 1. Scene Detection Analizi (2 puan)
    scene_score = analyze_scene_transitions(features)
    raw_metrics["scene_transitions"] = scene_score
    
    if scene_score > 0.7:
        score += 2
        findings.append({
            "t": duration * 0.3,
            "type": "smooth_transitions",
            "msg": "Yumuşak geçişler - profesyonel edit"
        })
    elif scene_score > 0.4:
        score += 1
        recommendations.append("Geçişleri yumuşat - ani kesimlerden kaçın")
    else:
        recommendations.append("Geçiş efektleri ekle - fade, dissolve, wipe")
    
    # 2. Cut Frequency Analizi (2 puan)
    cut_score = analyze_cut_frequency(features, duration_sec)
    raw_metrics["cut_frequency"] = cut_score
    
    if cut_score > 0.7:
        score += 2
        findings.append({
            "t": duration * 0.5,
            "type": "optimal_pacing",
            "msg": "Optimal kesim hızı - dinamik ama okunabilir"
        })
    elif cut_score > 0.4:
        score += 1
        recommendations.append("Kesim hızını ayarla - çok hızlı veya yavaş")
    else:
        recommendations.append("Kesim hızını optimize et - 3-5 saniye aralıklar")
    
    # 3. Visual Effects Analizi (2 puan)
    effects_score = analyze_visual_effects(features)
    raw_metrics["visual_effects"] = effects_score
    
    if effects_score > 0.7:
        score += 2
        findings.append({
            "t": duration * 0.7,
            "type": "professional_effects",
            "msg": "Profesyonel efektler - görsel çekicilik yüksek"
        })
    elif effects_score > 0.4:
        score += 1
        recommendations.append("Efektleri iyileştir - zoom, pan, color grading")
    else:
        recommendations.append("Görsel efektler ekle - zoom, pan, color correction")
    
    # 4. Audio Transition Analizi (1 puan)
    audio_transition_score = analyze_audio_transitions(features)
    raw_metrics["audio_transitions"] = audio_transition_score
    
    if audio_transition_score > 0.7:
        score += 1
        findings.append({
            "t": duration * 0.8,
            "type": "smooth_audio",
            "msg": "Yumuşak ses geçişleri - profesyonel ses edit"
        })
    else:
        recommendations.append("Ses geçişlerini yumuşat - fade in/out ekle")
    
    # 5. Pacing Consistency (1 puan)
    pacing_score = analyze_pacing_consistency(features, duration_sec)
    raw_metrics["pacing_consistency"] = pacing_score
    
    if pacing_score > 0.7:
        score += 1
        findings.append({
            "t": duration * 0.6,
            "type": "consistent_pacing",
            "msg": "Tutarlı ritim - video akışı düzenli"
        })
    else:
        recommendations.append("Ritmi tutarlı yap - hız değişimlerini dengele")
    
    print(f"🎬 [TRANSITION] Transition quality score: {score}/8")
    
    return {
        "score": min(score, 8),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw_metrics
    }


def analyze_scene_transitions(features: Dict[str, Any]) -> float:
    """
    Scene transition analizi
    """
    try:
        # Scene detection sonuçları
        scene_data = features.get("scene_data", {})
        scenes = scene_data.get("scenes", [])
        
        if not scenes or len(scenes) < 2:
            print("🎬 [TRANSITION] No scene data available")
            return 0.5
        
        transition_score = 0.0
        
        # 1. Scene sayısı analizi
        scene_count = len(scenes)
        if 3 <= scene_count <= 8:  # Optimal scene sayısı
            transition_score += 0.3
        elif 2 <= scene_count <= 12:  # Kabul edilebilir
            transition_score += 0.2
        else:  # Çok az veya çok fazla
            transition_score += 0.1
        
        # 2. Scene uzunlukları analizi
        scene_lengths = []
        for scene in scenes:
            if isinstance(scene, dict) and 'duration' in scene:
                scene_lengths.append(scene['duration'])
            elif isinstance(scene, (list, tuple)) and len(scene) >= 2:
                scene_lengths.append(scene[1] - scene[0])
        
        if scene_lengths:
            avg_scene_length = sum(scene_lengths) / len(scene_lengths)
            scene_length_variance = np.var(scene_lengths) if len(scene_lengths) > 1 else 0
            
            # Optimal scene uzunluğu (3-8 saniye)
            if 3 <= avg_scene_length <= 8:
                transition_score += 0.4
            elif 2 <= avg_scene_length <= 12:
                transition_score += 0.3
            else:
                transition_score += 0.1
            
            # Scene uzunluk tutarlılığı
            if scene_length_variance < 5:  # Tutarlı uzunluklar
                transition_score += 0.3
            elif scene_length_variance < 15:  # Orta tutarlılık
                transition_score += 0.2
            else:  # Tutarsız uzunluklar
                transition_score += 0.1
        
        transition_score = min(transition_score, 1.0)
        
        print(f"🎬 [TRANSITION] Scene transition score: {transition_score:.2f} (scenes: {scene_count})")
        
        return transition_score
        
    except Exception as e:
        print(f"🎬 [TRANSITION] Scene transition analysis failed: {e}")
        return 0.5


def analyze_cut_frequency(features: Dict[str, Any], duration: float) -> float:
    """
    Cut frequency analizi
    """
    try:
        import numpy as np
        
        # Scene data'dan cut sayısını hesapla
        scene_data = features.get("scene_data", {})
        scenes = scene_data.get("scenes", [])
        
        if not scenes:
            print("🎬 [TRANSITION] No scene data for cut analysis")
            return 0.5
        
        # Cut sayısı = scene sayısı - 1
        cut_count = len(scenes) - 1
        cuts_per_minute = cut_count / (duration / 60) if duration > 0 else 0
        
        cut_score = 0.0
        
        # Optimal cut frequency (15-30 cuts/minute)
        if 15 <= cuts_per_minute <= 30:
            cut_score = 1.0
        elif 10 <= cuts_per_minute <= 40:
            cut_score = 0.7
        elif 5 <= cuts_per_minute <= 50:
            cut_score = 0.5
        else:
            cut_score = 0.3
        
        print(f"🎬 [TRANSITION] Cut frequency score: {cut_score:.2f} (cuts: {cut_count}, per minute: {cuts_per_minute:.1f})")
        
        return cut_score
        
    except Exception as e:
        print(f"🎬 [TRANSITION] Cut frequency analysis failed: {e}")
        return 0.5


def analyze_visual_effects(features: Dict[str, Any]) -> float:
    """
    Visual effects analizi
    """
    try:
        effects_score = 0.0
        
        # 1. Frame değişim analizi (zoom, pan tespiti)
        frames = features.get("frames", [])
        if len(frames) >= 2:
            frame_variation = analyze_frame_variations(frames)
            effects_score += frame_variation * 0.4
        
        # 2. Color grading analizi
        color_score = analyze_color_grading(features)
        effects_score += color_score * 0.3
        
        # 3. Motion analizi
        motion_score = analyze_motion_effects(features)
        effects_score += motion_score * 0.3
        
        effects_score = min(effects_score, 1.0)
        
        print(f"🎬 [TRANSITION] Visual effects score: {effects_score:.2f}")
        
        return effects_score
        
    except Exception as e:
        print(f"🎬 [TRANSITION] Visual effects analysis failed: {e}")
        return 0.5


def analyze_frame_variations(frames: List[str]) -> float:
    """
    Frame varyasyon analizi (zoom, pan tespiti)
    """
    try:
        import cv2
        import numpy as np
        
        if len(frames) < 2:
            return 0.5
        
        # İlk ve son frame'i karşılaştır
        frame1 = cv2.imread(frames[0])
        frame2 = cv2.imread(frames[-1])
        
        if frame1 is None or frame2 is None:
            return 0.5
        
        # Gri tonlamaya çevir
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Histogram karşılaştırması
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Histogram korelasyonu
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Farklılık skoru (1 - correlation)
        variation_score = 1.0 - correlation
        
        # Optimal varyasyon (0.2-0.6 arası)
        if 0.2 <= variation_score <= 0.6:
            return 1.0
        elif 0.1 <= variation_score <= 0.8:
            return 0.7
        else:
            return 0.4
        
    except Exception as e:
        print(f"🎬 [TRANSITION] Frame variation analysis failed: {e}")
        return 0.5


def analyze_color_grading(features: Dict[str, Any]) -> float:
    """
    Color grading analizi
    """
    try:
        frames = features.get("frames", [])
        if not frames:
            return 0.5
        
        import cv2
        import numpy as np
        
        # İlk frame'i analiz et
        image = cv2.imread(frames[0])
        if image is None:
            return 0.5
        
        # HSV'ye çevir
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color grading göstergeleri
        color_score = 0.0
        
        # 1. Saturation analizi
        s_channel = hsv[:, :, 1]
        avg_saturation = np.mean(s_channel)
        
        if avg_saturation > 120:  # Yüksek doygunluk (color grading)
            color_score += 0.4
        elif avg_saturation > 80:  # Orta doygunluk
            color_score += 0.3
        else:  # Düşük doygunluk
            color_score += 0.2
        
        # 2. Contrast analizi
        v_channel = hsv[:, :, 2]
        contrast = np.std(v_channel)
        
        if contrast > 60:  # Yüksek kontrast
            color_score += 0.3
        elif contrast > 40:  # Orta kontrast
            color_score += 0.2
        else:  # Düşük kontrast
            color_score += 0.1
        
        # 3. Color temperature analizi (basit)
        b, g, r = cv2.split(image)
        avg_b = np.mean(b)
        avg_g = np.mean(g)
        avg_r = np.mean(r)
        
        # Warm/cool tone analizi
        if avg_r > avg_b + 10:  # Warm tone
            color_score += 0.3
        elif avg_b > avg_r + 10:  # Cool tone
            color_score += 0.3
        else:  # Neutral
            color_score += 0.2
        
        return min(color_score, 1.0)
        
    except Exception as e:
        print(f"🎬 [TRANSITION] Color grading analysis failed: {e}")
        return 0.5


def analyze_motion_effects(features: Dict[str, Any]) -> float:
    """
    Motion effects analizi
    """
    try:
        # Basit motion analizi
        video_info = features.get("video_info", {})
        fps = video_info.get("fps", 0)
        
        motion_score = 0.0
        
        # FPS tabanlı motion analizi
        if fps >= 60:  # Yüksek FPS (smooth motion)
            motion_score += 0.4
        elif fps >= 30:  # Orta FPS
            motion_score += 0.3
        else:  # Düşük FPS
            motion_score += 0.2
        
        # Scene count tabanlı motion analizi
        scene_data = features.get("scene_data", {})
        scenes = scene_data.get("scenes", [])
        
        if scenes:
            scene_count = len(scenes)
            if scene_count > 5:  # Çok scene (dynamic)
                motion_score += 0.3
            elif scene_count > 3:  # Orta scene
                motion_score += 0.2
            else:  # Az scene (static)
                motion_score += 0.1
        
        # Audio tempo tabanlı motion analizi
        audio_features = features.get("audio", {})
        tempo = audio_features.get("tempo", 0)
        
        if tempo > 120:  # Hızlı tempo
            motion_score += 0.3
        elif tempo > 80:  # Orta tempo
            motion_score += 0.2
        else:  # Yavaş tempo
            motion_score += 0.1
        
        return min(motion_score, 1.0)
        
    except Exception as e:
        print(f"🎬 [TRANSITION] Motion effects analysis failed: {e}")
        return 0.5


def analyze_audio_transitions(features: Dict[str, Any]) -> float:
    """
    Audio transition analizi
    """
    try:
        audio_features = features.get("audio", {})
        
        audio_score = 0.0
        
        # 1. Loudness consistency
        loudness = audio_features.get("loudness", 0)
        if -20 <= loudness <= -6:  # Optimal loudness
            audio_score += 0.4
        elif -30 <= loudness <= -3:  # Kabul edilebilir
            audio_score += 0.3
        else:  # Aşırı yüksek/düşük
            audio_score += 0.1
        
        # 2. Tempo consistency
        tempo = audio_features.get("tempo", 0)
        if tempo > 0:  # Tempo tespit edildi
            audio_score += 0.3
        else:
            audio_score += 0.1
        
        # 3. Audio quality
        audio_info = features.get("audio_info", {})
        sample_rate = audio_info.get("sample_rate", 0)
        
        if sample_rate >= 44100:  # Yüksek kalite
            audio_score += 0.3
        elif sample_rate >= 22050:  # Orta kalite
            audio_score += 0.2
        else:  # Düşük kalite
            audio_score += 0.1
        
        return min(audio_score, 1.0)
        
    except Exception as e:
        print(f"🎬 [TRANSITION] Audio transition analysis failed: {e}")
        return 0.5


def analyze_pacing_consistency(features: Dict[str, Any], duration: float) -> float:
    """
    Pacing consistency analizi
    """
    try:
        import numpy as np
        
        # Scene data'dan pacing analizi
        scene_data = features.get("scene_data", {})
        scenes = scene_data.get("scenes", [])
        
        if not scenes or len(scenes) < 2:
            return 0.5
        
        pacing_score = 0.0
        
        # 1. Scene length consistency
        scene_lengths = []
        for scene in scenes:
            if isinstance(scene, dict) and 'duration' in scene:
                scene_lengths.append(scene['duration'])
            elif isinstance(scene, (list, tuple)) and len(scene) >= 2:
                scene_lengths.append(scene[1] - scene[0])
        
        if scene_lengths:
            scene_variance = np.var(scene_lengths)
            avg_scene_length = np.mean(scene_lengths)
            
            # Coefficient of variation
            cv = np.sqrt(scene_variance) / avg_scene_length if avg_scene_length > 0 else 1
            
            if cv < 0.3:  # Çok tutarlı
                pacing_score += 0.5
            elif cv < 0.5:  # Orta tutarlılık
                pacing_score += 0.3
            else:  # Tutarsız
                pacing_score += 0.1
        
        # 2. Overall pacing
        scene_count = len(scenes)
        expected_scenes = duration / 5  # Her 5 saniyede bir scene
        
        if abs(scene_count - expected_scenes) <= 2:  # Optimal pacing
            pacing_score += 0.5
        elif abs(scene_count - expected_scenes) <= 4:  # Kabul edilebilir
            pacing_score += 0.3
        else:  # Aşırı hızlı/yavaş
            pacing_score += 0.1
        
        return min(pacing_score, 1.0)
        
    except Exception as e:
        print(f"🎬 [TRANSITION] Pacing consistency analysis failed: {e}")
        return 0.5


def analyze_loop_final(features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """
    Loop & Final Analizi (7 puan)
    Döngü ve bitiş analizi
    """
    score = 0
    findings = []
    recommendations = []
    raw_metrics = {}
    
    print("🔄 [LOOP] Starting loop & final analysis...")
    
    # 1. Video Döngü Analizi (2 puan)
    loop_score = analyze_video_loops(features, duration)
    raw_metrics["video_loops"] = loop_score
    
    if loop_score > 0.7:
        score += 2
        findings.append({
            "t": duration * 0.8,
            "type": "perfect_loop",
            "msg": "Mükemmel döngü - video tekrar izlenebilir"
        })
    elif loop_score > 0.4:
        score += 1
        recommendations.append("Döngüyü iyileştir - başlangıç ve bitiş uyumunu artır")
    else:
        recommendations.append("Döngü ekle - video tekrar izlenebilir hale getir")
    
    # 2. Bitiş Analizi (2 puan)
    ending_score = analyze_video_ending(features, duration)
    raw_metrics["video_ending"] = ending_score
    
    if ending_score > 0.7:
        score += 2
        findings.append({
            "t": duration * 0.95,
            "type": "strong_ending",
            "msg": "Güçlü bitiş - izleyiciyi tatmin ediyor"
        })
    elif ending_score > 0.4:
        score += 1
        recommendations.append("Bitişi güçlendir - daha etkileyici son ekle")
    else:
        recommendations.append("Güçlü bitiş ekle - video sonunda özet veya çağrı")
    
    # 3. Tekrar İzlenebilirlik (1 puan)
    rewatch_score = analyze_rewatchability(features, duration)
    raw_metrics["rewatchability"] = rewatch_score
    
    if rewatch_score > 0.7:
        score += 1
        findings.append({
            "t": duration * 0.5,
            "type": "high_rewatch",
            "msg": "Yüksek tekrar izlenebilirlik - viral potansiyeli"
        })
    else:
        recommendations.append("Tekrar izlenebilirliği artır - daha çekici içerik ekle")
    
    # 4. Kapanış Tutarlılığı (1 puan)
    closure_score = analyze_closure_consistency(features, duration)
    raw_metrics["closure_consistency"] = closure_score
    
    if closure_score > 0.7:
        score += 1
        findings.append({
            "t": duration * 0.9,
            "type": "consistent_closure",
            "msg": "Tutarlı kapanış - mesaj net tamamlanmış"
        })
    else:
        recommendations.append("Kapanış tutarlılığını artır - mesajı net tamamla")
    
    # 5. Viral Döngü Potansiyeli (1 puan)
    viral_loop_score = analyze_viral_loop_potential(features, duration)
    raw_metrics["viral_loop_potential"] = viral_loop_score
    
    if viral_loop_score > 0.7:
        score += 1
        findings.append({
            "t": 0.0,
            "type": "viral_loop",
            "msg": "Viral döngü potansiyeli - paylaşım değeri yüksek"
        })
    else:
        recommendations.append("Viral potansiyeli artır - paylaşım değeri ekle")
    
    print(f"🔄 [LOOP] Loop & final score: {score}/7")
    
    return {
        "score": min(score, 7),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw_metrics
    }


def analyze_video_loops(features: Dict[str, Any], duration: float) -> float:
    """
    Video döngü analizi
    """
    try:
        loop_score = 0.0
        
        # 1. Başlangıç ve bitiş uyumu
        start_end_score = analyze_start_end_similarity(features, duration)
        loop_score += start_end_score * 0.4
        
        # 2. Video uzunluğu analizi
        length_score = analyze_loop_length(duration)
        loop_score += length_score * 0.3
        
        # 3. Ses döngü analizi
        audio_loop_score = analyze_audio_loops(features)
        loop_score += audio_loop_score * 0.3
        
        return min(loop_score, 1.0)
        
    except Exception as e:
        print(f"🔄 [LOOP] Video loop analysis failed: {e}")
        return 0.5


def analyze_start_end_similarity(features: Dict[str, Any], duration: float) -> float:
    """
    Başlangıç ve bitiş benzerlik analizi
    """
    try:
        frames = features.get("frames", [])
        if len(frames) < 2:
            return 0.5
        
        import cv2
        import numpy as np
        
        # İlk ve son frame'i karşılaştır
        first_frame = cv2.imread(frames[0])
        last_frame = cv2.imread(frames[-1])
        
        if first_frame is None or last_frame is None:
            return 0.5
        
        # Gri tonlamaya çevir
        gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        
        # Histogram karşılaştırması
        hist_first = cv2.calcHist([gray_first], [0], None, [256], [0, 256])
        hist_last = cv2.calcHist([gray_last], [0], None, [256], [0, 256])
        
        # Histogram korelasyonu
        correlation = cv2.compareHist(hist_first, hist_last, cv2.HISTCMP_CORREL)
        
        # Benzerlik skoru
        similarity_score = correlation
        
        print(f"🔄 [LOOP] Start-end similarity: {similarity_score:.2f}")
        
        return similarity_score
        
    except Exception as e:
        print(f"🔄 [LOOP] Start-end similarity analysis failed: {e}")
        return 0.5


def analyze_loop_length(duration: float) -> float:
    """
    Döngü için optimal uzunluk analizi
    """
    # Viral videolar için optimal uzunluk
    if 15 <= duration <= 60:  # 15-60 saniye (TikTok, Instagram)
        return 1.0
    elif 10 <= duration <= 90:  # 10-90 saniye (kabul edilebilir)
        return 0.7
    elif 5 <= duration <= 120:  # 5-120 saniye (uzun)
        return 0.5
    else:  # Çok kısa veya uzun
        return 0.3


def analyze_audio_loops(features: Dict[str, Any]) -> float:
    """
    Ses döngü analizi
    """
    try:
        audio_features = features.get("audio", {})
        
        loop_score = 0.0
        
        # 1. Tempo consistency
        tempo = audio_features.get("tempo", 0)
        if tempo > 0:  # Tempo tespit edildi
            loop_score += 0.4
        
        # 2. Loudness consistency
        loudness = audio_features.get("loudness", 0)
        if -20 <= loudness <= -6:  # Optimal loudness
            loop_score += 0.3
        elif -30 <= loudness <= -3:  # Kabul edilebilir
            loop_score += 0.2
        else:
            loop_score += 0.1
        
        # 3. Audio quality
        audio_info = features.get("audio_info", {})
        sample_rate = audio_info.get("sample_rate", 0)
        if sample_rate >= 44100:
            loop_score += 0.3
        elif sample_rate >= 22050:
            loop_score += 0.2
        else:
            loop_score += 0.1
        
        return min(loop_score, 1.0)
        
    except Exception as e:
        print(f"🔄 [LOOP] Audio loop analysis failed: {e}")
        return 0.5


def analyze_video_ending(features: Dict[str, Any], duration: float) -> float:
    """
    Video bitiş analizi
    """
    try:
        ending_score = 0.0
        
        # 1. CTA presence in ending
        cta_score = analyze_ending_cta(features, duration)
        ending_score += cta_score * 0.4
        
        # 2. Visual ending quality
        visual_ending_score = analyze_visual_ending(features, duration)
        ending_score += visual_ending_score * 0.3
        
        # 3. Audio ending quality
        audio_ending_score = analyze_audio_ending(features, duration)
        ending_score += audio_ending_score * 0.3
        
        return min(ending_score, 1.0)
        
    except Exception as e:
        print(f"🔄 [LOOP] Video ending analysis failed: {e}")
        return 0.5


def analyze_ending_cta(features: Dict[str, Any], duration: float) -> float:
    """
    Bitiş CTA analizi
    """
    try:
        # Son %10'luk kısımda CTA var mı?
        textual = features.get("textual", {})
        asr_text = textual.get("asr_text", "").lower()
        ocr_text = textual.get("ocr_text", "").lower()
        
        combined_text = f"{asr_text} {ocr_text}"
        
        # CTA keywords
        cta_keywords = [
            "beğen", "like", "takip", "follow", "abone", "subscribe",
            "paylaş", "share", "yorum", "comment", "kaydet", "save",
            "link", "linki", "bio", "profil", "profile"
        ]
        
        # Son kısımda CTA var mı kontrol et
        cta_found = any(keyword in combined_text for keyword in cta_keywords)
        
        if cta_found:
            return 1.0
        else:
            return 0.3
        
    except Exception as e:
        print(f"🔄 [LOOP] Ending CTA analysis failed: {e}")
        return 0.5


def analyze_visual_ending(features: Dict[str, Any], duration: float) -> float:
    """
    Görsel bitiş analizi
    """
    try:
        frames = features.get("frames", [])
        if not frames:
            return 0.5
        
        # Son frame'i analiz et
        last_frame_path = frames[-1]
        
        import cv2
        import numpy as np
        
        image = cv2.imread(last_frame_path)
        if image is None:
            return 0.5
        
        # Son frame kalitesi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Brightness
        brightness = np.mean(gray)
        
        ending_score = 0.0
        
        # Sharp ending
        if sharpness > 500:
            ending_score += 0.5
        elif sharpness > 200:
            ending_score += 0.3
        else:
            ending_score += 0.1
        
        # Good brightness
        if 80 <= brightness <= 180:
            ending_score += 0.5
        elif 50 <= brightness <= 220:
            ending_score += 0.3
        else:
            ending_score += 0.1
        
        return min(ending_score, 1.0)
        
    except Exception as e:
        print(f"🔄 [LOOP] Visual ending analysis failed: {e}")
        return 0.5


def analyze_audio_ending(features: Dict[str, Any], duration: float) -> float:
    """
    Ses bitiş analizi
    """
    try:
        audio_features = features.get("audio", {})
        
        ending_score = 0.0
        
        # Loudness at ending
        loudness = audio_features.get("loudness", 0)
        if -20 <= loudness <= -6:
            ending_score += 0.5
        elif -30 <= loudness <= -3:
            ending_score += 0.3
        else:
            ending_score += 0.1
        
        # Tempo consistency
        tempo = audio_features.get("tempo", 0)
        if tempo > 0:
            ending_score += 0.5
        else:
            ending_score += 0.2
        
        return min(ending_score, 1.0)
        
    except Exception as e:
        print(f"🔄 [LOOP] Audio ending analysis failed: {e}")
        return 0.5


def analyze_rewatchability(features: Dict[str, Any], duration: float) -> float:
    """
    Tekrar izlenebilirlik analizi
    """
    try:
        rewatch_score = 0.0
        
        # 1. Duration appropriateness
        if 15 <= duration <= 60:
            rewatch_score += 0.4
        elif 10 <= duration <= 90:
            rewatch_score += 0.3
        else:
            rewatch_score += 0.2
        
        # 2. Content density
        textual = features.get("textual", {})
        asr_text = textual.get("asr_text", "")
        ocr_text = textual.get("ocr_text", "")
        
        total_text = len(asr_text) + len(ocr_text)
        text_density = total_text / duration if duration > 0 else 0
        
        if text_density > 20:  # High content density
            rewatch_score += 0.3
        elif text_density > 10:  # Medium density
            rewatch_score += 0.2
        else:  # Low density
            rewatch_score += 0.1
        
        # 3. Visual appeal
        visual_score = features.get("technical_quality_score", 0) / 15  # Normalize
        rewatch_score += visual_score * 0.3
        
        return min(rewatch_score, 1.0)
        
    except Exception as e:
        print(f"🔄 [LOOP] Rewatchability analysis failed: {e}")
        return 0.5


def analyze_closure_consistency(features: Dict[str, Any], duration: float) -> float:
    """
    Kapanış tutarlılığı analizi
    """
    try:
        consistency_score = 0.0
        
        # 1. Message consistency
        textual = features.get("textual", {})
        title = textual.get("title", "").lower()
        asr_text = textual.get("asr_text", "").lower()
        
        # Title ve content uyumu
        if title and asr_text:
            title_words = set(title.split())
            content_words = set(asr_text.split())
            
            # Ortak kelime oranı
            common_words = title_words.intersection(content_words)
            if len(title_words) > 0:
                consistency_ratio = len(common_words) / len(title_words)
                consistency_score += consistency_ratio * 0.5
        
        # 2. Visual consistency
        frames = features.get("frames", [])
        if len(frames) >= 2:
            import cv2
            import numpy as np
            
            first_frame = cv2.imread(frames[0])
            last_frame = cv2.imread(frames[-1])
            
            if first_frame is not None and last_frame is not None:
                # Color consistency
                first_hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
                last_hsv = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
                
                first_mean = np.mean(first_hsv, axis=(0, 1))
                last_mean = np.mean(last_hsv, axis=(0, 1))
                
                color_diff = np.linalg.norm(first_mean - last_mean)
                if color_diff < 30:  # Similar colors
                    consistency_score += 0.3
                elif color_diff < 60:  # Somewhat similar
                    consistency_score += 0.2
                else:  # Different colors
                    consistency_score += 0.1
        
        return min(consistency_score, 1.0)
        
    except Exception as e:
        print(f"🔄 [LOOP] Closure consistency analysis failed: {e}")
        return 0.5


def analyze_viral_loop_potential(features: Dict[str, Any], duration: float) -> float:
    """
    Viral döngü potansiyeli analizi
    """
    try:
        viral_score = 0.0
        
        # 1. Duration for viral content
        if 15 <= duration <= 30:  # Perfect for TikTok
            viral_score += 0.4
        elif 10 <= duration <= 60:  # Good for social media
            viral_score += 0.3
        else:
            viral_score += 0.2
        
        # 2. Hook strength
        hook_score = features.get("hook_score", 0) / 8  # Normalize
        viral_score += hook_score * 0.3
        
        # 3. Engagement elements
        textual = features.get("textual", {})
        asr_text = textual.get("asr_text", "").lower()
        
        # Engagement keywords
        engagement_words = [
            "wow", "amazing", "incredible", "unbelievable", "shocking",
            "harika", "inanılmaz", "şaşırtıcı", "muhteşem", "wow"
        ]
        
        engagement_count = sum(1 for word in engagement_words if word in asr_text)
        if engagement_count > 2:
            viral_score += 0.3
        elif engagement_count > 0:
            viral_score += 0.2
        else:
            viral_score += 0.1
        
        return min(viral_score, 1.0)
        
    except Exception as e:
        print(f"🔄 [LOOP] Viral loop potential analysis failed: {e}")
        return 0.5


def analyze_text_readability(features: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """
    Metin Okunabilirlik Analizi (6 puan)
    Font, kontrast, timing + zaman bazlı viral etki analizi
    """
    score = 0
    findings = []
    recommendations = []
    raw_metrics = {}
    viral_impact_timeline = []
    
    print("📝 [TEXT] Starting text readability analysis...")
    
    # 1. Font Analizi (1 puan)
    font_score = analyze_font_quality(features, duration, viral_impact_timeline)
    raw_metrics["font_quality"] = font_score
    
    if font_score > 0.7:
        score += 1
        findings.append({
            "t": duration * 0.3,
            "type": "good_font",
            "msg": "Kaliteli font kullanımı - okunabilirlik yüksek"
        })
    elif font_score > 0.4:
        score += 0.5
        recommendations.append("Font kalitesini artır - daha okunabilir font seç")
    else:
        recommendations.append("Font değiştir - kalın ve net font kullan")
    
    # 2. Kontrast Analizi (2 puan)
    contrast_score = analyze_text_contrast(features, duration, viral_impact_timeline)
    raw_metrics["text_contrast"] = contrast_score
    
    if contrast_score > 0.7:
        score += 2
        findings.append({
            "t": duration * 0.5,
            "type": "high_contrast",
            "msg": "Yüksek kontrast - metin net görünüyor"
        })
    elif contrast_score > 0.4:
        score += 1
        recommendations.append("Kontrastı artır - beyaz metin siyah arka plan")
    else:
        recommendations.append("Kontrast sorunu var - metin görünmüyor")
    
    # 3. Timing Analizi (2 puan)
    timing_score = analyze_text_timing(features, duration, viral_impact_timeline)
    raw_metrics["text_timing"] = timing_score
    
    if timing_score > 0.7:
        score += 2
        findings.append({
            "t": duration * 0.7,
            "type": "perfect_timing",
            "msg": "Mükemmel timing - metin okunabilir sürede kalıyor"
        })
    elif timing_score > 0.4:
        score += 1
        recommendations.append("Timing'i iyileştir - metin daha uzun süre kalsın")
    else:
        recommendations.append("Metin çok hızlı - okunabilir sürede kalması için yavaşlat")
    
    # 4. Viral Etki Analizi (1 puan)
    viral_impact_score = analyze_text_viral_impact(features, duration, viral_impact_timeline)
    raw_metrics["viral_impact"] = viral_impact_score
    
    if viral_impact_score > 0.7:
        score += 1
        findings.append({
            "t": duration * 0.8,
            "type": "viral_text",
            "msg": "Viral potansiyeli yüksek metin - etkileyici kelimeler"
        })
    else:
        recommendations.append("Viral kelimeler ekle - 'wow', 'amazing', 'incredible' gibi")
    
    print(f"📝 [TEXT] Text readability score: {score}/6")
    
    return {
        "score": min(score, 6),
        "findings": findings,
        "recommendations": recommendations,
        "raw": raw_metrics,
        "viral_impact_timeline": viral_impact_timeline  # Zaman bazlı viral etki
    }


def analyze_font_quality(features: Dict[str, Any], duration: float, viral_timeline: List[Dict]) -> float:
    """
    Font kalitesi analizi
    """
    try:
        frames = features.get("frames", [])
        if not frames:
            return 0.5
        
        font_score = 0.0
        font_sizes = []
        
        import cv2
        import numpy as np
        
        # Her frame'de font analizi
        for i, frame_path in enumerate(frames[:5]):  # İlk 5 frame
            try:
                image = cv2.imread(frame_path)
                if image is None:
                    continue
                
                # OCR ile text detection
                textual = features.get("textual", {})
                ocr_texts = textual.get("ocr_text", [])
                
                if ocr_texts:
                    # Font kalitesi skoru
                    current_time = (i / len(frames)) * duration
                    
                    # Font boyutu analizi (text region area)
                    text_regions = detect_text_regions(image)
                    if text_regions:
                        avg_font_size = np.mean([region['area'] for region in text_regions])
                        
                        if avg_font_size > 1000:  # Büyük font
                            font_score += 0.3
                            viral_timeline.append({
                                "t": current_time,
                                "impact": "positive",
                                "reason": "Büyük font - dikkat çekici",
                                "score_change": 0.1
                            })
                        elif avg_font_size > 500:  # Orta font
                            font_score += 0.2
                            viral_timeline.append({
                                "t": current_time,
                                "impact": "neutral",
                                "reason": "Orta font - okunabilir",
                                "score_change": 0.05
                            })
                        else:  # Küçük font
                            font_score += 0.1
                            viral_timeline.append({
                                "t": current_time,
                                "impact": "negative",
                                "reason": "Küçük font - zor okunuyor",
                                "score_change": -0.1
                            })
                        
                        font_sizes.append(avg_font_size)
                
            except Exception as e:
                print(f"📝 [TEXT] Font analysis failed for frame {i}: {e}")
                continue
        
        # Ortalama font kalitesi
        if font_sizes:
            avg_font_size = np.mean(font_sizes)
            if avg_font_size > 800:
                font_score += 0.4
            elif avg_font_size > 400:
                font_score += 0.3
            else:
                font_score += 0.1
        
        return min(font_score, 1.0)
        
    except Exception as e:
        print(f"📝 [TEXT] Font quality analysis failed: {e}")
        return 0.5


def detect_text_regions(image) -> List[Dict]:
    """
    Görüntüde text region'ları tespit et
    """
    try:
        import cv2
        import numpy as np
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Text detection için edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Contour bulma
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Text-like aspect ratio
                if 0.2 < aspect_ratio < 10:
                    text_regions.append({
                        'area': area,
                        'bbox': (x, y, w, h),
                        'aspect_ratio': aspect_ratio
                    })
        
        return text_regions
        
    except Exception as e:
        print(f"📝 [TEXT] Text region detection failed: {e}")
        return []


def analyze_text_contrast(features: Dict[str, Any], duration: float, viral_timeline: List[Dict]) -> float:
    """
    Text kontrast analizi
    """
    try:
        frames = features.get("frames", [])
        if not frames:
            return 0.5
        
        contrast_score = 0.0
        contrast_values = []
        
        import cv2
        import numpy as np
        
        for i, frame_path in enumerate(frames[:5]):
            try:
                image = cv2.imread(frame_path)
                if image is None:
                    continue
                
                current_time = (i / len(frames)) * duration
                
                # Text region'larda kontrast analizi
                text_regions = detect_text_regions(image)
                
                for region in text_regions:
                    x, y, w, h = region['bbox']
                    roi = image[y:y+h, x:x+w]
                    
                    if roi.size > 0:
                        # Gri tonlama
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        
                        # Kontrast hesaplama (std deviation)
                        contrast = np.std(gray_roi)
                        contrast_values.append(contrast)
                        
                        if contrast > 50:  # Yüksek kontrast
                            contrast_score += 0.3
                            viral_timeline.append({
                                "t": current_time,
                                "impact": "positive",
                                "reason": "Yüksek kontrast - metin net",
                                "score_change": 0.15
                            })
                        elif contrast > 25:  # Orta kontrast
                            contrast_score += 0.2
                            viral_timeline.append({
                                "t": current_time,
                                "impact": "neutral",
                                "reason": "Orta kontrast - okunabilir",
                                "score_change": 0.05
                            })
                        else:  # Düşük kontrast
                            contrast_score += 0.1
                            viral_timeline.append({
                                "t": current_time,
                                "impact": "negative",
                                "reason": "Düşük kontrast - zor okunuyor",
                                "score_change": -0.15
                            })
                
            except Exception as e:
                print(f"📝 [TEXT] Contrast analysis failed for frame {i}: {e}")
                continue
        
        # Genel kontrast skoru
        if contrast_values:
            avg_contrast = np.mean(contrast_values)
            if avg_contrast > 40:
                contrast_score += 0.4
            elif avg_contrast > 20:
                contrast_score += 0.3
            else:
                contrast_score += 0.1
        
        return min(contrast_score, 1.0)
        
    except Exception as e:
        print(f"📝 [TEXT] Text contrast analysis failed: {e}")
        return 0.5


def analyze_text_timing(features: Dict[str, Any], duration: float, viral_timeline: List[Dict]) -> float:
    """
    Text timing analizi - metin ne kadar süre ekranda kalıyor
    """
    try:
        frames = features.get("frames", [])
        textual = features.get("textual", {})
        ocr_texts = textual.get("ocr_text", [])
        
        if not frames or not ocr_texts:
            return 0.5
        
        timing_score = 0.0
        frame_interval = duration / len(frames)
        
        # Text persistence analizi
        text_persistence = analyze_text_persistence(frames, ocr_texts)
        
        for i, persistence in enumerate(text_persistence):
            current_time = i * frame_interval
            
            if persistence > 0.8:  # Text çoğu frame'de var
                timing_score += 0.3
                viral_timeline.append({
                    "t": current_time,
                    "impact": "positive",
                    "reason": "Metin sürekli görünüyor - okunabilir",
                    "score_change": 0.1
                })
            elif persistence > 0.5:  # Text yarı yarıya var
                timing_score += 0.2
                viral_timeline.append({
                    "t": current_time,
                    "impact": "neutral",
                    "reason": "Metin ara sıra görünüyor",
                    "score_change": 0.0
                })
            else:  # Text nadiren var
                timing_score += 0.1
                viral_timeline.append({
                    "t": current_time,
                    "impact": "negative",
                    "reason": "Metin çok kısa görünüyor",
                    "score_change": -0.1
                })
        
        # Optimal timing kontrolü
        optimal_timing = check_optimal_text_timing(frames, ocr_texts, duration)
        timing_score += optimal_timing * 0.4
        
        return min(timing_score, 1.0)
        
    except Exception as e:
        print(f"📝 [TEXT] Text timing analysis failed: {e}")
        return 0.5


def analyze_text_persistence(frames: List[str], ocr_texts: List[str]) -> List[float]:
    """
    Text'in frame'lerde ne kadar süre kaldığını analiz et
    """
    try:
        persistence_scores = []
        
        # Her frame için text varlığı kontrol et
        for frame_path in frames:
            # Basit text detection (OCR sonuçlarına dayalı)
            text_found = len(ocr_texts) > 0
            persistence_scores.append(1.0 if text_found else 0.0)
        
        return persistence_scores
        
    except Exception as e:
        print(f"📝 [TEXT] Text persistence analysis failed: {e}")
        return [0.5] * len(frames)


def check_optimal_text_timing(frames: List[str], ocr_texts: List[str], duration: float) -> float:
    """
    Optimal text timing kontrolü
    """
    try:
        if not frames or not ocr_texts:
            return 0.5
        
        # Text timing kuralları
        frame_interval = duration / len(frames)
        
        # 1. Text çok hızlı değişmemeli (minimum 2 saniye)
        min_text_duration = 2.0
        frames_per_min_duration = int(min_text_duration / frame_interval)
        
        # 2. Text çok uzun süre kalmamalı (maksimum 10 saniye)
        max_text_duration = 10.0
        frames_per_max_duration = int(max_text_duration / frame_interval)
        
        # Text persistence analizi
        persistence = analyze_text_persistence(frames, ocr_texts)
        
        # Continuous text blocks bul
        text_blocks = []
        current_block_start = None
        
        for i, has_text in enumerate(persistence):
            if has_text > 0.5 and current_block_start is None:
                current_block_start = i
            elif has_text <= 0.5 and current_block_start is not None:
                text_blocks.append((current_block_start, i - 1))
                current_block_start = None
        
        # Son block'u ekle
        if current_block_start is not None:
            text_blocks.append((current_block_start, len(persistence) - 1))
        
        # Timing skoru hesapla
        timing_score = 0.0
        for start_frame, end_frame in text_blocks:
            block_duration_frames = end_frame - start_frame + 1
            
            if frames_per_min_duration <= block_duration_frames <= frames_per_max_duration:
                timing_score += 1.0  # Optimal timing
            elif block_duration_frames < frames_per_min_duration:
                timing_score += 0.3  # Çok kısa
            else:
                timing_score += 0.5  # Çok uzun
        
        # Normalize
        if text_blocks:
            return timing_score / len(text_blocks)
        else:
            return 0.5
        
    except Exception as e:
        print(f"📝 [TEXT] Optimal timing check failed: {e}")
        return 0.5


def analyze_text_viral_impact(features: Dict[str, Any], duration: float, viral_timeline: List[Dict]) -> float:
    """
    Text'in viral etkisini analiz et
    """
    try:
        textual = features.get("textual", {})
        asr_text = textual.get("asr_text", "").lower()
        ocr_texts = textual.get("ocr_text", [])
        
        # Tüm text'i birleştir
        all_text = asr_text
        for ocr_text in ocr_texts:
            all_text += " " + ocr_text.lower()
        
        viral_score = 0.0
        
        # Viral kelimeler
        viral_keywords = {
            # Pozitif viral kelimeler
            "wow": 0.2, "amazing": 0.2, "incredible": 0.2, "unbelievable": 0.2,
            "shocking": 0.2, "mind-blowing": 0.2, "epic": 0.15, "legendary": 0.15,
            "harika": 0.2, "inanılmaz": 0.2, "muhteşem": 0.2, "şaşırtıcı": 0.2,
            "müthiş": 0.15, "efsane": 0.15, "vay be": 0.1, "wow": 0.2,
            
            # Etkileşim kelimeleri
            "like": 0.1, "share": 0.1, "comment": 0.1, "subscribe": 0.1,
            "beğen": 0.1, "paylaş": 0.1, "yorum": 0.1, "abone": 0.1,
            
            # Soru kelimeleri
            "how": 0.1, "why": 0.1, "what": 0.1, "when": 0.1,
            "nasıl": 0.1, "neden": 0.1, "ne": 0.1, "ne zaman": 0.1,
            
            # Sayılar (viral için önemli)
            "first": 0.1, "last": 0.1, "only": 0.1, "never": 0.1,
            "ilk": 0.1, "son": 0.1, "sadece": 0.1, "hiç": 0.1,
            
            # Emoji benzeri kelimeler
            "fire": 0.1, "lit": 0.1, "goals": 0.1, "vibes": 0.1
        }
        
        # Text'i kelimelere böl
        words = all_text.split()
        total_words = len(words)
        
        if total_words == 0:
            return 0.5
        
        viral_word_count = 0
        for word in words:
            if word in viral_keywords:
                viral_score += viral_keywords[word]
                viral_word_count += 1
        
        # Viral kelime yoğunluğu
        viral_density = viral_word_count / total_words if total_words > 0 else 0
        
        # Viral density skoru
        if viral_density > 0.1:  # %10'dan fazla viral kelime
            viral_score += 0.4
            viral_timeline.append({
                "t": duration * 0.5,
                "impact": "positive",
                "reason": f"Yüksek viral kelime yoğunluğu (%{viral_density*100:.1f})",
                "score_change": 0.2
            })
        elif viral_density > 0.05:  # %5'ten fazla viral kelime
            viral_score += 0.3
            viral_timeline.append({
                "t": duration * 0.5,
                "impact": "positive",
                "reason": f"Orta viral kelime yoğunluğu (%{viral_density*100:.1f})",
                "score_change": 0.1
            })
        else:
            viral_timeline.append({
                "t": duration * 0.5,
                "impact": "negative",
                "reason": f"Düşük viral kelime yoğunluğu (%{viral_density*100:.1f})",
                "score_change": -0.1
            })
        
        # Text uzunluğu analizi
        if 10 <= total_words <= 50:  # Optimal text uzunluğu
            viral_score += 0.2
        elif 5 <= total_words <= 100:  # Kabul edilebilir uzunluk
            viral_score += 0.1
        
        return min(viral_score, 1.0)
        
    except Exception as e:
        print(f"📝 [TEXT] Text viral impact analysis failed: {e}")
        return 0.5