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
    
    # Gerek video verilerini al
    audio = features.get("audio", {})
    visual = features.get("visual", {})
    textual = features.get("textual", {})
    duration = features.get("duration_seconds", 0)
    
    loudness = audio.get("loudness_lufs", 0)
    tempo = audio.get("tempo_bpm", 0)
    flow = visual.get("optical_flow_mean", 0)
    ocr_text = textual.get("ocr_text", "")
    asr_text = textual.get("asr_text", "")
    
    # BPM analizi - doal konuma tarznda
    if tempo > 0:
        if tempo < 60:
            suggestions.append(f"Mziiniz {tempo:.0f} BPM'de ok sakin. Biraz daha hareketli bir ark seseniz izleyiciler daha ok dikkat eder.")
        elif tempo < 80:
            suggestions.append(f"{tempo:.0f} BPM gzel bir tempo ama viral olmak iin 100-120 BPM aras daha etkili. Biraz daha hzl mzik deneyin.")
        elif tempo < 100:
            suggestions.append(f"{tempo:.0f} BPM mkemmel! Bu tempoyu kesinlikle koruyun, ok dengeli.")
        elif tempo < 140:
            suggestions.append(f"{tempo:.0f} BPM ok enerjik! Bu hz viral ierik iin harika, ok dikkat ekici.")
        else:
            suggestions.append(f"{tempo:.0f} BPM biraz fazla hzl olmu. Baz izleyiciler rahatsz olabilir, biraz yavalatn.")
    
    # Ses seviyesi analizi - doal konuma tarznda
    if loudness != 0:
        if loudness < -25:
            suggestions.append(f" Sesiniz ok ksk ({loudness:.1f} LUFS). Biraz daha ykseltirseniz dinleyiciler daha rahat duyar.")
        elif loudness < -18:
            suggestions.append(f" Ses seviyeniz biraz dk ({loudness:.1f} LUFS). Biraz daha asanz ok daha iyi olur.")
        elif loudness < -12:
            suggestions.append(f" Ses seviyeniz harika ({loudness:.1f} LUFS)! Tam ideal seviyede, byle devam edin.")
        elif loudness < -8:
            suggestions.append(f" Ses biraz yksek ({loudness:.1f} LUFS). Hafife ksarsanz daha rahat dinlenir.")
        else:
            suggestions.append(f" Ses ok yksek ({loudness:.1f} LUFS). Biraz ksn, dinleyiciler rahatsz olabilir.")
    
    # Grsel hareket analizi - doal konuma tarznda
    if flow > 0:
        if flow < 0.3:
            suggestions.append(f" Videonuz ok statik grnyor. Biraz daha kamera hareketi veya zoom ekleseniz daha canl olur.")
        elif flow < 0.8:
            suggestions.append(f" Hareket seviyeniz orta ({flow:.2f}). Biraz daha dinamik ekimler deneyin, daha dikkat ekici olur.")
        elif flow < 2.0:
            suggestions.append(f" Hareket dengeniz harika ({flow:.2f})! Bu seviyeyi kesinlikle koruyun, ok profesyonel.")
        elif flow < 3.5:
            suggestions.append(f" ok youn hareket var ({flow:.2f}). Baz sahneleri biraz yavalatrsanz daha rahat izlenir.")
        else:
            suggestions.append(f" Geiler ok hzl ({flow:.2f}). Biraz daha yava edit yaparsanz izleyiciler takip edebilir.")
    
    # Metin tespiti - doal konuma tarznda
    if len(ocr_text) > 0:
        suggestions.append(f"Videonuzda '{ocr_text[:50]}...' yazs var. Bu gzel bir balang, yazlar ok etkili!")
        if len(ocr_text) < 30:
            suggestions.append("Biraz daha yaz ekleseniz harika olur. Metinler viral ierik iin sper etkili.")
    else:
        suggestions.append("Videonuzda hi yaz gremiyorum. Anahtar kelimeleri ekranda gsterirseniz ok daha etkili olur.")
    
    # Konuma analizi - doal konuma tarznda
    if len(asr_text) > 0:
        suggestions.append(f" Konumanz duyuyorum: '{asr_text[:40]}...' - Bu ok gzel!")
        if len(asr_text) < 50:
            suggestions.append(" Biraz daha uzun konusanz harika olur. Detayl aklamalar viral iin ok etkili.")
    else:
        suggestions.append(" Videonuzda konuma duymuyorum. Sesli aklama eklerseniz ok daha etkileimli olur.")
    
    # Sre analizi - doal konuma tarznda
    if duration > 0:
        suggestions.append(f"Videonuz {duration:.1f} saniye sryor.")
        if platform.lower() == "instagram":
            if duration < 20:
                suggestions.append("Instagram Reels iin biraz ksa. En az 30 saniye yaparsanz daha etkili olur.")
            elif duration > 90:
                suggestions.append("Instagram iin biraz uzun olmu. 60 saniyeden ksa tutarsanz daha iyi.")
            else:
                suggestions.append("Sre Instagram iin mkemmel! Bu uzunlukta devam edin.")
    
    # Skorlara gre spesifik neriler - doal konuma tarznda
    hook_score = scores.get("hook_score", 0)
    flow_score = scores.get("flow_score", 0)
    audio_score = scores.get("audio_quality_score", 0)
    
    if hook_score < 0.5:
        suggestions.append("lk 3 saniyeniz biraz snk. Daha gl bir al yaparsanz ok daha dikkat eker.")
    elif hook_score < 0.7:
        suggestions.append("Alnz orta seviyede. Biraz daha dikkat ekici bir balang denerseniz harika olur.")
    
    if flow_score < 0.5:
        suggestions.append("Video aknz biraz yava. Daha hzl edit yaparsanz daha canl olur.")
    elif flow_score < 0.7:
        suggestions.append("Aknz gzel ama biraz daha ritmik olabilir. Biraz daha hzl geiler deneyin.")
    
    if audio_score < 0.5:
        suggestions.append(" Ses kaliteniz biraz dk. Daha temiz bir kayt yaparsanz ok daha profesyonel olur.")
    elif audio_score < 0.7:
        suggestions.append(" Ses kaliteniz orta seviyede. Biraz daha iyiletirirseniz mkemmel olur.")
    
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


# Mock OpenAI fonksiyonu kaldrld - sadece gerek analiz kullanlyor


def generate_suggestions(
    platform: str,
    caption: str,
    title: str,
    tags: List[str],
    features: Dict[str, Any],
    scores: Dict[str, float],
    fast_mode: bool = False,
) -> Dict[str, Any]:
    # Sadece gerek analiz kullan - mock OpenAI fonksiyonunu kullanma
    return _rule_based_suggestions(platform, caption, title, tags, features, scores)


