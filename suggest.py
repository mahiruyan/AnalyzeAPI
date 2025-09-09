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

    if scores.get("hook_score", 0.0) < 0.6:
        suggestions.append("İlk 3 saniyede merak uyandıran bir soru veya vaat ekleyin.")
    if scores.get("flow_score", 0.0) < 0.6:
        suggestions.append("Kısa kesmeler ve ritmik geçişlerle tempoyu artırın.")
    if scores.get("audio_quality_score", 0.0) < 0.6:
        suggestions.append("Arka plan gürültüsünü azaltın, ses seviyesini dengeli tutun.")
    if scores.get("content_fit_score", 0.0) < 0.6:
        suggestions.append("Metinde etiketlerinizle uyumlu anahtar kelimeleri öne çıkarın.")

    # Alternatif kısa kancalar
    base_hook = "Herkesin kaçırdığı şu detaya bak!" if not title else f"{title.strip()} — Peki ya şu?"
    alt_hooks = [
        base_hook,
        "Bunu öğrenmeden paylaşma!",
        "3 saniyede fikrini değiştirecek bilgi!",
    ]

    # Hashtag önerileri
    tags_lower = [t.lower() for t in tags]
    base_tags = list({*(tags_lower), "viral", "trending", platform.lower()[:10]})

    return {
        "tips": suggestions,
        "alternative_hooks": alt_hooks,
        "hashtags": base_tags[:8],
        "captions": [caption or "Kısa ve net bir mesajla başla."],
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


