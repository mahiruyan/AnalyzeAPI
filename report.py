from io import BytesIO
from typing import Dict, Any
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


def build_pdf(scores: Dict[str, float], why: str, suggestions: Dict[str, Any]) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 2 * cm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, "Video Analiz Raporu")
    y -= 1 * cm

    c.setFont("Helvetica", 12)
    c.drawString(2 * cm, y, "Özet:")
    y -= 0.6 * cm
    for line in _wrap_text(why, 90):
        c.drawString(2.5 * cm, y, line)
        y -= 0.5 * cm

    y -= 0.4 * cm
    c.setFont("Helvetica", 12)
    c.drawString(2 * cm, y, "Skorlar:")
    y -= 0.6 * cm
    for k, v in scores.items():
        c.drawString(2.5 * cm, y, f"{k}: {v}")
        y -= 0.5 * cm

    tips = suggestions.get("tips", [])
    if tips:
        y -= 0.4 * cm
        c.setFont("Helvetica", 12)
        c.drawString(2 * cm, y, "Öneriler:")
        y -= 0.6 * cm
        for t in tips[:6]:
            for line in _wrap_text(f"- {t}", 90):
                c.drawString(2.5 * cm, y, line)
                y -= 0.5 * cm

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


def _wrap_text(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width=width)


