import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any


def _smtp_client():
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASS", "")
    use_tls = os.getenv("SMTP_TLS", "true").lower() == "true"
    if not host or not user or not password:
        raise RuntimeError("SMTP yapılandırması eksik (SMTP_HOST/SMTP_USER/SMTP_PASS)")
    server = smtplib.SMTP(host, port, timeout=30)
    if use_tls:
        server.starttls()
    server.login(user, password)
    return server, user


def _format_summary_html(scores: Dict[str, float], why: str) -> str:
    rows = "".join(
        f"<tr><td style='padding:6px 12px;border:1px solid #eee'>{k}</td><td style='padding:6px 12px;border:1px solid #eee'>{v}</td></tr>"
        for k, v in scores.items()
    )
    return f"""
    <html>
      <body style='font-family:Inter,Arial,sans-serif'>
        <h2 style='margin-bottom:8px'>Video Analiz Özeti</h2>
        <p style='color:#444'>{why}</p>
        <h3 style='margin:16px 0 8px'>Skorlar</h3>
        <table style='border-collapse:collapse'>{rows}</table>
      </body>
    </html>
    """


def send_analysis_email(to_email: str, platform: str, scores: Dict[str, float], why: str, suggestions: Dict[str, Any]) -> None:
    subject = f"{platform.title()} video analizi hazır"
    html = _format_summary_html(scores, why)
    tips = suggestions.get("tips", [])
    hooks = suggestions.get("alternative_hooks", [])
    hashtags = suggestions.get("hashtags", [])
    extra = ""
    if tips:
        extra += "<h3 style='margin:16px 0 8px'>Öneriler</h3><ul>" + "".join(
            f"<li>{t}</li>" for t in tips[:6]
        ) + "</ul>"
    if hooks:
        extra += "<h3 style='margin:16px 0 8px'>Alternatif Hook'lar</h3><ul>" + "".join(
            f"<li>{h}</li>" for h in hooks[:3]
        ) + "</ul>"
    if hashtags:
        extra += "<p><strong>Hashtag önerileri:</strong> " + ", ".join(f"#{h}" for h in hashtags[:8]) + "</p>"
    html = html.replace("</body>", extra + "</body>")

    server, from_user = _smtp_client()
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_user
        msg["To"] = to_email
        part = MIMEText(html, "html", _charset="utf-8")
        msg.attach(part)
        server.sendmail(from_user, [to_email], msg.as_string())
    finally:
        try:
            server.quit()
        except Exception:
            pass


