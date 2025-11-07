"""Pydantic modelleri ve veri yapıları (yorum + etkileşim)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class CommentRecord(BaseModel):
    """Tek bir yoruma ait temel veri."""

    comment_id: str = Field(..., description="Platform tarafından verilen benzersiz ID")
    author_id: Optional[str] = Field(
        None, description="Yorumu yapan hesabın ID'si (erişilebiliyorsa)"
    )
    author_username: Optional[str] = Field(None, description="Kullanıcı adı")
    author_avatar_url: Optional[HttpUrl] = Field(
        None, description="Kullanıcı profil fotoğrafı"
    )
    text: str = Field(..., description="Yorum içeriği")
    language: Optional[str] = Field(
        None, description="ISO 639-1 dil kodu (algılanmış veya orijinal)"
    )
    created_at: Optional[str] = Field(
        None, description="ISO8601 oluşturulma zamanı (platform saati)"
    )
    like_count: Optional[int] = Field(0, ge=0, description="Yorum beğeni sayısı")
    reply_count: Optional[int] = Field(0, ge=0, description="Yanıt sayısı")
    is_pinned: bool = Field(False, description="Yorum sabitlenmiş mi")
    media_urls: Optional[List[HttpUrl]] = Field(
        default=None, description="Yoruma eklenen medya (görüntü/gif vb.)"
    )


class CommentThread(BaseModel):
    """Bir yorum ve varsa yanıtları."""

    parent: CommentRecord
    replies: List[CommentRecord] = Field(default_factory=list)


class AudienceEngagementMetrics(BaseModel):
    """Kullanıcının manuel sağlayacağı etkileşim metrikleri."""

    impressions: Optional[int] = Field(None, ge=0)
    reach: Optional[int] = Field(None, ge=0)
    views: Optional[int] = Field(None, ge=0)
    plays: Optional[int] = Field(None, ge=0)
    likes: Optional[int] = Field(None, ge=0)
    comments: Optional[int] = Field(None, ge=0)
    shares: Optional[int] = Field(None, ge=0)
    saves: Optional[int] = Field(None, ge=0)
    reposts: Optional[int] = Field(None, ge=0)
    profile_visits: Optional[int] = Field(None, ge=0)
    follows: Optional[int] = Field(None, ge=0)
    click_through: Optional[int] = Field(None, ge=0, description="Link tıklamaları")
    avg_watch_time_s: Optional[float] = Field(None, ge=0)
    completion_rate: Optional[float] = Field(None, ge=0, le=1)


class AudienceSentimentSummary(BaseModel):
    """Yorumlardan türetilen duygu metriği özetleri."""

    positive_ratio: float = Field(..., ge=0, le=1)
    negative_ratio: float = Field(..., ge=0, le=1)
    neutral_ratio: float = Field(..., ge=0, le=1)
    dominant_emotions: List[str] = Field(
        default_factory=list, description="En baskın duygular (örn. joy, anger)"
    )
    keywords: List[str] = Field(
        default_factory=list, description="Yorumlarda öne çıkan kelimeler"
    )
    sample_quotes_positive: List[str] = Field(default_factory=list)
    sample_quotes_negative: List[str] = Field(default_factory=list)
    sample_quotes_neutral: List[str] = Field(default_factory=list)
    total_comments: int = Field(..., ge=0)
    analyzed_comments: int = Field(..., ge=0)


