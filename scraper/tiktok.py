"""TikTok yorum scraping."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import httpx

from audience import CommentRecord, CommentThread

from monitoring import log_scraper_error, log_scraper_event
from .base import ScrapeError, ScraperConfig


SIGI_DATA_RE = re.compile(r"<script id=\"SIGI_STATE\"[^>]*>(?P<data>.*?)</script>")


class TikTokScraper:
    """TikTok video sayfalarndan yorum eker."""

    def __init__(self, config: Optional[ScraperConfig] = None) -> None:
        self.config = config or ScraperConfig()

    def _fetch_page(self, client: httpx.Client, video_url: str) -> str:
        resp = client.get(video_url)
        if resp.status_code == 429:
            raise ScrapeError("TikTok rate limit verdi (429). Session cookie gerekli")
        if resp.status_code != 200:
            raise ScrapeError(f"TikTok sayfa isteği başarısız: {resp.status_code}")
        return resp.text

    def _extract_state(self, html: str) -> Dict:
        match = SIGI_DATA_RE.search(html)
        if not match:
            raise ScrapeError("TikTok sayfasından SIGI_STATE JSON'u bulunamadı")
        data = match.group("data")
        return json.loads(data)

    def fetch_comments(
        self,
        video_url: str,
        *,
        max_comments: int = 200,
    ) -> List[CommentThread]:
        with self.config.client("tiktok") as client:
            try:
                page_html = self._fetch_page(client, video_url)
            except Exception as exc:
                log_scraper_error("tiktok", "fetch_page", str(exc), {"url": video_url})
                raise
            try:
                state = self._extract_state(page_html)
            except Exception as exc:
                log_scraper_error("tiktok", "extract_state", str(exc))
                raise

        comments_dict = state.get("Comment", {}).get("commentData", {})
        users_dict = state.get("UserModule", {}).get("users", {})
        bindings = state.get("Comment", {}).get("comments", {})

        threads: List[CommentThread] = []

        for comment_id, meta in comments_dict.items():
            parent = bindings.get(comment_id, {})
            if parent.get("commentType", 1) != 1:  # 1 => parent comment
                continue
            user_info = users_dict.get(str(meta.get("user_id")), {})
            parent_record = CommentRecord(
                comment_id=str(comment_id),
                author_id=str(meta.get("user_id")) if meta.get("user_id") else None,
                author_username=user_info.get("uniqueId"),
                author_avatar_url=user_info.get("avatarThumb"),
                text=parent.get("text", ""),
                language=parent.get("language"),
                created_at=str(parent.get("createTime")) if parent.get("createTime") else None,
                like_count=int(parent.get("diggCount", 0)),
                reply_count=int(parent.get("replyCommentTotal", 0)),
                is_pinned=bool(parent.get("isTop")),
            )

            replies: List[CommentRecord] = []
            for reply in parent.get("reply_comment_ids", []) or []:
                reply_meta = comments_dict.get(reply, {})
                reply_binding = bindings.get(reply, {})
                reply_user = users_dict.get(str(reply_meta.get("user_id")), {})
                replies.append(
                    CommentRecord(
                        comment_id=str(reply),
                        author_id=str(reply_meta.get("user_id")) if reply_meta.get("user_id") else None,
                        author_username=reply_user.get("uniqueId"),
                        author_avatar_url=reply_user.get("avatarThumb"),
                        text=reply_binding.get("text", ""),
                        language=reply_binding.get("language"),
                        created_at=str(reply_binding.get("createTime"))
                        if reply_binding.get("createTime")
                        else None,
                        like_count=int(reply_binding.get("diggCount", 0)),
                        reply_count=int(reply_binding.get("replyCommentTotal", 0)),
                        is_pinned=bool(reply_binding.get("isTop")),
                    )
                )

            threads.append(CommentThread(parent=parent_record, replies=replies))

            if len(threads) >= max_comments:
                break

        log_scraper_event(
            "info",
            "tiktok",
            "comments_scraped",
            {"count": len(threads), "url": video_url},
        )

        return threads

