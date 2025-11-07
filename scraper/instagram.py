"""Instagram yorum scraping."""

from __future__ import annotations

import re
from typing import Dict, Iterator, List, Optional

import httpx

from audience import CommentRecord, CommentThread

from monitoring import log_scraper_error, log_scraper_event
from .base import ScrapeError, ScraperConfig

GRAPHQL_COMMENTS_QUERY_HASH = "97b41c52301f77ce508f55e66d17620e"


class InstagramScraper:
    """Instagram sayfalarndan yorum ekmek iin yardmc14."""

    def __init__(self, config: Optional[ScraperConfig] = None) -> None:
        self.config = config or ScraperConfig()

    def _resolve_shortcode(self, media_url: str) -> str:
        match = re.search(r"/(p|reel|tv)/([\w-]+)/?", media_url)
        if not match:
            raise ScrapeError("Instagram URL'sinden shortcode krilemedi")
        return match.group(2)

    def _initial_request(self, client: httpx.Client, shortcode: str) -> Dict:
        url = f"https://www.instagram.com/p/{shortcode}/?__a=1&__d=dis"
        resp = client.get(url)
        if resp.status_code == 429:
            raise ScrapeError("Instagram 429 verdi - rate limit. Daha fazla cookie/başlık gerekiyor")
        if resp.status_code != 200:
            raise ScrapeError(f"Instagram medya isteği başarıs5z: {resp.status_code}")
        return resp.json()

    def _pagination_request(
        self,
        client: httpx.Client,
        shortcode: str,
        after: str,
        batch_size: int,
    ) -> Dict:
        params = {
            "query_hash": GRAPHQL_COMMENTS_QUERY_HASH,
            "variables": {
                "shortcode": shortcode,
                "first": batch_size,
                "after": after,
            },
        }
        resp = client.get("https://www.instagram.com/graphql/query/", params=params)
        if resp.status_code != 200:
            raise ScrapeError(f"Instagram yorum isteği başarısız: {resp.status_code}")
        return resp.json()

    def fetch_comments(
        self,
        media_url: str,
        *,
        max_comments: int = 200,
        include_replies: bool = True,
    ) -> List[CommentThread]:
        """Verilen medya URL'i için yorumları çeker."""

        shortcode = self._resolve_shortcode(media_url)
        with self.config.client("instagram") as client:
            try:
                data = self._initial_request(client, shortcode)
            except Exception as exc:
                log_scraper_error("instagram", "initial_request", str(exc))
                raise

            edges = (
                data.get("graphql", {})
                .get("shortcode_media", {})
                .get("edge_media_to_parent_comment", {})
                .get("edges", [])
            )
            page_info = (
                data.get("graphql", {})
                .get("shortcode_media", {})
                .get("edge_media_to_parent_comment", {})
                .get("page_info", {})
            )

            comments: List[CommentThread] = []

            def convert_node(node: Dict) -> CommentRecord:
                owner = node.get("owner", {})
                return CommentRecord(
                    comment_id=node.get("id", ""),
                    author_id=owner.get("id"),
                    author_username=owner.get("username"),
                    author_avatar_url=None,
                    text=node.get("text", ""),
                    language=node.get("did_report_as_spam", None),
                    created_at=node.get("created_at"),
                    like_count=node.get("edge_liked_by", {}).get("count", 0),
                    reply_count=node.get("edge_threaded_comments", {})
                    .get("count", 0),
                    is_pinned=node.get("did_report_as_spam", False) is False
                    and node.get("is_ranked_comment", False),
                )

            def convert_thread(edge: Dict) -> CommentThread:
                node = edge.get("node", {})
                parent = convert_node(node)
                replies_iter: Iterator[CommentRecord] = []
                if include_replies:
                    reply_edges = (
                        node.get("edge_threaded_comments", {}).get("edges", [])
                    )
                    replies_iter = (convert_node(reply.get("node", {})) for reply in reply_edges)
                return CommentThread(parent=parent, replies=list(replies_iter))

            for edge in edges:
                if len(comments) >= max_comments:
                    break
                comments.append(convert_thread(edge))

            end_cursor = page_info.get("end_cursor")
            has_next_page = page_info.get("has_next_page", False)

            while has_next_page and len(comments) < max_comments:
                remaining = max_comments - len(comments)
                batch = min(50, remaining)
                try:
                    json_data = self._pagination_request(client, shortcode, end_cursor, batch)
                except Exception as exc:
                    log_scraper_error("instagram", "pagination", str(exc), {"cursor": end_cursor})
                    break
                comment_edges = (
                    json_data.get("data", {})
                    .get("shortcode_media", {})
                    .get("edge_media_to_parent_comment", {})
                    .get("edges", [])
                )
                for edge in comment_edges:
                    if len(comments) >= max_comments:
                        break
                    comments.append(convert_thread(edge))

                page_info = (
                    json_data.get("data", {})
                    .get("shortcode_media", {})
                    .get("edge_media_to_parent_comment", {})
                    .get("page_info", {})
                )
                has_next_page = page_info.get("has_next_page", False)
                end_cursor = page_info.get("end_cursor")

            log_scraper_event(
                "info",
                "instagram",
                "comments_scraped",
                {"count": len(comments), "shortcode": shortcode},
            )

            return comments

