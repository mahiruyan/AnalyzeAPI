"""Platform scraper paketinin genel arabirimi."""

from .base import ScraperConfig, ScrapeError
from .instagram import InstagramScraper
from .tiktok import TikTokScraper

__all__ = [
    "ScraperConfig",
    "ScrapeError",
    "InstagramScraper",
    "TikTokScraper",
]

