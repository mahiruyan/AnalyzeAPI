from scraper.instagram import InstagramScraper
from scraper.tiktok import TikTokScraper


def test_instagram_shortcode_resolution():
    scraper = InstagramScraper()
    shortcode = scraper._resolve_shortcode("https://www.instagram.com/reel/ABC123xyz/")
    assert shortcode == "ABC123xyz"


def test_tiktok_sigi_state_parsing():
    scraper = TikTokScraper()
    html = """
    <html><head></head><body>
    <script id="SIGI_STATE">{"Comment":{"commentData":{},"comments":{}}}</script>
    </body></html>
    """
    state = scraper._extract_state(html)
    assert "Comment" in state

