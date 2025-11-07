from audience import CommentRecord, CommentThread
from analytics import AudienceSentimentAnalyzer, SentimentConfig


def _build_thread(parent_text: str, replies: list[str] | None = None) -> CommentThread:
    parent = CommentRecord(comment_id="1", text=parent_text)
    reply_records = []
    for idx, text in enumerate(replies or [], start=1):
        reply_records.append(CommentRecord(comment_id=f"1-{idx}", text=text))
    return CommentThread(parent=parent, replies=reply_records)


def test_sentiment_positive_bias():
    analyzer = AudienceSentimentAnalyzer(SentimentConfig())
    threads = [
        _build_thread("This video is amazing and I love it"),
        _build_thread("Great job team"),
    ]

    summary = analyzer.analyze(threads)

    assert summary.total_comments == 2
    assert summary.analyzed_comments == 2
    assert summary.positive_ratio > summary.negative_ratio


def test_sentiment_negative_bias():
    analyzer = AudienceSentimentAnalyzer(SentimentConfig())
    threads = [
        _build_thread("Bu video berbat oldu"),
        _build_thread("Çok kötü ve sıkıcı"),
    ]

    summary = analyzer.analyze(threads)

    assert summary.total_comments == 2
    assert summary.analyzed_comments == 2
    assert summary.negative_ratio > summary.positive_ratio

