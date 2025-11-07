"""Yorumlara dayalı audience sentiment analizi."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from langdetect import DetectorFactory, LangDetectException, detect

from audience import AudienceSentimentSummary, CommentRecord, CommentThread


DetectorFactory.seed = 0

DEFAULT_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "is",
    "are",
    "was",
    "were",
    "be",
    "of",
    "with",
    "for",
    "this",
    "that",
    "it",
    "its",
    "on",
    "in",
    "at",
    "mi",
    "mı",
    "mu",
    "mü",
    "ve",
    "de",
    "da",
    "bir",
    "çok",
    "ama",
    "gibi",
    "için",
    "ile",
    "sen",
    "ben",
    "biz",
    "siz",
    "they",
    "you",
    "we",
}


POSITIVE_WORDS_EN = {
    "amazing",
    "awesome",
    "cool",
    "love",
    "loved",
    "loving",
    "great",
    "fantastic",
    "perfect",
    "best",
    "good",
    "nice",
    "happy",
    "enjoy",
    "enjoyed",
    "fire",
    "lit",
    "wow",
    "thanks",
    "fun",
    "favorite",
    "powerful",
    "solid",
    "top",
    "liked",
    "support",
    "beautiful",
    "brilliant",
    "excellent",
}

NEGATIVE_WORDS_EN = {
    "bad",
    "terrible",
    "awful",
    "hate",
    "hated",
    "boring",
    "dislike",
    "worst",
    "ugly",
    "garbage",
    "trash",
    "annoying",
    "fake",
    "lame",
    "poor",
    "problem",
    "issue",
    "bug",
    "broken",
    "slow",
    "pain",
    "angry",
    "sucks",
    "suck",
    "cringe",
}

POSITIVE_WORDS_TR = {
    "harika",
    "süper",
    "mükemmel",
    "efsane",
    "güzel",
    "şahane",
    "bayıldım",
    "sevdim",
    "severim",
    "kaliteli",
    "hızlı",
    "mutlu",
    "iyi",
    "enfes",
    "muazzam",
    "favorim",
    "teşekkürler",
    "efsanevi",
    "başarılı",
    "şaheser",
}

NEGATIVE_WORDS_TR = {
    "berbat",
    "kötü",
    "rezalet",
    "nefret",
    "iğrenç",
    "sinir",
    "sıkıcı",
    "beğenmedim",
    "çöp",
    "fiyasko",
    "problem",
    "sorun",
    "bozuk",
    "hatalı",
    "yavaş",
    "kriz",
    "düşük",
    "rezil",
    "saçma",
}


@dataclass
class SentimentConfig:
    positive_words_en: Tuple[str, ...] = tuple(sorted(POSITIVE_WORDS_EN))
    negative_words_en: Tuple[str, ...] = tuple(sorted(NEGATIVE_WORDS_EN))
    positive_words_tr: Tuple[str, ...] = tuple(sorted(POSITIVE_WORDS_TR))
    negative_words_tr: Tuple[str, ...] = tuple(sorted(NEGATIVE_WORDS_TR))
    neutral_threshold: float = 0.15


class AudienceSentimentAnalyzer:
    """Yorum listesi üzerinden lexicon tabanlı sentiment skorları hesaplar."""

    def __init__(self, config: Optional[SentimentConfig] = None) -> None:
        self.config = config or SentimentConfig()

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"@[\w_]+", "", text)
        text = re.sub(r"#[\w_]+", "", text)
        text = text.replace("\n", " ")
        return text.strip()

    @staticmethod
    def _detect_language_safe(text: str) -> Optional[str]:
        try:
            if len(text) < 3:
                return None
            return detect(text)
        except LangDetectException:
            return None

    def _prepare_samples(
        self, threads: List[CommentThread]
    ) -> List[CommentRecord]:
        comments: List[CommentRecord] = []
        for thread in threads:
            comments.append(thread.parent)
            comments.extend(thread.replies)
        return comments

    def _score_text(self, text: str, language: Optional[str]) -> float:
        tokens = re.findall(r"[\wöçşığüÖÇŞİĞÜ]+", text.lower())
        if language and language.startswith("tr"):
            pos_set = self.config.positive_words_tr
            neg_set = self.config.negative_words_tr
        else:
            pos_set = self.config.positive_words_en
            neg_set = self.config.negative_words_en

        positive = sum(token in pos_set for token in tokens)
        negative = sum(token in neg_set for token in tokens)
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total

    @staticmethod
    def _label_from_score(score: float, neutral_threshold: float) -> str:
        if score > neutral_threshold:
            return "positive"
        if score < -neutral_threshold:
            return "negative"
        return "neutral"

    @staticmethod
    def _top_keywords(texts: Iterable[str], limit: int = 15) -> List[str]:
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = re.findall(r"[\wöçşığüÖÇŞİĞÜ]+", text.lower())
            for token in tokens:
                if token in DEFAULT_STOPWORDS or len(token) < 3:
                    continue
                counter[token] += 1
        return [token for token, _ in counter.most_common(limit)]

    def analyze(self, threads: List[CommentThread]) -> AudienceSentimentSummary:
        records = self._prepare_samples(threads)
        normalized_texts = [self._normalize_text(r.text) for r in records if r.text]
        filtered_records = [r for r, text in zip(records, normalized_texts) if text]
        filtered_texts = [text for text in normalized_texts if text]

        sentiment_totals = defaultdict(float)
        samples = defaultdict(list)

        for record, text in zip(filtered_records, filtered_texts):
            lang = record.language or self._detect_language_safe(text)
            record.language = lang
            score = self._score_text(text, lang)
            label = self._label_from_score(score, self.config.neutral_threshold)
            sentiment_totals[label] += 1
            if len(samples[label]) < 5:
                samples[label].append(text)

        total = float(sum(sentiment_totals.values()))
        positive_ratio = (sentiment_totals.get("positive", 0.0) / total) if total else 0.0
        negative_ratio = (sentiment_totals.get("negative", 0.0) / total) if total else 0.0
        neutral_ratio = (sentiment_totals.get("neutral", 0.0) / total) if total else 0.0

        dominant_pairs = sorted(
            (
                ("joy", positive_ratio),
                ("anger", negative_ratio),
                ("calm", neutral_ratio),
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        dominant_emotions = [label for label, ratio in dominant_pairs if ratio > 0]

        keywords = self._top_keywords(filtered_texts)

        return AudienceSentimentSummary(
            positive_ratio=min(1.0, max(0.0, positive_ratio)),
            negative_ratio=min(1.0, max(0.0, negative_ratio)),
            neutral_ratio=min(1.0, max(0.0, neutral_ratio)),
            dominant_emotions=dominant_emotions,
            keywords=keywords,
            sample_quotes_positive=samples.get("positive", [])[:5],
            sample_quotes_negative=samples.get("negative", [])[:5],
            sample_quotes_neutral=samples.get("neutral", [])[:5],
            total_comments=len(records),
            analyzed_comments=len(filtered_records),
        )


