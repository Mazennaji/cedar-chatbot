from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    label: SentimentLabel
    score: float 
    confidence: float


POSITIVE_WORDS = {
    "en": {"good", "great", "awesome", "amazing", "love", "happy", "excellent",
           "wonderful", "fantastic", "nice", "perfect", "beautiful", "best",
           "thanks", "thank", "appreciate", "helpful", "brilliant"},
    "ar": {"جيد", "ممتاز", "رائع", "حلو", "منيح", "أحسن", "حبيبي",
           "شكرا", "يسلمو", "بحب", "مبسوط", "فرحان"},
    "arabizi": {"7elo", "mnih", "mni7", "a7san", "habibi", "merci",
                "shukran", "yslmo", "mabsout", "great", "nice"},
}

NEGATIVE_WORDS = {
    "en": {"bad", "terrible", "awful", "hate", "worst", "horrible", "ugly",
           "stupid", "boring", "annoying", "disappointed", "angry", "sad",
           "problem", "wrong", "broken", "fail", "useless"},
    "ar": {"سيء", "بشع", "مقرف", "زفت", "غلط", "مشكلة", "زعلان",
           "حزين", "غضبان", "مكسور"},
    "arabizi": {"zeft", "ba2ref", "ghalat", "mushkle", "mshkle",
                "za3lan", "7azin", "bad", "hate"},
}

INTENSIFIERS = {"very", "so", "really", "extremely", "ktir", "كتير", "جداً", "مرة"}
NEGATORS = {"not", "no", "don't", "doesn't", "never", "msh", "mish", "mesh",
            "ما", "مش", "لا", "مو"}


class SentimentAnalyzer:

    def __init__(self):
        self._pos_all = set()
        self._neg_all = set()
        for words in POSITIVE_WORDS.values():
            self._pos_all.update(words)
        for words in NEGATIVE_WORDS.values():
            self._neg_all.update(words)

    def analyze(self, text: str) -> SentimentResult:

        if not text or not text.strip():
            return SentimentResult(SentimentLabel.NEUTRAL, 0.0, 0.5)

        words = text.lower().split()
        pos_score = 0.0
        neg_score = 0.0
        negate = False
        intensify = 1.0

        for i, word in enumerate(words):
            if word in NEGATORS:
                negate = True
                continue

            if word in INTENSIFIERS:
                intensify = 1.5
                continue

            if word in self._pos_all:
                if negate:
                    neg_score += 1.0 * intensify
                else:
                    pos_score += 1.0 * intensify
                negate = False
                intensify = 1.0

            elif word in self._neg_all:
                if negate:
                    pos_score += 0.5 * intensify
                else:
                    neg_score += 1.0 * intensify
                negate = False
                intensify = 1.0
            else:

                if i > 0:
                    negate = False
                    intensify = 1.0

        total = pos_score + neg_score
        if total == 0:
            return SentimentResult(SentimentLabel.NEUTRAL, 0.0, 0.4)

        raw_score = (pos_score - neg_score) / total  # -1.0 to 1.0
        confidence = min(total / len(words), 1.0) if words else 0.0

        if raw_score > 0.15:
            label = SentimentLabel.POSITIVE
        elif raw_score < -0.15:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        return SentimentResult(
            label=label,
            score=round(raw_score, 3),
            confidence=round(confidence, 3),
        )


def analyze_sentiment(text: str) -> SentimentResult:
    return SentimentAnalyzer().analyze(text)