import re
from enum import Enum
from dataclasses import dataclass


class Language(str, Enum):
    ENGLISH = "english"
    ARABIC_MSA = "arabic_msa"
    LEBANESE_ARABIC = "lebanese_arabic"
    LEBANESE_ARABIZI = "lebanese_arabizi"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    language: Language
    confidence: float
    details: dict


ARABIZI_MARKERS = {
    "2", "3", "5", "7", "8", "9",
}

ARABIZI_WORDS = {
    "shu", "kifak", "keefak", "kifik", "keefik", "wen", "wayn",
    "lesh", "ya3ne", "ya3ni", "yalla", "khalas", "habibi", "habibti",
    "7abibi", "marhaba", "mar7aba", "ahla", "inshallah", "mashallah",
    "tfaddal", "mnih", "mni7", "ktir", "halla2", "bukra", "badde",
    "baddi", "baddak", "hek", "heik", "bas", "3am", "ta3mel",
    "zalame", "saba7o", "masa2", "la2", "5alas", "3a", "fi",
    "mish", "mesh", "howwe", "hiyye", "na7na", "ba3ref",
}

LEBANESE_ARABIC_MARKERS = {
    "شو", "كيفك", "وين", "ليش", "يعني", "هيك", "بس",
    "عم", "منيح", "كتير", "هلأ", "بكرا", "بدي", "بدك",
    "مبارح", "زلمة", "هوي", "هيي", "نحنا", "بعرف",
}


class LanguageDetector:

    def detect(self, text: str) -> DetectionResult:
        if not text or not text.strip():
            return DetectionResult(Language.UNKNOWN, 0.0, {})

        text = text.strip()
        scores = {
            Language.ENGLISH: 0.0,
            Language.ARABIC_MSA: 0.0,
            Language.LEBANESE_ARABIC: 0.0,
            Language.LEBANESE_ARABIZI: 0.0,
        }

        arabic_ratio, latin_ratio, digit_ratio = self._script_ratios(text)

        if arabic_ratio > 0.5:
            scores[Language.ARABIC_MSA] = arabic_ratio * 0.7
            scores[Language.LEBANESE_ARABIC] = arabic_ratio * 0.3

            lb_count = self._count_dialect_markers(text)
            if lb_count > 0:
                boost = min(lb_count * 0.15, 0.5)
                scores[Language.LEBANESE_ARABIC] += boost
                scores[Language.ARABIC_MSA] -= boost * 0.5

        elif latin_ratio > 0.5:
            scores[Language.ENGLISH] = latin_ratio * 0.6

            arabizi_score = self._arabizi_score(text)
            if arabizi_score > 0.2:
                scores[Language.LEBANESE_ARABIZI] = arabizi_score
                scores[Language.ENGLISH] *= (1 - arabizi_score)
            else:
                scores[Language.ENGLISH] += 0.3

        best_lang = max(scores, key=scores.get)
        confidence = min(scores[best_lang], 1.0)

        return DetectionResult(
            language=best_lang,
            confidence=round(confidence, 3),
            details={
                "scores": {k.value: round(v, 3) for k, v in scores.items()},
                "arabic_ratio": round(arabic_ratio, 3),
                "latin_ratio": round(latin_ratio, 3),
            },
        )

    def _script_ratios(self, text: str) -> tuple:
        total = len(text)
        if total == 0:
            return 0.0, 0.0, 0.0

        arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        latin = sum(1 for c in text if c.isascii() and c.isalpha())
        digits = sum(1 for c in text if c.isdigit())

        return arabic / total, latin / total, digits / total

    def _arabizi_score(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0

        score = 0.0
        total_words = len(words)

        numeral_pattern = re.compile(r"[a-z]*[235789][a-z]+|[a-z]+[235789][a-z]*")
        numeral_words = sum(1 for w in words if numeral_pattern.match(w))
        score += (numeral_words / total_words) * 0.5

        known_words = sum(1 for w in words if w in ARABIZI_WORDS)
        score += (known_words / total_words) * 0.5

        return min(score, 1.0)

    def _count_dialect_markers(self, text: str) -> int:
        count = 0
        for marker in LEBANESE_ARABIC_MARKERS:
            if marker in text:
                count += 1
        return count

_detector = LanguageDetector()


def detect_language(text: str) -> DetectionResult:
    return _detector.detect(text)