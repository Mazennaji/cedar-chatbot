from enum import Enum
from dataclasses import dataclass
import re


class Intent(str, Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    QUESTION = "question"
    COMPLAINT = "complaint"
    FEEDBACK = "feedback"
    REQUEST = "request"
    CHITCHAT = "chitchat"
    THANKS = "thanks"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    all_scores: dict


GREETING_PHRASES = [
    "how are you", "how r you", "how are u", "what's up", "what up", "sup",
    "how's it going", "how is it going", "how have you been",
    "كيف حالك", "كيف الحال", "كيف حالك؟", "كيف الحال؟",
    "كيفك", "كيفكم", "شو أخبارك", "شو اخبارك",
    "kifak", "keefak", "kifik", "kif halak", "kif el hal",
]

QUESTION_STARTERS = ["what", "why", "when", "where", "who", "which", "how"]

INTENT_PATTERNS = {
    Intent.GREETING: {
        "en": ["hello", "hi", "hey", "good morning", "good evening", "howdy"],
        "ar": ["مرحبا", "أهلا", "السلام عليكم", "صباح الخير", "مساء الخير"],
        "arabizi": ["marhaba", "mar7aba", "ahla", "saba7o"],
    },
    Intent.FAREWELL: {
        "en": ["bye", "goodbye", "see you", "take care", "later", "goodnight"],
        "ar": ["مع السلامة", "باي", "تصبح على خير", "الله معك", "يلا باي"],
        "arabizi": ["bye", "yalla bye", "ma3 el salame", "bbye"],
    },
    Intent.QUESTION: {
        "en": ["what", "why", "when", "where", "who", "which", "can you", "do you", "is it", "tell me", "how"],
        "ar": ["ما", "لماذا", "ليش", "متى", "أين", "وين", "مين", "شو", "هل", "ماذا"],
        "arabizi": ["shu", "lesh", "wen", "wayn", "min", "emta"],
    },
    Intent.COMPLAINT: {
        "en": ["bad", "terrible", "awful", "hate", "wrong", "broken", "doesn't work", "problem", "issue", "bug"],
        "ar": ["سيء", "مشكلة", "خطأ", "ما بيشتغل", "بشع"],
        "arabizi": ["mshkle", "mushkle", "ghalat", "ma byeshteghel"],
    },
    Intent.THANKS: {
        "en": ["thank", "thanks", "appreciate", "grateful"],
        "ar": ["شكرا", "يسلمو", "يعطيك العافية", "ممنون"],
        "arabizi": ["merci", "shukran", "yslmo", "ya3tik", "thanks"],
    },
    Intent.FEEDBACK: {
        "en": ["suggest", "feedback", "improve", "better", "opinion", "think"],
        "ar": ["رأي", "اقتراح", "تحسين"],
        "arabizi": ["ra2y", "suggestion"],
    },
    Intent.REQUEST: {
        "en": ["please", "could you", "would you", "help", "need", "want", "give me", "show me"],
        "ar": ["ساعدني", "بدي", "أعطيني", "عطيني", "من فضلك"],
        "arabizi": ["badde", "baddi", "sa3edni", "3atini"],
    },
}


def _whole_word_match(keyword: str, text: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
    return bool(re.search(pattern, text))


class IntentClassifier:

    def __init__(self):
        self.patterns = INTENT_PATTERNS

    def classify(self, text: str) -> IntentResult:
        if not text or not text.strip():
            return IntentResult(Intent.UNKNOWN, 0.0, {})

        text_lower = text.lower().strip()

        first_word = text_lower.split()[0].rstrip("?!.,") if text_lower.split() else ""
        has_question_mark = "?" in text_lower

        if first_word in QUESTION_STARTERS or has_question_mark:
            for phrase in GREETING_PHRASES:
                if phrase in text_lower:
                    pass
            if first_word in QUESTION_STARTERS or has_question_mark:
                is_pure_greeting = any(
                    text_lower.strip("?! ") == phrase or text_lower == phrase
                    for phrase in GREETING_PHRASES
                )
                if not is_pure_greeting:
                    for phrase in GREETING_PHRASES:
                        if phrase in text_lower and first_word not in QUESTION_STARTERS:
                            return IntentResult(
                                intent=Intent.GREETING,
                                confidence=0.95,
                                all_scores={i.value: 0.0 for i in Intent} | {Intent.GREETING.value: 0.95},
                            )
                    scores = self._score(text_lower)
                    total = sum(scores.values())
                    if total > 0:
                        scores = {k: v / total for k, v in scores.items()}
                    best = max(scores, key=scores.get)
                    return IntentResult(
                        intent=best,
                        confidence=round(scores[best], 3),
                        all_scores={k.value: round(v, 3) for k, v in scores.items()},
                    )

        for phrase in GREETING_PHRASES:
            if phrase in text_lower:
                return IntentResult(
                    intent=Intent.GREETING,
                    confidence=0.95,
                    all_scores={i.value: 0.0 for i in Intent} | {Intent.GREETING.value: 0.95},
                )

        scores = self._score(text_lower)
        total = sum(scores.values())

        if total == 0:
            return IntentResult(Intent.CHITCHAT, 0.5, {i.value: 0 for i in Intent})

        scores = {k: v / total for k, v in scores.items()}
        best = max(scores, key=scores.get)

        return IntentResult(
            intent=best,
            confidence=round(scores[best], 3),
            all_scores={k.value: round(v, 3) for k, v in scores.items()},
        )

    def _score(self, text_lower: str) -> dict:
        scores = {}
        for intent, lang_patterns in self.patterns.items():
            score = 0.0
            for lang, keywords in lang_patterns.items():
                for kw in keywords:
                    if kw in text_lower:
                        if _whole_word_match(kw, text_lower):
                            score += 1.0
                        else:
                            score += 0.5
            scores[intent] = score
        return scores


def classify_intent(text: str) -> IntentResult:
    return IntentClassifier().classify(text)