from enum import Enum
from dataclasses import dataclass


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

INTENT_PATTERNS = {
    Intent.GREETING: {
        "en": ["hello", "hi", "hey", "good morning", "good evening", "howdy", "what's up", "sup"],
        "ar": ["مرحبا", "أهلا", "السلام عليكم", "صباح الخير", "مساء الخير", "كيفك", "كيف حالك"],
        "arabizi": ["marhaba", "mar7aba", "ahla", "kifak", "keefak", "kifik", "saba7o", "hi"],
    },
    Intent.FAREWELL: {
        "en": ["bye", "goodbye", "see you", "take care", "later", "goodnight"],
        "ar": ["مع السلامة", "باي", "تصبح على خير", "الله معك", "يلا باي"],
        "arabizi": ["bye", "yalla bye", "ma3 el salame", "bbye"],
    },
    Intent.QUESTION: {
        "en": ["what", "how", "why", "when", "where", "who", "which", "can you", "do you", "is it", "tell me"],
        "ar": ["ما", "كيف", "لماذا", "ليش", "متى", "أين", "وين", "مين", "شو", "هل"],
        "arabizi": ["shu", "kif", "lesh", "wen", "wayn", "min", "emta"],
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


class IntentClassifier:

    def __init__(self):
        self.patterns = INTENT_PATTERNS

    def classify(self, text: str) -> IntentResult:

        if not text or not text.strip():
            return IntentResult(Intent.UNKNOWN, 0.0, {})

        text_lower = text.lower().strip()
        scores = {}

        for intent, lang_patterns in self.patterns.items():
            score = 0.0
            for lang, keywords in lang_patterns.items():
                for kw in keywords:
                    if kw in text_lower:
                        if f" {kw} " in f" {text_lower} ":
                            score += 1.0
                        else:
                            score += 0.5

            scores[intent] = score

        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        if total == 0:
            return IntentResult(Intent.CHITCHAT, 0.5, {i.value: 0 for i in Intent})

        best = max(scores, key=scores.get)
        return IntentResult(
            intent=best,
            confidence=round(scores[best], 3),
            all_scores={k.value: round(v, 3) for k, v in scores.items()},
        )


def classify_intent(text: str) -> IntentResult:
    return IntentClassifier().classify(text)