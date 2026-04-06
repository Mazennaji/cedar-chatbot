from enum import Enum
from dataclasses import dataclass


class ArabicDialect(str, Enum):
    MSA = "msa"
    LEBANESE = "lebanese"
    EGYPTIAN = "egyptian"
    GULF = "gulf"
    UNKNOWN = "unknown"


DIALECT_LABELS = {
    ArabicDialect.MSA: "Modern Standard Arabic (فصحى)",
    ArabicDialect.LEBANESE: "Lebanese (لبناني)",
    ArabicDialect.EGYPTIAN: "Egyptian (مصري)",
    ArabicDialect.GULF: "Gulf (خليجي)",
    ArabicDialect.UNKNOWN: "Unknown",
}


@dataclass
class DialectResult:
    dialect: ArabicDialect
    dialect_label: str
    confidence: float
    scores: dict
    markers_found: dict


LEBANESE_MARKERS = {
    "هلق", "هلأ",
    "شو",
    "كيفك",
    "منيح",
    "كتير",
    "هيك",
    "بس",
    "هيدا",
    "هيدي",
    "ليش",
    "وين",
    "بدي",
    "بدك",
    "بدنا",
    "عم",
    "رح",
    "فيني",
    "فيك",
    "مبارح",
    "بكرا",
    "زلمة",
    "يلا",
    "خلص",
    "حكي",
    "منحكي",
    "نحنا",
    "هوي",
    "هيي",
    "إنت",
    "إنتي",
    "ميرسي",
    "يسلمو",
    "تكرم",
    "معلش",
    "بعرف",
    "ما بعرف",
    "مش",
    "حبيبي",
    "صباحو",
    "يعطيك العافية",
    "عفواً",
    "تفضل",
    "نشالله",
    "إنشاالله",
    "ماشاالله",
}

EGYPTIAN_MARKERS = {
    "إزيك",
    "ازيك",
    "ازاي",
    "إزاي",
    "كده",
    "كدا",
    "دلوقتي",
    "بتاع",
    "بتاعت",
    "بتاعي",
    "ليه",
    "فين",
    "عايز",
    "عايزة",
    "عاوز",
    "هنا",
    "يعني",
    "بقى",
    "خالص",
    "أوي",
    "قوي",
    "مفيش",
    "حاجة",
    "حاجات",
    "ده",
    "دي",
    "دول",
    "إحنا",
    "إنتو",
    "هو",
    "هي",
    "مش",
    "بس",
    "طب",
    "يا باشا",
    "يا أستاذ",
    "ربنا",
    "يا ريت",
    "بالظبط",
    "أصلي",
    "عشان",
    "علشان",
    "كمان",
    "برضو",
    "بردو",
    "خلاص",
    "ماشي",
    "حلو",
    "جميل",
    "تمام",
    "بأه",
    "أنهي",
}

GULF_MARKERS = {
    "شلونك",
    "شلونج",
    "وش",
    "ليش",
    "وين",
    "ابي",
    "أبي",
    "أبغى",
    "يبي",
    "تبي",
    "زين",
    "مرة",
    "واجد",
    "هالحين",
    "الحين",
    "حلو",
    "يا حبيبي",
    "يا خوي",
    "يا اخوي",
    "يالله",
    "خلاص",
    "إنشاء الله",
    "ماشاء الله",
    "هذا",
    "هذي",
    "ذا",
    "ذي",
    "حنا",
    "انت",
    "انتي",
    "هم",
    "شي",
    "يا رجال",
    "يا ولد",
    "حياك",
    "حياك الله",
    "ما عليك",
    "الله يعافيك",
    "يا سلام",
    "عاد",
    "توه",
    "توها",
    "لا هنت",
    "يا وجه الخير",
    "صج",
    "أكيد",
    "كفو",
}

MSA_MARKERS = {
    "الذي",             
    "التي",             
    "الذين",           
    "إن",               
    "أن",               
    "لكن",              
    "حيث",              
    "إذا",              
    "على",              
    "من",               
    "إلى",              
    "بالتالي",          
    "بالإضافة",         
    "علاوة",            
    "يتعين",            
    "ينبغي",            
    "لذلك",             
    "هذا",              
    "ذلك",              
    "أولئك",            
    "نحن",              
    "أنتم",             
    "يمكن",             
    "يجب",              
    "سوف",              
    "ليس",              
    "لأن",              
    "رغم",              
    "بينما",            
    "خلال",             
    "حول",             
    "تعتبر",            
    "يعد",              
    "فضلاً",            
    "وفقاً",            
    "نظراً",            
}

LEBANESE_ARABIZI_MARKERS = {
    "kifak", "keefak", "kifik", "keefik", "shu", "wen", "wayn",
    "halla2", "mni7", "ktir", "heik", "hek", "badde", "baddi",
    "baddak", "zalame", "yalla", "5alas", "khalas", "7abibi",
    "habibi", "marhaba", "mar7aba", "3am", "ta3mel", "bukra",
    "mbereh", "ne7na", "howwe", "hiyye", "ba3ref", "merci",
    "yslmo", "tekram", "tfaddal", "nshallah", "inshallah",
    "masa2", "saba7o", "la2", "fi", "mish", "mesh", "bas",
}

EGYPTIAN_ARABIZI_MARKERS = {
    "ezayak", "ezzayak", "ezayek", "ezzayek", "ezay",
    "keda", "kda", "delwa2ti", "dilwa2ti", "beta3",
    "3ayez", "3ayza", "3awez", "feen", "leih", "leh",
    "2awy", "awi", "gamed", "gameela", "mafesh",
    "7aga", "da", "di", "dol", "e7na", "ento",
    "tab", "basha", "yasta", "ya ray7", "3ashan",
    "kaman", "bardo", "brdo", "5alas", "mashi",
    "tamam", "7elw", "gameel",
}

GULF_ARABIZI_MARKERS = {
    "shlonak", "shlonk", "shlonich", "wesh", "wsh",
    "abi", "abgha", "aby", "yabi", "tabi",
    "zain", "zen", "wayed", "wajed", "marra",
    "7alyeen", "el7een", "hal7in",
    "ya 5ooy", "ya khooy", "5oy",
    "7ayak", "7ayak allah", "la hant",
    "9aj", "saj", "akeed", "kafo", "kfo",
    "wallah", "inshallah", "mashallah",
    "tawwa", "tawah",
}


class ArabicDialectDetector:
    def __init__(self):
        self.dialect_markers = {
            ArabicDialect.LEBANESE: LEBANESE_MARKERS,
            ArabicDialect.EGYPTIAN: EGYPTIAN_MARKERS,
            ArabicDialect.GULF: GULF_MARKERS,
            ArabicDialect.MSA: MSA_MARKERS,
        }

        self.arabizi_markers = {
            ArabicDialect.LEBANESE: LEBANESE_ARABIZI_MARKERS,
            ArabicDialect.EGYPTIAN: EGYPTIAN_ARABIZI_MARKERS,
            ArabicDialect.GULF: GULF_ARABIZI_MARKERS,
        }

    def detect(self, text: str) -> DialectResult:
        if not text or not text.strip():
            return DialectResult(
                dialect=ArabicDialect.UNKNOWN,
                dialect_label="Unknown",
                confidence=0.0,
                scores={},
                markers_found={},
            )

        text = text.strip()

        is_arabic_script = self._is_arabic(text)

        if is_arabic_script:
            scores, found = self._score_arabic(text)
        else:
            scores, found = self._score_arabizi(text)

        if not scores or all(v == 0 for v in scores.values()):
            best = ArabicDialect.UNKNOWN
            confidence = 0.0
        else:
            best = max(scores, key=scores.get)
            total = sum(scores.values())
            confidence = scores[best] / total if total > 0 else 0.0

        return DialectResult(
            dialect=best,
            dialect_label=DIALECT_LABELS.get(best, "Unknown"),
            confidence=round(confidence, 3),
            scores={k.value: round(v, 3) for k, v in scores.items()},
            markers_found={k.value: v for k, v in found.items()},
        )

    def _score_arabic(self, text: str) -> tuple:
        words = set(text.split())
        scores = {}
        found = {}

        for dialect, markers in self.dialect_markers.items():
            matches = words & markers
            for marker in markers:
                if " " in marker and marker in text:
                    matches.add(marker)

            scores[dialect] = len(matches)
            found[dialect] = list(matches)

        return scores, found

    def _score_arabizi(self, text: str) -> tuple:
        words = set(text.lower().split())
        words = {w.strip("?!.,;:'\"()[]") for w in words}

        scores = {}
        found = {}

        for dialect, markers in self.arabizi_markers.items():
            matches = words & markers
            scores[dialect] = len(matches)
            found[dialect] = list(matches)

        scores[ArabicDialect.MSA] = 0
        found[ArabicDialect.MSA] = []

        return scores, found

    def _is_arabic(self, text: str) -> bool:
        arabic_count = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        latin_count = sum(1 for c in text if c.isascii() and c.isalpha())
        return arabic_count > latin_count

    def get_dialect_info(self, dialect: ArabicDialect) -> dict:
        info = {
            ArabicDialect.LEBANESE: {
                "name": "Lebanese Arabic",
                "name_ar": "لبناني",
                "region": "Lebanon",
                "characteristics": [
                    "Uses 'شو' for 'what' instead of 'ماذا'",
                    "Future tense with 'رح' instead of 'سوف'",
                    "Progressive marker 'عم' before verbs",
                    "French loanwords like 'ميرسي'",
                    "'هلأ/هلق' for 'now'",
                    "Negation with 'ما' or 'مش'",
                ],
                "example": "كيفك؟ شو عم تعمل هلأ؟",
            },
            ArabicDialect.EGYPTIAN: {
                "name": "Egyptian Arabic",
                "name_ar": "مصري",
                "region": "Egypt",
                "characteristics": [
                    "Uses 'إزاي' for 'how'",
                    "'دلوقتي' for 'now'",
                    "'عايز/عاوز' for 'I want'",
                    "'بتاع' for possession",
                    "'أوي/قوي' for 'very'",
                    "Uses 'ده/دي/دول' for demonstratives",
                ],
                "example": "إزيك؟ عامل إيه دلوقتي؟",
            },
            ArabicDialect.GULF: {
                "name": "Gulf Arabic",
                "name_ar": "خليجي",
                "region": "Saudi Arabia, UAE, Kuwait, Qatar, Bahrain, Oman",
                "characteristics": [
                    "Uses 'شلونك' for 'how are you'",
                    "'وش' for 'what'",
                    "'أبغى/أبي' for 'I want'",
                    "'الحين/هالحين' for 'now'",
                    "'واجد/مرة' for 'very'",
                    "'حياك الله' for welcome",
                ],
                "example": "شلونك؟ وش تبي الحين؟",
            },
            ArabicDialect.MSA: {
                "name": "Modern Standard Arabic",
                "name_ar": "فصحى",
                "region": "All Arab countries (formal)",
                "characteristics": [
                    "Formal grammar and vocabulary",
                    "Uses 'الذي/التي' for relative pronouns",
                    "'سوف/سَ' for future tense",
                    "'ينبغي/يجب' for obligation",
                    "Full case endings (in formal speech)",
                    "Used in media, education, and official documents",
                ],
                "example": "كيف حالك؟ ماذا تفعل الآن؟",
            },
        }
        return info.get(dialect, {})


_detector = ArabicDialectDetector()


def detect_dialect(text: str) -> DialectResult:
    return _detector.detect(text)


def get_dialect_info(dialect: ArabicDialect) -> dict:
    return _detector.get_dialect_info(dialect)