import json
import re
import os
from pathlib import Path
from typing import Optional

ARABIZI_MAP = {
    "2": "ء", 
    "3": "ع",
    "3'": "غ",
    "5": "خ",
    "6": "ط",
    "6'": "ظ",
    "7": "ح",
    "8": "ق",
    "9": "ص", 
    "9'": "ض", 

    "a": "ا",
    "b": "ب",
    "t": "ت",
    "th": "ث",
    "j": "ج",
    "ch": "تش",
    "h": "ه",
    "kh": "خ",
    "d": "د",
    "dh": "ذ",
    "r": "ر",
    "z": "ز",
    "s": "س",
    "sh": "ش",
    "ss": "ص",
    "dd": "ض",
    "tt": "ط",
    "dz": "ظ",
    "aa": "ع",
    "gh": "غ",
    "f": "ف",
    "q": "ق",
    "k": "ك",
    "l": "ل",
    "m": "م",
    "n": "ن",
    "w": "و",
    "y": "ي",
    "e": "ي",
    "i": "ي",
    "o": "و",
    "u": "و",
}

MULTI_CHAR_MAP = {
    "3'": "غ",
    "6'": "ظ",
    "9'": "ض",
    "sh": "ش",
    "ch": "تش",
    "th": "ث",
    "kh": "خ",
    "dh": "ذ",
    "gh": "غ",
    "ss": "ص",
    "dd": "ض",
    "tt": "ط",
    "dz": "ظ",
    "aa": "ع",
    "ou": "و",
    "ee": "ي",
    "oo": "و",
}


LEBANESE_PHRASES = {
    "keefak": "كيفك",
    "keefik": "كيفك",
    "kifak": "كيفك",
    "kifik": "كيفك",
    "marhaba": "مرحبا",
    "mar7aba": "مرحبا",
    "ahla": "أهلا",
    "ahlan": "أهلاً",
    "salam": "سلام",
    "sabaho": "صباحو",
    "saba7o": "صباحو",
    "saba7": "صباح",
    "masa": "مسا",
    "masa2": "مساء",

    "shu": "شو",
    "sho": "شو",
    "wen": "وين",
    "wayn": "وين",
    "kif": "كيف",
    "lesh": "ليش",
    "la2": "لأ",
    "eh": "إيه",
    "ya3ne": "يعني",
    "ya3ni": "يعني",
    "yalla": "يلا",
    "khalas": "خلص",
    "5alas": "خلص",
    "habibi": "حبيبي",
    "7abibi": "حبيبي",
    "habibti": "حبيبتي",
    "7abibti": "حبيبتي",
    "inshalla": "إنشاالله",
    "inshallah": "إنشاالله",
    "mashalla": "ماشاالله",
    "mashallah": "ماشاالله",
    "tfaddal": "تفضل",
    "tfaddali": "تفضلي",
    "merci": "ميرسي",
    "mnih": "منيح",
    "mni7": "منيح",
    "mni7a": "منيحة",
    "ktir": "كتير",
    "halla2": "هلأ",
    "halla": "هلأ",
    "bukra": "بكرا",
    "mberi7": "مبارح",
    "3am": "عم",
    "badde": "بدي",
    "baddi": "بدي",
    "baddak": "بدك",
    "baddik": "بدك",
    "fi": "في",
    "ma": "ما",
    "mish": "مش",
    "mesh": "مش",
    "hek": "هيك",
    "heik": "هيك",
    "bas": "بس",
    "w": "و",
    "la": "ل",
    "bi": "ب",
    "min": "من",
    "3a": "ع",
    "3al": "عال",
    "ta3mel": "تعمل",
    "ta3mol": "تعمل",
    "zalame": "زلمة",
    "za3ame": "زلمة",
    "enta": "إنت",
    "ente": "إنتي",
    "howwe": "هوي",
    "hiyye": "هيي",
    "na7na": "نحنا",
    "ne7na": "نحنا",

    "ba3ref": "بعرف",
    "b3rf": "بعرف",
    "baddna": "بدنا",
    "re7na": "رحنا",
    "e7na": "إحنا",
    "jeye": "جايي",
    "raye7": "رايح",
    "ray7a": "رايحة",
}


class ArabiziNormalizer:

    def __init__(self, custom_phrases_path: Optional[str] = None):
        self.phrases = dict(LEBANESE_PHRASES)
        self.multi_char = dict(MULTI_CHAR_MAP)
        self.single_char = {k: v for k, v in ARABIZI_MAP.items() if len(k) == 1}

        if custom_phrases_path and os.path.exists(custom_phrases_path):
            with open(custom_phrases_path, "r", encoding="utf-8") as f:
                custom = json.load(f)
                self.phrases.update(custom)

        self._sorted_multi = sorted(
            self.multi_char.items(), key=lambda x: len(x[0]), reverse=True
        )

    def normalize(self, text: str) -> str:

        if not text or not text.strip():
            return text

        if self._is_arabic(text):
            return text

        words = text.strip().split()
        result = []

        for word in words:
            normalized = self._normalize_word(word.lower())
            result.append(normalized)

        output = " ".join(result)
        return self._post_process(output)

    def _normalize_word(self, word: str) -> str:

        clean, prefix_punct, suffix_punct = self._strip_punctuation(word)

        if clean in self.phrases:
            return prefix_punct + self.phrases[clean] + suffix_punct

        arabic = self._transliterate(clean)
        return prefix_punct + arabic + suffix_punct

    def _transliterate(self, text: str) -> str:
        result = []
        i = 0

        while i < len(text):
            matched = False

            for pattern, replacement in self._sorted_multi:
                if text[i:i + len(pattern)] == pattern:
                    result.append(replacement)
                    i += len(pattern)
                    matched = True
                    break

            if not matched:
                char = text[i]
                if char in self.single_char:
                    result.append(self.single_char[char])
                elif char.isdigit() and char in ARABIZI_MAP:
                    result.append(ARABIZI_MAP[char])
                else:
                    result.append(char)
                i += 1

        return "".join(result)

    def _strip_punctuation(self, word: str) -> tuple:
        prefix = ""
        suffix = ""

        while word and not word[0].isalnum():
            prefix += word[0]
            word = word[1:]

        while word and not word[-1].isalnum():
            suffix = word[-1] + suffix
            word = word[:-1]

        return word, prefix, suffix

    def _post_process(self, text: str) -> str:

        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        text = re.sub(r"\s+([،؛؟])", r"\1", text)
        return text.strip()

    @staticmethod
    def _is_arabic(text: str) -> bool:
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha == 0:
            return False
        return (arabic_chars / total_alpha) > 0.5

    def add_phrase(self, arabizi: str, arabic: str):
        self.phrases[arabizi.lower()] = arabic

    def get_mapping_stats(self) -> dict:
        return {
            "phrase_count": len(self.phrases),
            "multi_char_rules": len(self.multi_char),
            "single_char_rules": len(self.single_char),
        }


_default_normalizer = None


def normalize_arabizi(text: str) -> str:
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = ArabiziNormalizer()
    return _default_normalizer.normalize(text)