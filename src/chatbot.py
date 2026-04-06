import time
import random
import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.normalizer import ArabiziNormalizer
from src.language_detector import LanguageDetector, Language
from src.intent_classifier import IntentClassifier, Intent
from src.sentiment import SentimentAnalyzer
from src.memory import ConversationMemory

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    response: str
    session_id: str
    message_id: str = ""
    metadata: dict = field(default_factory=dict)


ARABIZI_RESPONSES = {
    Intent.GREETING: [
        "Ahla w sahla! Kifak/kifik? 🌲",
        "Marhaba! Shu akhbarak lyom?",
        "Hala wallah! Keefak ya zalameh?",
        "Hey! Ahla fiik, shu 3am ta3mel?",
        "Marhaba habibi! Kifak, mni7?",
        "Sabbaho! Shu akhbar el yom?",
        "Ahlan! Ana hon la sa3dak, shu baddak?",
        "Ya ahla w sahla! Keefak el yom?",
    ],
    Intent.FAREWELL: [
        "Yalla bye! Allah ma3ak 👋",
        "Ma3 el salame! Nshallah mne7ke ba3den",
        "Bye habibi! Take care ya zalameh",
        "Yalla, bkra mne7ke. Tisbah 3a kher!",
        "Allah y7afzak! Bye bye 👋",
        "Yalla ma3 el salame, nshallah mne7ke 2arib!",
    ],
    Intent.QUESTION: [
        "Soual mni7! Khalline fakker shway... {answer}",
        "Ah, soual 7elo! {answer}",
        "Ya3ne, {answer}",
        "Hala2 bi jawbak... {answer}",
        "Mni7 ennak sa2alt! {answer}",
        "Soual 7elo wallah! {answer}",
    ],
    Intent.THANKS: [
        "3afwan habibi! Ay wa2et 🌲",
        "Tekram ya zalameh! Ana hon dayman",
        "La shukr 3a wajeb! Shu baddak kamen?",
        "3afwan! Btsharrafna fiik",
        "Ahla fiik! Ma fi shi, ay khedme",
    ],
    Intent.COMPLAINT: [
        "Sorry ya zalameh, khalline 7awel sa3dak a7san",
        "Ma3lesh, 7a2ak 3layye. Shu fi2e sa3dak?",
        "Ah wallah sorry, khalline jarreb marra tanye",
        "Ma tkun za3lan, 7a7awel a7san el marra el jeye",
    ],
    Intent.FEEDBACK: [
        "Merci 3al feedback! Bi sa3edne ktir",
        "Shukran! Ra7 7awel et7assan",
        "Mni7 ennak 2eltelle, merci!",
    ],
    Intent.CHITCHAT: [
        "Hahaha wallah! {answer}",
        "Eh sah, {answer}",
        "Ya3ne, {answer}",
        "Ahh mni7! {answer}",
        "Wallah? {answer}",
        "Hala2 bi 2ellak shi... {answer}",
        "Ah ktir 7elo! {answer}",
    ],
    Intent.REQUEST: [
        "Akid habibi! {answer}",
        "Tab3an, khalas 3mol hek: {answer}",
        "Inshallah bsa3dak! {answer}",
        "Eh ma3 kell sourour! {answer}",
    ],
    Intent.UNKNOWN: [
        "Hmm, ma fhemet ktir. Fi2ik t3id el soual?",
        "Sorry, ma 2dert efham. Shu 2asdak?",
        "Msh fahemha ktir, bas khalline 7awel... {answer}",
    ],
}

ARABIC_RESPONSES = {
    Intent.GREETING: [
        "أهلاً وسهلاً! كيفك اليوم؟ 🌲",
        "مرحبا! شو أخبارك؟",
        "هلا والله! أهلاً فيك",
        "أهلاً حبيبي! كيف حالك؟",
        "مرحبا! أنا هون لمساعدتك، شو بدك؟",
        "يا هلا! كيفك إنشاالله منيح؟",
        "أهلاً! نورت، شو بقدر ساعدك؟",
    ],
    Intent.FAREWELL: [
        "مع السلامة! الله معك 👋",
        "باي! إنشاالله منحكي بكرا",
        "يلا باي! تصبح على خير",
        "الله يحفظك! مع السلامة",
        "يلا مع السلامة! إنشاالله منحكي قريب 👋",
    ],
    Intent.QUESTION: [
        "سؤال منيح! خليني فكر شوي... {answer}",
        "هلأ بجاوبك... {answer}",
        "يعني، {answer}",
        "أه سؤال حلو! {answer}",
        "منيح إنك سألت! {answer}",
        "سؤال حلو والله! {answer}",
    ],
    Intent.THANKS: [
        "عفواً حبيبي! أي وقت 🌲",
        "تكرم! أنا هون دايماً",
        "لا شكر على واجب!",
        "أهلاً فيك! أي خدمة",
        "عفواً! بتشرفنا فيك",
    ],
    Intent.COMPLAINT: [
        "معلش، حقك عليي. شو فيني ساعدك؟",
        "آسف! خليني حاول أحسن المرة الجاية",
        "والله آسف، خليني جرب مرة تانية",
        "ما تكون زعلان، رح حاول أحسن",
    ],
    Intent.FEEDBACK: [
        "شكراً عالملاحظات! بتساعدني كتير",
        "ميرسي! رح حاول إتحسن",
        "منيح إنك قلتلي، شكراً!",
    ],
    Intent.CHITCHAT: [
        "والله! {answer}",
        "إيه صح، {answer}",
        "يعني، {answer}",
        "آه منيح! {answer}",
        "هلأ بقلك شي... {answer}",
        "آه كتير حلو! {answer}",
    ],
    Intent.REQUEST: [
        "أكيد حبيبي! {answer}",
        "طبعاً! {answer}",
        "إنشاالله بساعدك! {answer}",
        "إيه مع كل سرور! {answer}",
    ],
    Intent.UNKNOWN: [
        "ما فهمت كتير، فيك تعيد السؤال؟",
        "معلش ما قدرت إفهم، شو قصدك؟",
        "مش فاهمها كتير، بس خليني حاول... {answer}",
    ],
}

EN_TO_ARABIZI = {
    "hello": "marhaba",
    "hi": "ahla",
    "yes": "eh",
    "no": "la2",
    "good": "mni7",
    "very": "ktir",
    "thank": "shukran",
    "thanks": "merci",
    "please": "3mol ma3rouf",
    "sorry": "sorry ya zalameh",
    "ok": "tamam",
    "okay": "tamam",
    "what": "shu",
    "how": "kif",
    "why": "lesh",
    "where": "wen",
    "when": "emta",
    "who": "min",
    "now": "halla2",
    "today": "lyom",
    "tomorrow": "bukra",
    "yesterday": "mbereh",
    "friend": "sa7be",
    "love": "7ob",
    "beautiful": "7elo",
    "food": "akl",
    "water": "may",
    "come": "ta3a",
    "go": "rou7",
    "want": "badde",
    "know": "ba3ref",
    "think": "bfakker",
    "see": "shuf",
    "much": "ktir",
    "big": "kbir",
    "small": "zghir",
    "new": "jdid",
    "old": "2adim",
}

class CedarChatbot:
    DEFAULT_MODEL = "facebook/blenderbot-400M-distill"

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_turns: int = 10,
        max_length: int = 128,
        device: str = "cpu",
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length
        self.device = device

        self.normalizer = ArabiziNormalizer()
        self.lang_detector = LanguageDetector()
        self.intent_classifier = IntentClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.memory = ConversationMemory(max_turns=max_turns)

        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        start_time = time.time()

        session = self.memory.get_or_create_session(session_id, user_id)
        sid = session.session_id

        lang_result = self.lang_detector.detect(message)

        normalized = message
        if lang_result.language == Language.LEBANESE_ARABIZI:
            normalized = self.normalizer.normalize(message)

        intent_result = self.intent_classifier.classify(message)

        sentiment_result = self.sentiment_analyzer.analyze(message)

        user_msg = self.memory.add_message(
            sid, "user", normalized,
            metadata={
                "original": message,
                "language": lang_result.language.value,
                "intent": intent_result.intent.value,
                "sentiment": sentiment_result.label.value,
            }
        )

        if lang_result.language in (Language.ARABIC_MSA, Language.LEBANESE_ARABIC):
            raw_response = self._generate_english_fallback(normalized)
            response_text = self._to_arabic_response(raw_response, intent_result.intent)

        elif lang_result.language == Language.LEBANESE_ARABIZI:
            raw_response = self._generate_english_fallback(normalized)
            response_text = self._to_arabizi_response(raw_response, intent_result.intent)

        else:
            context = self.memory.get_context(sid)
            response_text = self._generate(context, message)

        self.memory.add_message(sid, "assistant", response_text)

        elapsed = round((time.time() - start_time) * 1000, 1)

        return ChatResponse(
            response=response_text,
            session_id=sid,
            message_id=user_msg.message_id,
            metadata={
                "detected_language": lang_result.language.value,
                "language_confidence": lang_result.confidence,
                "normalized_text": normalized if normalized != message else None,
                "intent": intent_result.intent.value,
                "intent_confidence": intent_result.confidence,
                "sentiment": {
                    "label": sentiment_result.label.value,
                    "score": sentiment_result.score,
                },
                "model": self.model_name,
                "response_time_ms": elapsed,
            },
        )

    def _to_arabic_response(self, raw_response: str, intent: Intent) -> str:

        templates = ARABIC_RESPONSES.get(intent, ARABIC_RESPONSES[Intent.CHITCHAT])

        if intent in (Intent.GREETING, Intent.FAREWELL, Intent.THANKS, Intent.COMPLAINT, Intent.FEEDBACK):
            return random.choice(templates)

        template = random.choice(templates)
        if "{answer}" in template:
            if raw_response:
                return template.format(answer=raw_response)
            else:
                fallback = random.choice(ARABIC_RESPONSES.get(Intent.UNKNOWN, ["ما فهمت، فيك تعيد؟"]))
                return fallback
        return template

    def _to_arabizi_response(self, raw_response: str, intent: Intent) -> str:
        templates = ARABIZI_RESPONSES.get(intent, ARABIZI_RESPONSES[Intent.CHITCHAT])

        if intent in (Intent.GREETING, Intent.FAREWELL, Intent.THANKS, Intent.COMPLAINT, Intent.FEEDBACK):
            return random.choice(templates)

        template = random.choice(templates)
        if "{answer}" in template:
            if raw_response:
                arabizi_answer = self._sprinkle_arabizi(raw_response)
                return template.format(answer=arabizi_answer)
            else:
                fallback = random.choice(ARABIZI_RESPONSES.get(Intent.UNKNOWN, ["Ma fhemet, fi2ik t3id?"]))
                return fallback
        return template

    def _sprinkle_arabizi(self, text: str) -> str:

        words = text.split()
        result = []

        for word in words:
            lower = word.lower().strip(".,!?;:")
            if lower in EN_TO_ARABIZI and random.random() > 0.5:
                punct = ""
                if word and word[-1] in ".,!?;:":
                    punct = word[-1]
                result.append(EN_TO_ARABIZI[lower] + punct)
            else:
                result.append(word)

        return " ".join(result)
    
    def _generate(self, context: str, message: str) -> str:
        try:
            if context:
                input_text = f"{context} \n {message}"
            else:
                input_text = message

            inputs = self._tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(self.device)

            outputs = self._model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

            response = self._tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            return response.strip()

        except Exception as e:
            logger.warning(f"Generation error: {e}")
            return "I'm sorry, I encountered an error. Could you try again?"

    def _generate_english_fallback(self, message: str) -> str:
        try:
            inputs = self._tokenizer(
                message,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(self.device)

            outputs = self._model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

            response = self._tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            return response.strip()

        except Exception:
            return ""

    def get_history(self, session_id: str) -> list:
        return self.memory.get_history(session_id)

    def submit_feedback(
        self, session_id: str, message_id: str, rating: int, comment: str = ""
    ):
        self.memory.add_feedback(session_id, message_id, rating, comment)

    def get_stats(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            **self.memory.get_stats(),
        }