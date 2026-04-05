import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.normalizer import ArabiziNormalizer
from src.language_detector import LanguageDetector, Language
from src.intent_classifier import IntentClassifier
from src.sentiment import SentimentAnalyzer
from src.memory import ConversationMemory

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    response: str
    session_id: str
    message_id: str = ""
    metadata: dict = field(default_factory=dict)


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

        context = self.memory.get_context(sid)
        response_text = self._generate(context, normalized)

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
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

            response = self._tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            return response.strip()

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I'm sorry, I encountered an error. Could you try again?"

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