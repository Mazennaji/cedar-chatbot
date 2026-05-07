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


KNOWLEDGE_BASE = {
    "what is nlp": (
        "NLP stands for Natural Language Processing — a field of AI that helps computers "
        "understand, interpret, and generate human language. It powers things like chatbots, "
        "translation, speech recognition, and sentiment analysis."
    ),
    "what is natural language processing": (
        "Natural Language Processing (NLP) is a branch of artificial intelligence focused on "
        "enabling computers to understand and work with human language. It includes tasks like "
        "text classification, machine translation, named entity recognition, and more."
    ),
    "what is ai": (
        "AI (Artificial Intelligence) is the simulation of human intelligence by machines. "
        "It includes subfields like machine learning, deep learning, NLP, and computer vision."
    ),
    "what is artificial intelligence": (
        "Artificial Intelligence is the science of building machines that can perform tasks "
        "that normally require human intelligence — like reasoning, learning, and understanding language."
    ),
    "what is ml": (
        "ML stands for Machine Learning — a subset of AI where models learn patterns from data "
        "instead of being explicitly programmed. Examples include decision trees, neural networks, "
        "and support vector machines."
    ),
    "what is machine learning": (
        "Machine Learning is a branch of AI that allows systems to learn from data and improve "
        "over time without being explicitly programmed. It's used in recommendations, fraud detection, "
        "image recognition, and much more."
    ),
    "what is deep learning": (
        "Deep Learning is a subset of machine learning that uses neural networks with many layers "
        "to learn from large amounts of data. It powers image recognition, speech synthesis, and LLMs."
    ),
    "what is a neural network": (
        "A neural network is a computational model inspired by the human brain. It consists of "
        "layers of interconnected nodes (neurons) that learn to recognize patterns from data."
    ),
    "what is a large language model": (
        "A Large Language Model (LLM) is a deep learning model trained on massive text datasets "
        "to understand and generate human language. Examples include GPT-4, Claude, and Gemini."
    ),
    "what is llm": (
        "LLM stands for Large Language Model — an AI model trained on vast amounts of text to "
        "understand and generate language. Examples: GPT-4, Claude, Gemini, LLaMA."
    ),
    "what is a transformer": (
        "A Transformer is a neural network architecture introduced in 2017 that uses self-attention "
        "to process sequences in parallel. It is the foundation of modern LLMs like BERT and GPT."
    ),
    "what is bert": (
        "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained NLP model "
        "by Google. It reads text bidirectionally and is widely used for classification and Q&A."
    ),
    "what is gpt": (
        "GPT (Generative Pre-trained Transformer) is a family of large language models by OpenAI. "
        "GPT models generate human-like text and are the basis for ChatGPT."
    ),
    "what is cedar": (
        "Cedar is this trilingual AI chatbot! It supports English, Modern Standard Arabic, and "
        "Lebanese dialect including Arabizi. It uses BlenderBot for conversation and custom NLP "
        "modules for language detection, intent classification, and sentiment analysis."
    ),
    "tell me about lebanon": (
        "Lebanon is a small country in the Middle East on the Mediterranean coast. It's known for "
        "its rich history, diverse culture, cedar forests (the national symbol 🌲), and cuisine "
        "like hummus, tabbouleh, and kibbeh. Beirut is its capital."
    ),
    "what is arabizi": (
        "Arabizi (also called Franco-Arabic) is Arabic written in Latin script with numbers "
        "replacing sounds that don't exist in English. For example: 7 = ح, 3 = ع, 2 = ء. "
        "It's widely used by Lebanese and Arab youth online."
    ),
    "what is sentiment analysis": (
        "Sentiment analysis is an NLP task that determines the emotional tone of text — whether "
        "it's positive, negative, or neutral. It's used in product reviews, social media monitoring, "
        "and customer feedback systems."
    ),
    "what is intent classification": (
        "Intent classification is an NLP task that identifies what a user wants to accomplish from "
        "their message — like greeting, asking a question, or making a request. It's core to "
        "chatbot and virtual assistant systems."
    ),
    "what is named entity recognition": (
        "Named Entity Recognition (NER) is an NLP task that identifies and classifies named entities "
        "in text — such as people, organizations, locations, and dates."
    ),
    "what is ner": (
        "NER stands for Named Entity Recognition — an NLP technique that detects names, places, "
        "organizations, and other entities in text."
    ),
    "what is tokenization": (
        "Tokenization is the process of splitting text into smaller units called tokens — usually "
        "words or subwords. It's one of the first steps in any NLP pipeline."
    ),
    "what is a chatbot": (
        "A chatbot is a software application that simulates human conversation through text or voice. "
        "Modern chatbots use NLP and machine learning to understand context and generate responses."
    ),
    "what is blenderbot": (
        "BlenderBot is an open-domain chatbot by Meta AI, trained on large conversational datasets. "
        "Cedar uses BlenderBot-400M-distill as its base conversation engine for English responses."
    ),
}

_KB_KEYWORD_MAP = [
    (["nlp", "natural language processing"],         "what is nlp"),
    (["machine learning", " ml "],                   "what is machine learning"),
    (["deep learning"],                              "what is deep learning"),
    (["neural network"],                             "what is a neural network"),
    (["transformer"],                                "what is a transformer"),
    (["bert"],                                       "what is bert"),
    (["gpt"],                                        "what is gpt"),
    (["llm", "large language model"],                "what is llm"),
    (["artificial intelligence", " ai "],            "what is ai"),
    (["sentiment analysis"],                         "what is sentiment analysis"),
    (["intent classif"],                             "what is intent classification"),
    (["arabizi"],                                    "what is arabizi"),
    (["cedar chatbot", "what is cedar"],             "what is cedar"),
    (["tokeniz"],                                    "what is tokenization"),
    (["named entity", " ner "],                      "what is named entity recognition"),
    (["blenderbot"],                                 "what is blenderbot"),
    (["lebanon"],                                    "tell me about lebanon"),
    (["chatbot"],                                    "what is a chatbot"),
]


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
    "hello": "marhaba", "hi": "ahla", "yes": "eh", "no": "la2",
    "good": "mni7", "very": "ktir", "thank": "shukran", "thanks": "merci",
    "please": "3mol ma3rouf", "sorry": "sorry ya zalameh", "ok": "tamam",
    "okay": "tamam", "what": "shu", "how": "kif", "why": "lesh",
    "where": "wen", "when": "emta", "who": "min", "now": "halla2",
    "today": "lyom", "tomorrow": "bukra", "yesterday": "mbereh",
    "friend": "sa7be", "love": "7ob", "beautiful": "7elo", "food": "akl",
    "water": "may", "come": "ta3a", "go": "rou7", "want": "badde",
    "know": "ba3ref", "think": "bfakker", "see": "shuf", "much": "ktir",
    "big": "kbir", "small": "zghir", "new": "jdid", "old": "2adim",
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


    def _knowledge_lookup(self, message: str) -> Optional[str]:
        cleaned = message.lower().strip().rstrip("?!.,")

        if cleaned in KNOWLEDGE_BASE:
            return KNOWLEDGE_BASE[cleaned]

        for prefix in ("explain ", "describe ", "define ", "tell me about ",
                        "what do you know about ", "can you explain "):
            if cleaned.startswith(prefix):
                remainder = cleaned[len(prefix):]
                key = f"what is {remainder}"
                if key in KNOWLEDGE_BASE:
                    return KNOWLEDGE_BASE[key]

        padded = f" {cleaned} "
        for keywords, kb_key in _KB_KEYWORD_MAP:
            if any(kw in padded for kw in keywords):
                answer = KNOWLEDGE_BASE.get(kb_key)
                if answer:
                    return answer

        return None

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

        knowledge_answer = self._knowledge_lookup(message)

        if knowledge_answer:

            if lang_result.language in (Language.ARABIC_MSA, Language.LEBANESE_ARABIC):
                response_text = self._to_arabic_response(knowledge_answer, intent_result.intent)
            elif lang_result.language == Language.LEBANESE_ARABIZI:
                response_text = self._to_arabizi_response(knowledge_answer, intent_result.intent)
            else:
                response_text = knowledge_answer

        elif lang_result.language in (Language.ARABIC_MSA, Language.LEBANESE_ARABIC):
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
                "knowledge_hit": knowledge_answer is not None,
            },
        )

    def _to_arabic_response(self, raw_response: str, intent: Intent) -> str:
        templates = ARABIC_RESPONSES.get(intent, ARABIC_RESPONSES[Intent.CHITCHAT])
        if intent in (Intent.GREETING, Intent.FAREWELL, Intent.THANKS,
                      Intent.COMPLAINT, Intent.FEEDBACK):
            return random.choice(templates)
        template = random.choice(templates)
        if "{answer}" in template:
            if raw_response:
                return template.format(answer=raw_response)
            return random.choice(ARABIC_RESPONSES.get(Intent.UNKNOWN, ["ما فهمت، فيك تعيد؟"]))
        return template

    def _to_arabizi_response(self, raw_response: str, intent: Intent) -> str:
        templates = ARABIZI_RESPONSES.get(intent, ARABIZI_RESPONSES[Intent.CHITCHAT])
        if intent in (Intent.GREETING, Intent.FAREWELL, Intent.THANKS,
                      Intent.COMPLAINT, Intent.FEEDBACK):
            return random.choice(templates)
        template = random.choice(templates)
        if "{answer}" in template:
            if raw_response:
                arabizi_answer = self._sprinkle_arabizi(raw_response)
                return template.format(answer=arabizi_answer)
            return random.choice(ARABIZI_RESPONSES.get(Intent.UNKNOWN, ["Ma fhemet, fi2ik t3id?"]))
        return template

    def _sprinkle_arabizi(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            lower = word.lower().strip(".,!?;:")
            if lower in EN_TO_ARABIZI and random.random() > 0.5:
                punct = word[-1] if word and word[-1] in ".,!?;:" else ""
                result.append(EN_TO_ARABIZI[lower] + punct)
            else:
                result.append(word)
        return " ".join(result)

    def _generate(self, context: str, message: str) -> str:
        try:
            input_text = f"{context} \n {message}" if context else message
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
            return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
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
            return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception:
            return ""

    def get_history(self, session_id: str) -> list:
        return self.memory.get_history(session_id)

    def submit_feedback(self, session_id: str, message_id: str, rating: int, comment: str = ""):
        self.memory.add_feedback(session_id, message_id, rating, comment)

    def get_stats(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            **self.memory.get_stats(),
        }