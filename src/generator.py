import logging
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class MultilingualGenerator:

    DEFAULT_MODEL = "google/mt5-small"

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu", max_length: int = 96):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._load()

    def _load(self):
        try:
            logger.info(f"Loading multilingual decoder: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info("Multilingual decoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load multilingual decoder: {e}")
            raise

    def generate(self, prompt: str, max_length: Optional[int] = None) -> str:
        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=max_length or self.max_length,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    do_sample=False,
                )

            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return text
        except Exception as e:
            logger.warning(f"Generation error: {e}")
            return ""

    def generate_arabic(self, user_message: str, context_fact: str = "") -> str:
        if context_fact:
            prompt = (
                f"أجب على السؤال التالي بالعربية بأسلوب طبيعي ومفيد. "
                f"معلومة مرجعية: {context_fact} "
                f"السؤال: {user_message} "
                f"الجواب:"
            )
        else:
            prompt = (
                f"أجب على هذه الرسالة بالعربية بأسلوب ودود: {user_message} الجواب:"
            )
        return self.generate(prompt)

    def generate_arabizi(self, user_message: str, normalized: str = "", context_fact: str = "") -> str:
        if context_fact:
            prompt = (
                f"Reply in Lebanese Arabizi (Latin letters with numbers like 7, 3, 2). "
                f"Keep it casual and natural. "
                f"Reference fact: {context_fact} "
                f"User said: {user_message} "
                f"Reply:"
            )
        else:
            prompt = (
                f"Reply in Lebanese Arabizi (Latin letters with numbers) to: {user_message} Reply:"
            )
        return self.generate(prompt)