import logging
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class MultilingualGenerator:

    DEFAULT_MODEL = "bigscience/mt0-small"

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
            return self._clean(text)
        except Exception as e:
            logger.warning(f"Generation error: {e}")
            return ""

    @staticmethod
    def _clean(text: str) -> str:
        import re
        text = re.sub(r"<extra_id_\d+>", "", text)
        text = text.replace("<pad>", "").replace("</s>", "").strip()
        return text

    def generate_arabic(self, user_message: str, context_fact: str = "") -> str:
        if context_fact:
            prompt = (
                f"بناءً على هذه المعلومة: \"{context_fact}\" "
                f"أجب بإيجاز وبأسلوب ودود على السؤال: {user_message}"
            )
        else:
            prompt = f"أجب بإيجاز وبأسلوب ودود على هذه الرسالة بالعربية: {user_message}"
        return self.generate(prompt)

    def generate_arabizi(self, user_message: str, normalized: str = "", context_fact: str = "") -> str:
        if context_fact:
            prompt = (
                f"Based on this fact: \"{context_fact}\" "
                f"write a short friendly reply in Lebanese Arabizi "
                f"(Arabic written with Latin letters and numbers like 7, 3, 2) "
                f"to: {user_message}"
            )
        else:
            prompt = (
                f"Write a short friendly reply in Lebanese Arabizi "
                f"(Arabic written with Latin letters and numbers) to: {user_message}"
            )
        return self.generate(prompt)