import pytest
from src.language_detector import LanguageDetector, Language
from src.intent_classifier import IntentClassifier, Intent
from src.sentiment import SentimentAnalyzer, SentimentLabel


class TestLanguageDetector:

    def setup_method(self):
        self.detector = LanguageDetector()

    def test_detect_english(self):
        result = self.detector.detect("Hello, how are you doing today?")
        assert result.language == Language.ENGLISH

    def test_detect_arabic(self):
        result = self.detector.detect("كيف حالك اليوم؟")
        assert result.language in (Language.ARABIC_MSA, Language.LEBANESE_ARABIC)

    def test_detect_arabizi(self):
        result = self.detector.detect("keefak ya zalame shu 3am ta3mel")
        assert result.language == Language.LEBANESE_ARABIZI

    def test_detect_lebanese_arabic(self):
        result = self.detector.detect("شو عم تعمل هلأ؟ كيفك منيح؟")
        assert result.language == Language.LEBANESE_ARABIC

    def test_empty_input(self):
        result = self.detector.detect("")
        assert result.language == Language.UNKNOWN

    def test_confidence_range(self):
        result = self.detector.detect("Hello world")
        assert 0.0 <= result.confidence <= 1.0

    def test_arabizi_with_numerals(self):
        result = self.detector.detect("7abibi 3am ba3ref")
        assert result.language == Language.LEBANESE_ARABIZI


class TestIntentClassifier:

    def setup_method(self):
        self.classifier = IntentClassifier()

    def test_greeting_english(self):
        result = self.classifier.classify("Hello! How are you?")
        assert result.intent == Intent.GREETING

    def test_greeting_arabic(self):
        result = self.classifier.classify("مرحبا كيف حالك")
        assert result.intent == Intent.GREETING

    def test_question(self):
        result = self.classifier.classify("What is machine learning?")
        assert result.intent == Intent.QUESTION

    def test_question_arabic(self):
        result = self.classifier.classify("شو هو التعلم العميق؟")
        assert result.intent == Intent.QUESTION

    def test_thanks(self):
        result = self.classifier.classify("Thank you so much!")
        assert result.intent == Intent.THANKS

    def test_farewell(self):
        result = self.classifier.classify("Goodbye, see you later!")
        assert result.intent == Intent.FAREWELL

    def test_empty_input(self):
        result = self.classifier.classify("")
        assert result.intent == Intent.UNKNOWN


class TestSentimentAnalyzer:

    def setup_method(self):
        self.analyzer = SentimentAnalyzer()

    def test_positive_english(self):
        result = self.analyzer.analyze("This is great and amazing!")
        assert result.label == SentimentLabel.POSITIVE
        assert result.score > 0

    def test_negative_english(self):
        result = self.analyzer.analyze("This is terrible and awful")
        assert result.label == SentimentLabel.NEGATIVE
        assert result.score < 0

    def test_neutral(self):
        result = self.analyzer.analyze("The weather today is moderate")
        assert result.label == SentimentLabel.NEUTRAL

    def test_negation(self):
        result = self.analyzer.analyze("This is not good")
        assert result.label == SentimentLabel.NEGATIVE

    def test_intensifier(self):
        result = self.analyzer.analyze("This is very good")
        assert result.label == SentimentLabel.POSITIVE
        r2 = self.analyzer.analyze("This is good")
        assert result.score >= r2.score 

    def test_arabic_positive(self):
        result = self.analyzer.analyze("ممتاز رائع حلو")
        assert result.label == SentimentLabel.POSITIVE

    def test_arabic_negative(self):
        result = self.analyzer.analyze("سيء بشع مقرف")
        assert result.label == SentimentLabel.NEGATIVE

    def test_empty_input(self):
        result = self.analyzer.analyze("")
        assert result.label == SentimentLabel.NEUTRAL