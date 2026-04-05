import pytest
from src.normalizer import ArabiziNormalizer, normalize_arabizi


class TestArabiziNormalizer:

    def setup_method(self):
        self.normalizer = ArabiziNormalizer()


    def test_greeting_kifak(self):
        assert self.normalizer.normalize("kifak") == "كيفك"

    def test_greeting_keefak(self):
        assert self.normalizer.normalize("keefak") == "كيفك"

    def test_marhaba(self):
        assert self.normalizer.normalize("marhaba") == "مرحبا"

    def test_mar7aba_with_numeral(self):
        assert self.normalizer.normalize("mar7aba") == "مرحبا"

    def test_habibi_with_7(self):
        assert self.normalizer.normalize("7abibi") == "حبيبي"

    def test_yalla(self):
        assert self.normalizer.normalize("yalla") == "يلا"

    def test_khalas(self):
        assert self.normalizer.normalize("khalas") == "خلص"

    def test_5alas_with_numeral(self):
        assert self.normalizer.normalize("5alas") == "خلص"

    def test_inshallah(self):
        assert self.normalizer.normalize("inshallah") == "إنشاالله"


    def test_keefak_ya_zalame(self):
        result = self.normalizer.normalize("keefak ya zalame")
        assert "كيفك" in result
        assert "زلمة" in result

    def test_shu_3am_ta3mel(self):
        result = self.normalizer.normalize("shu 3am ta3mel")
        assert "شو" in result
        assert "عم" in result
        assert "تعمل" in result


    def test_arabic_passthrough(self):
        arabic = "كيف حالك؟"
        assert self.normalizer.normalize(arabic) == arabic

    def test_empty_string(self):
        assert self.normalizer.normalize("") == ""

    def test_whitespace_only(self):
        assert self.normalizer.normalize("   ") == "   "


    def test_preserves_question_mark(self):
        result = self.normalizer.normalize("kifak?")
        assert result.endswith("?") or result.endswith("؟")

    def test_preserves_exclamation(self):
        result = self.normalizer.normalize("yalla!")
        assert "!" in result


    def test_3_becomes_ain(self):
        result = self.normalizer.normalize("3am")
        assert "عم" == result or "ع" in result

    def test_7_becomes_haa(self):
        result = self.normalizer.normalize("7abibi")
        assert "حبيبي" == result or "ح" in result


    def test_convenience_function(self):
        assert normalize_arabizi("kifak") == "كيفك"


    def test_add_custom_phrase(self):
        self.normalizer.add_phrase("wagif", "واقف")
        assert self.normalizer.normalize("wagif") == "واقف"


    def test_mapping_stats(self):
        stats = self.normalizer.get_mapping_stats()
        assert stats["phrase_count"] > 50
        assert stats["multi_char_rules"] > 10
        assert stats["single_char_rules"] > 15


class TestEdgeCases:

    def setup_method(self):
        self.normalizer = ArabiziNormalizer()

    def test_mixed_case(self):
        result = self.normalizer.normalize("KIFAK")
        assert "كيفك" in result

    def test_numbers_only(self):
        result = self.normalizer.normalize("12345")
        assert result is not None

    def test_special_characters(self):
        result = self.normalizer.normalize("@#$%")
        assert result is not None

    def test_very_long_input(self):
        long_text = "kifak " * 1000
        result = self.normalizer.normalize(long_text)
        assert "كيفك" in result