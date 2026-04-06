import pytest
from src.dialect_detector import ArabicDialectDetector, ArabicDialect, detect_dialect


class TestArabicDialectDetector:

    def setup_method(self):
        self.detector = ArabicDialectDetector()

    def test_lebanese_greeting(self):
        result = self.detector.detect("كيفك؟ شو عم تعمل هلأ؟")
        assert result.dialect == ArabicDialect.LEBANESE

    def test_lebanese_conversation(self):
        result = self.detector.detect("منيح كتير! بدي روح عالبيت بكرا")
        assert result.dialect == ArabicDialect.LEBANESE

    def test_lebanese_expression(self):
        result = self.detector.detect("يلا خلص بدنا نروح هلأ")
        assert result.dialect == ArabicDialect.LEBANESE

    def test_lebanese_arabizi(self):
        result = self.detector.detect("keefak ya zalame? shu 3am ta3mel?")
        assert result.dialect == ArabicDialect.LEBANESE

    def test_lebanese_arabizi_2(self):
        result = self.detector.detect("mni7 ktir habibi, yalla bye")
        assert result.dialect == ArabicDialect.LEBANESE

    def test_egyptian_greeting(self):
        result = self.detector.detect("إزيك؟ عامل إيه دلوقتي؟")
        assert result.dialect == ArabicDialect.EGYPTIAN

    def test_egyptian_conversation(self):
        result = self.detector.detect("أنا عايز أروح البيت بقى خلاص")
        assert result.dialect == ArabicDialect.EGYPTIAN

    def test_egyptian_expression(self):
        result = self.detector.detect("كده كويس أوي يا باشا")
        assert result.dialect == ArabicDialect.EGYPTIAN

    def test_egyptian_arabizi(self):
        result = self.detector.detect("ezayak ya basha? 3amel eih delwa2ti?")
        assert result.dialect == ArabicDialect.EGYPTIAN

    def test_egyptian_arabizi_2(self):
        result = self.detector.detect("ana 3ayez 2awy a7aga kaman")
        assert result.dialect == ArabicDialect.EGYPTIAN

    def test_gulf_greeting(self):
        result = self.detector.detect("شلونك؟ وش تبي الحين؟")
        assert result.dialect == ArabicDialect.GULF

    def test_gulf_conversation(self):
        result = self.detector.detect("أبغى أروح البيت الحين يا خوي")
        assert result.dialect == ArabicDialect.GULF

    def test_gulf_expression(self):
        result = self.detector.detect("حياك الله يا وجه الخير")
        assert result.dialect == ArabicDialect.GULF

    def test_gulf_arabizi(self):
        result = self.detector.detect("shlonak ya 5ooy? wesh tabi el7een?")
        assert result.dialect == ArabicDialect.GULF


    def test_msa_formal(self):
        result = self.detector.detect("ينبغي علينا أن نتعلم من التجارب التي مررنا بها")
        assert result.dialect == ArabicDialect.MSA

    def test_msa_news(self):
        result = self.detector.detect("بالإضافة إلى ذلك يجب أن نأخذ بالتالي هذه النقاط")
        assert result.dialect == ArabicDialect.MSA

    def test_empty_input(self):
        result = self.detector.detect("")
        assert result.dialect == ArabicDialect.UNKNOWN

    def test_confidence_range(self):
        result = self.detector.detect("كيفك شو عم تعمل")
        assert 0.0 <= result.confidence <= 1.0

    def test_markers_found(self):
        result = self.detector.detect("كيفك شو عم تعمل منيح")
        assert len(result.markers_found.get("lebanese", [])) > 0

    def test_scores_present(self):
        result = self.detector.detect("إزيك عامل إيه")
        assert "egyptian" in result.scores

    def test_dialect_label(self):
        result = self.detector.detect("شلونك وش تبي")
        assert "Gulf" in result.dialect_label or result.dialect == ArabicDialect.GULF

    def test_get_dialect_info_lebanese(self):
        info = self.detector.get_dialect_info(ArabicDialect.LEBANESE)
        assert info["name"] == "Lebanese Arabic"
        assert info["region"] == "Lebanon"

    def test_get_dialect_info_egyptian(self):
        info = self.detector.get_dialect_info(ArabicDialect.EGYPTIAN)
        assert info["name"] == "Egyptian Arabic"

    def test_get_dialect_info_gulf(self):
        info = self.detector.get_dialect_info(ArabicDialect.GULF)
        assert "Saudi" in info["region"]