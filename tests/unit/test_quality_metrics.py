"""
Unit tests for Scripts/CleanlinessMetrics/compute_metrics.py.

Covers every public function with a happy-path case and at least one
alternative or edge case (empty input, boundary values, known dirty text).
"""

import pytest

from compute_metrics import (
    classify_score,
    compute_quality_score,
    fragmented_words_ratio,
    header_integrity_ratio,
    score_fragmentation,
    score_lines,
    score_structure,
    short_lines_ratio,
)


# short_lines_ratio
class TestShortLinesRatio:
    def test_returns_zero_for_clean_text(self):
        # A text with long lines should have no short lines.
        text = "Primera línea larga\nSegunda línea larga\nTercera línea bien formada"
        assert short_lines_ratio(text) == 0.0

    def test_returns_one_when_all_lines_are_short(self):
        # If every line is short, the ratio must be 1.0.
        text = "ab\ncd\nef"
        assert short_lines_ratio(text) == 1.0

    def test_correct_proportion_mixed_text(self):
        # With 2 short lines out of 4 total, the ratio must be exactly 0.5.
        text = "ab\ncd\nEsta línea es larga\nOtra línea larga aquí"
        ratio = short_lines_ratio(text)
        assert ratio == pytest.approx(0.5, abs=1e-6)

    def test_empty_text_returns_zero(self):
        # An empty text has no lines, so the ratio must be 0.0.
        assert short_lines_ratio("") == 0.0

    def test_ignores_blank_lines(self):
        # Blank lines are not counted as content lines.
        text = "ab\n\nEsta es una línea larga"
        ratio = short_lines_ratio(text)
        assert ratio == pytest.approx(0.5, abs=1e-6)


# fragmented_words_ratio
class TestFragmentedWordsRatio:
    def test_returns_zero_for_normal_text(self):
        # Normal legal text should have no fragmented words.
        text = "Este es un texto legal normal sin fragmentación alguna"
        assert fragmented_words_ratio(text) == 0.0

    def test_detects_spaced_out_word(self):
        # 'A R T I C U L O' is a classic OCR artefact in scanned legal documents.
        text = "A R T I C U L O texto normal aquí para contexto"
        ratio = fragmented_words_ratio(text)
        assert ratio > 0.0

    def test_empty_text_returns_zero(self):
        # An empty text has no words, so the ratio must be 0.0.
        assert fragmented_words_ratio("") == 0.0


# header_integrity_ratio
class TestHeaderIntegrityRatio:
    def test_valid_article_headers_detected(self):
        # Well-formed article headers must be detected by the ratio function.
        text = (
            "ARTÍCULO 1. Primera disposición general.\n"
            "ARTÍCULO 2. Segunda disposición general.\n"
            "ARTÍCULO 3. Tercera disposición general."
        )
        ratio = header_integrity_ratio(text)
        assert ratio > 0.0

    def test_returns_one_when_all_headers_are_intact(self):
        # If all headers are well-formed, the ratio must be 1.0.
        text = "ARTÍCULO 1. Contenido\nARTÍCULO 2. Contenido\nARTÍCULO 3. Contenido"
        ratio = header_integrity_ratio(text)
        assert ratio == pytest.approx(1.0, abs=1e-6)

    def test_returns_zero_for_text_without_legal_headers(self):
        # Text without any legal headers must return 0.0.
        text = "Texto sin encabezados legales estructurados en absoluto."
        assert header_integrity_ratio(text) == 0.0

    def test_paragrafo_headers_detected(self):
        # Paragraph headers must also be recognised as legal structure.
        text = "PARÁGRAFO 1. Disposición transitoria aplicable."
        ratio = header_integrity_ratio(text)
        assert ratio > 0.0


# score_lines / score_fragmentation / score_structure
class TestScoringHelpers:
    @pytest.mark.parametrize(
        "ratio, expected",
        [(0.0, 30), (0.02, 25), (0.05, 15), (0.10, 0)],
    )
    def test_score_lines_boundaries(self, ratio, expected):
        # Each short-line ratio range must produce the correct score.
        assert score_lines(ratio) == expected

    @pytest.mark.parametrize(
        "ratio, expected",
        [(0.0, 30), (0.003, 20), (0.01, 10), (0.05, 0)],
    )
    def test_score_fragmentation_boundaries(self, ratio, expected):
        # Each fragmentation ratio range must produce the correct score.
        assert score_fragmentation(ratio) == expected

    @pytest.mark.parametrize(
        "ratio, expected",
        [(1.0, 25), (0.95, 20), (0.80, 10), (0.50, 0)],
    )
    def test_score_structure_boundaries(self, ratio, expected):
        # Each header integrity ratio range must produce the correct score.
        assert score_structure(ratio) == expected


# classify_score
class TestClassifyScore:
    @pytest.mark.parametrize("score", [85, 90, 100])
    def test_high_quality(self, score):
        # Scores of 85 or above must be classified as HIGH.
        assert classify_score(score) == "HIGH"

    @pytest.mark.parametrize("score", [70, 75, 84])
    def test_medium_quality(self, score):
        # Scores between 70 and 84 must be classified as MEDIUM.
        assert classify_score(score) == "MEDIUM"

    @pytest.mark.parametrize("score", [50, 60, 69])
    def test_low_quality(self, score):
        # Scores between 50 and 69 must be classified as LOW.
        assert classify_score(score) == "LOW"

    @pytest.mark.parametrize("score", [0, 25, 49])
    def test_defective_quality(self, score):
        # Scores below 50 must be classified as DEFECTIVE.
        assert classify_score(score) == "DEFECTIVE"


# compute_quality_score
class TestComputeQualityScore:
    def test_returns_expected_keys(self):
        # The result dictionary must contain all keys defined in the interface.
        result = compute_quality_score("ARTÍCULO 1. Texto de prueba bien formado.")
        expected_keys = {
            "line_ratio",
            "fragmented_ratio",
            "header_integrity",
            "quality_score",
            "quality_status",
            "version",
        }
        assert expected_keys.issubset(result.keys())

    def test_version_field_is_v1(self):
        # The version field must indicate the current version of the metrics algorithm.
        result = compute_quality_score("Cualquier texto")
        assert result["version"] == "V1"

    def test_quality_status_is_valid_category(self):
        # The quality_status field must be one of the four valid categories.
        result = compute_quality_score("ARTÍCULO 1. Norma de orden público.")
        assert result["quality_status"] in {"HIGH", "MEDIUM", "LOW", "DEFECTIVE"}

    def test_clean_legal_text_scores_higher_than_dirty_text(self):
        # Well-structured legal text must score higher than noisy/dirty text.
        clean = (
            "ARTÍCULO 1. Esta norma establece disposiciones de orden público.\n"
            "ARTÍCULO 2. Las presentes reglas aplican a todo el territorio nacional."
        )
        dirty = "ab\ncd\nef\nA R T I C U L O\nwww.spam.com\nxy\nzw\n12\n34"
        assert compute_quality_score(clean)["quality_score"] >= compute_quality_score(dirty)["quality_score"]

    def test_ratios_are_between_zero_and_one(self):
        # All intermediate ratios must be within the [0.0, 1.0] range.
        result = compute_quality_score("ARTÍCULO 1. Texto con estructura legal válida.")
        assert 0.0 <= result["line_ratio"] <= 1.0
        assert 0.0 <= result["fragmented_ratio"] <= 1.0
        assert 0.0 <= result["header_integrity"] <= 1.0

    def test_empty_text_does_not_raise(self):
        # An empty text must not raise an exception and must return a valid integer score.
        result = compute_quality_score("")
        assert isinstance(result["quality_score"], int)
