"""
Unit tests for Scripts/ProcessHTMLs/text_normalization.py – normalize_body().

normalize_body applies 40+ regex transformations to clean legal text extracted
from HTML. Tests are grouped by transformation category so failures are easy
to trace back to a specific rule.
"""

import pytest

from text_normalization import normalize_body


# Base cleanup (control characters, whitespace)
class TestBaseCleanup:
    def test_removes_control_characters(self):
        # Non-printable control characters must be stripped from the text.
        text = "Texto\x00con\x08caracteres\x1Fde control"
        result = normalize_body(text)
        for char in ["\x00", "\x08", "\x1F"]:
            assert char not in result

    def test_replaces_non_breaking_space(self):
        # The non-breaking space (\xa0) must be replaced with a regular space.
        text = "palabra\xa0otra"
        result = normalize_body(text)
        assert "\xa0" not in result

    def test_collapses_tabs_to_space(self):
        # Tab characters must be converted to spaces.
        text = "columna1\t\tcolumna2"
        result = normalize_body(text)
        assert "\t" not in result

    def test_collapses_multiple_newlines_to_one(self):
        # Multiple consecutive newlines must be reduced to a single one.
        text = "línea1\n\n\nlínea2"
        result = normalize_body(text)
        assert "\n\n" not in result

    def test_removes_empty_parentheses(self):
        # Empty parentheses are extraction artefacts and must be removed.
        text = "Texto con paréntesis vacíos () aquí"
        result = normalize_body(text)
        assert "()" not in result

    def test_strips_leading_and_trailing_whitespace(self):
        # The result must not have leading or trailing whitespace.
        text = "   Contenido real.   "
        assert normalize_body(text) == normalize_body(text).strip()


# Base cleanup with apply_body_rules=False
class TestApplyBodyRulesFalse:
    def test_skips_body_rules_when_disabled(self):
        # With apply_body_rules=False, ARTICULO normalisation must NOT run.
        text = "A R T Í C U L O texto"
        result = normalize_body(text, apply_body_rules=False)
        assert "A R T" in result

    def test_base_cleanup_still_applied(self):
        # Even with body rules disabled, base cleanup must still be applied.
        text = "texto\xa0con\x00control"
        result = normalize_body(text, apply_body_rules=False)
        assert "\xa0" not in result
        assert "\x00" not in result


# Legal structure normalisation
class TestLegalStructureNormalisation:
    def test_collapses_spaced_out_articulo(self):
        # 'A R T Í C U L O' is an OCR artefact that must collapse to 'ARTICULO'.
        text = "A R T Í C U L O texto del artículo"
        result = normalize_body(text)
        assert "ARTICULO" in result

    def test_joins_articulo_number_split_on_next_line(self):
        # 'ARTICULO\n1°' occurs due to HTML line breaks and must be joined.
        text = "ARTICULO\n1° El presente artículo"
        result = normalize_body(text)
        assert "ARTICULO 1°" in result

    def test_joins_comma_continuation_line(self):
        # A comma at the end of a line followed by content must not have a newline.
        text = "El titular del cargo,\nquien sea designado"
        result = normalize_body(text)
        assert "cargo,\nquien" not in result

    def test_joins_ordinal_split_across_lines(self):
        # '1\n°' occurs when the ordinal symbol lands on the following line.
        text = "artículo 5\n°"
        result = normalize_body(text)
        assert "5\n°" not in result


# Money and decimal normalisation
class TestMoneyNormalisation:
    def test_normalises_underscore_decimal_separator(self):
        # '2_50' is an OCR artefact for a decimal value and must become '2.50'.
        text = "valor de 2_50 pesos"
        result = normalize_body(text)
        assert "2_50" not in result
        assert "2.50" in result

    def test_normalises_hyphen_decimal_separator(self):
        # '0-04' is another OCR decimal artefact and must become '0.04'.
        text = "tasa de 0-04 por ciento"
        result = normalize_body(text)
        assert "0-04" not in result
        assert "0.04" in result


# List item formatting
class TestListFormatting:
    def test_joins_lowercase_letter_list_marker_with_content(self):
        # 'a.\nTexto' must be joined into a single line as 'a. Texto'.
        text = "a.\nTexto del literal primero"
        result = normalize_body(text)
        assert "a.\nTexto" not in result

    def test_joins_parenthesised_roman_numeral_with_content(self):
        # '(i)\nTexto' must be joined into a single line as '(i) Texto'.
        text = "(i)\nPrimera condición aplicable"
        result = normalize_body(text)
        assert "(i)\nPrimera" not in result


# Punctuation clean-up
class TestPunctuationCleanup:
    def test_removes_standalone_punctuation_lines(self):
        # Lines containing only ':', ';', or ',' must be removed.
        text = "Primera parte\n:\nSegunda parte"
        result = normalize_body(text)
        lines = [ln for ln in result.split("\n") if ln.strip()]
        assert not any(ln.strip() in {":", ";", ","} for ln in lines)

    def test_removes_standalone_hyphen_lines(self):
        # Lines containing only hyphens are OCR artefacts and must be removed.
        text = "Encabezado\n---\nContenido"
        result = normalize_body(text)
        lines = [ln for ln in result.split("\n") if ln.strip()]
        assert not any(set(ln.strip()) <= {"-"} for ln in lines)

    def test_output_has_no_double_newlines(self):
        # After all transformations, no consecutive newlines must remain.
        text = "línea1\n\n\n\nline2\n\n\nline3"
        result = normalize_body(text)
        assert "\n\n" not in result
