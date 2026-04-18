import pytest
from unittest.mock import patch, MagicMock
from typing import Any, Dict, List


# ---------- Reproduce the pure function under test ----------
def ranked_to_retrieval_context(
    ranked: List[Dict[str, Any]],
) -> List[str]:
    items = ranked
    context: List[str] = []
    for _, item in enumerate(items, start=1):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        context.append(text)
    return context


# ========================== Tests ==========================


class TestRankedToRetrievalContext:
    """Tests for ranked_to_retrieval_context()."""

    # ---- Happy path ----
    def test_returns_text_list_from_ranked_chunks(self):
        # Arrange
        ranked = [
            {"text": "Artículo 1. Contenido de la ley.", "metadata": {}, "rerank_score": 0.9},
            {"text": "Artículo 2. Otro contenido.", "metadata": {}, "rerank_score": 0.8},
            {"text": "Artículo 3. Más contenido.", "metadata": {}, "rerank_score": 0.7},
        ]

        # Act
        result = ranked_to_retrieval_context(ranked)

        # Assert
        assert len(result) == 3
        assert result[0] == "Artículo 1. Contenido de la ley."
        assert result[1] == "Artículo 2. Otro contenido."
        assert result[2] == "Artículo 3. Más contenido."

    # ---- Alternative / edge cases ----
    def test_empty_list_returns_empty(self):
        # Arrange
        ranked = []

        # Act
        result = ranked_to_retrieval_context(ranked)

        # Assert
        assert result == []

    def test_skips_items_with_empty_text(self):
        # Arrange
        ranked = [
            {"text": "Texto válido", "metadata": {}},
            {"text": "", "metadata": {}},
            {"text": "   ", "metadata": {}},  # whitespace-only
            {"text": "Otro texto válido", "metadata": {}},
        ]

        # Act
        result = ranked_to_retrieval_context(ranked)

        # Assert
        assert len(result) == 2
        assert "Texto válido" in result
        assert "Otro texto válido" in result

    def test_skips_items_with_none_text(self):
        # Arrange
        ranked = [
            {"text": None, "metadata": {}},
            {"text": "Contenido real", "metadata": {}},
        ]

        # Act
        result = ranked_to_retrieval_context(ranked)

        # Assert
        assert result == ["Contenido real"]

    def test_strips_whitespace_from_text(self):
        # Arrange
        ranked = [
            {"text": "  texto con espacios  ", "metadata": {}},
        ]

        # Act
        result = ranked_to_retrieval_context(ranked)

        # Assert
        assert result == ["texto con espacios"]

    def test_missing_text_key_returns_empty_string_default(self):
        # Arrange
        ranked = [
            {"metadata": {"source": "doc1"}},
        ]

        # Act
        result = ranked_to_retrieval_context(ranked)

        # Assert
        assert result == []

    @pytest.mark.parametrize(
        "text_value",
        ["", "   ", "\n", "\t", None],
        ids=["empty", "spaces", "newline", "tab", "none"],
    )
    def test_various_blank_values_are_skipped(self, text_value):
        # Arrange
        ranked = [{"text": text_value}]

        # Act
        result = ranked_to_retrieval_context(ranked)

        # Assert
        assert result == []