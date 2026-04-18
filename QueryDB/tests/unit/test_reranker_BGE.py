import sys
import pytest
from unittest.mock import MagicMock
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Create a fake FlagEmbedding module BEFORE importing reranker_BGE,
# so the `from FlagEmbedding import FlagReranker` inside the class works.
# ---------------------------------------------------------------------------
_mock_flag_module = MagicMock()
sys.modules["FlagEmbedding"] = _mock_flag_module


@pytest.fixture
def mock_flag_reranker():
    """Provides a fresh mocked FlagReranker instance for each test."""
    mock_instance = MagicMock()
    _mock_flag_module.FlagReranker.return_value = mock_instance
    yield mock_instance


@pytest.fixture
def reranker(mock_flag_reranker):
    """Instantiates Reranker with the mocked FlagReranker."""
    from reranker_BGE import Reranker

    r = Reranker()
    return r


# ========================== Tests ==========================


class TestRerankerInit:
    """Verifies that Reranker initialises correctly."""

    def test_creates_flag_reranker_instance(self, reranker, mock_flag_reranker):
        # Assert
        assert reranker.reranker is mock_flag_reranker

    def test_reranker_has_compute_score_method(self, reranker):
        # Assert
        assert hasattr(reranker.reranker, "compute_score")


class TestRerankSimilarityResults:
    # ---- Happy path ----
    def test_returns_top_k_sorted_by_rerank_score(self, reranker, mock_flag_reranker):
        # Arrange
        doc1 = Document(page_content="Ley 100 de 1993", metadata={"source": "ley100"})
        doc2 = Document(page_content="Decreto 2591 de 1991", metadata={"source": "decreto2591"})
        doc3 = Document(page_content="Ley 1564 de 2012", metadata={"source": "ley1564"})

        results: List[Tuple[Document, float]] = [
            (doc1, 0.85),
            (doc2, 0.80),
            (doc3, 0.75),
        ]

        # Simulate reranker giving doc3 the highest score
        mock_flag_reranker.compute_score.return_value = [0.3, 0.9, 0.6]

        # Act
        ranked = reranker.rerank_similarity_results("tutela", results, top_k=2)

        # Assert
        assert len(ranked) == 2
        assert ranked[0]["text"] == "Decreto 2591 de 1991"  # highest rerank score
        assert ranked[1]["text"] == "Ley 1564 de 2012"      # second highest
        assert ranked[0]["rerank_score"] == 0.9
        assert ranked[1]["rerank_score"] == 0.6

    def test_preserves_metadata_and_retriever_score(self, reranker, mock_flag_reranker):
        # Arrange
        doc = Document(
            page_content="Artículo 86",
            metadata={"source": "constitucion", "page": 12},
        )
        results = [(doc, 0.92)]
        mock_flag_reranker.compute_score.return_value = [0.75]

        # Act
        ranked = reranker.rerank_similarity_results("acción de tutela", results, top_k=1)

        # Assert
        assert ranked[0]["metadata"] == {"source": "constitucion", "page": 12}
        assert ranked[0]["retriever_score"] == 0.92
        assert ranked[0]["rerank_score"] == 0.75
        assert ranked[0]["text"] == "Artículo 86"

    # ---- Alternative / edge cases ----
    def test_empty_results_returns_empty(self, reranker, mock_flag_reranker):
        # Arrange & Act
        ranked = reranker.rerank_similarity_results("cualquier query", [], top_k=5)

        # Assert
        assert ranked == []
        mock_flag_reranker.compute_score.assert_not_called()

    def test_top_k_larger_than_results_returns_all(self, reranker, mock_flag_reranker):
        # Arrange
        doc1 = Document(page_content="Texto 1", metadata={})
        doc2 = Document(page_content="Texto 2", metadata={})
        results = [(doc1, 0.8), (doc2, 0.7)]
        mock_flag_reranker.compute_score.return_value = [0.5, 0.4]

        # Act
        ranked = reranker.rerank_similarity_results("query", results, top_k=10)

        # Assert
        assert len(ranked) == 2

    def test_single_result_is_returned(self, reranker, mock_flag_reranker):
        # Arrange
        doc = Document(page_content="Único documento", metadata={"source": "unico"})
        results = [(doc, 0.95)]
        mock_flag_reranker.compute_score.return_value = [0.88]

        # Act
        ranked = reranker.rerank_similarity_results("búsqueda", results, top_k=1)

        # Assert
        assert len(ranked) == 1
        assert ranked[0]["text"] == "Único documento"

    def test_correct_pairs_sent_to_compute_score(self, reranker, mock_flag_reranker):
        # Arrange
        doc1 = Document(page_content="Primero", metadata={})
        doc2 = Document(page_content="Segundo", metadata={})
        results = [(doc1, 0.9), (doc2, 0.8)]
        mock_flag_reranker.compute_score.return_value = [0.5, 0.6]

        # Act
        reranker.rerank_similarity_results("mi query", results, top_k=2)

        # Assert
        expected_pairs = [["mi query", "Primero"], ["mi query", "Segundo"]]
        mock_flag_reranker.compute_score.assert_called_once_with(expected_pairs)

    def test_equal_scores_preserves_all_items(self, reranker, mock_flag_reranker):
        # Arrange
        doc1 = Document(page_content="Doc A", metadata={})
        doc2 = Document(page_content="Doc B", metadata={})
        doc3 = Document(page_content="Doc C", metadata={})
        results = [(doc1, 0.8), (doc2, 0.8), (doc3, 0.8)]
        mock_flag_reranker.compute_score.return_value = [0.5, 0.5, 0.5]

        # Act
        ranked = reranker.rerank_similarity_results("query", results, top_k=3)

        # Assert
        assert len(ranked) == 3
        texts = {item["text"] for item in ranked}
        assert texts == {"Doc A", "Doc B", "Doc C"}