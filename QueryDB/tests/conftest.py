"""
Shared fixtures for the test suite.

Project-wide fixtures go here. Module-specific fixtures stay in their
respective test files.
"""

import pytest
from typing import Any, Dict, List


@pytest.fixture
def sample_ranked_chunks() -> List[Dict[str, Any]]:
    """A reusable set of ranked chunks for tests that need retrieval context."""
    return [
        {
            "id": "chunk_001",
            "text": "Ley 100 de 1993 establece el sistema de seguridad social.",
            "metadata": {"source": "ley100.pdf", "page": 1},
            "rerank_score": 0.95,
            "retriever_score": 0.88,
        },
        {
            "id": "chunk_002",
            "text": "Decreto 2591 de 1991 reglamenta la acción de tutela.",
            "metadata": {"source": "decreto2591.pdf", "page": 3},
            "rerank_score": 0.87,
            "retriever_score": 0.82,
        },
        {
            "id": "chunk_003",
            "text": "Artículo 86 de la Constitución Política de Colombia.",
            "metadata": {"source": "constitucion.pdf", "page": 15},
            "rerank_score": 0.80,
            "retriever_score": 0.79,
        },
    ]