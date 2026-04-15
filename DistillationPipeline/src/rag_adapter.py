from typing import Any, Dict, List

from langchain_chroma import Chroma

from CreateDB.define_BGEM3_embeddings import BgeM3Embeddings
from QueryDB.reranker_BGE import Reranker


_embedding_model = None
_reranker = None
_db = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = BgeM3Embeddings()
    return _embedding_model


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def get_db(config):
    global _db
    if _db is None:
        _db = Chroma(
            persist_directory=str(config["rag_chroma_path"]),
            embedding_function=get_embedding_model()
        )
    return _db


def ranked_to_retrieval_context(ranked: List[Dict[str, Any]]) -> List[str]:
    contexts: List[str] = []

    for item in ranked:
        text = (item.get("text") or "").strip()
        if text:
            contexts.append(text)

    return contexts


def retrieve_contexts(instruction: str, config) -> List[str]:
    instruction = (instruction or "").strip()

    if not instruction:
        return []

    db = get_db(config)
    reranker = get_reranker()

    results = db.similarity_search_with_relevance_scores(
        instruction,
        k=config.get("rag_top_k_initial", 10)
    )

    if not results:
        return []

    ranked_chunks = reranker.rerank_similarity_results(
        instruction,
        results,
        config.get("rag_top_k_final", 6)
    )

    return ranked_to_retrieval_context(ranked_chunks)