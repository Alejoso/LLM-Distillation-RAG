from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document


class Reranker:
    def __init__(self):
        from FlagEmbedding import FlagReranker

        self.reranker = FlagReranker(
            "BAAI/bge-reranker-v2-m3", use_fp16=True
        )  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    def rerank_similarity_results(
        self,
        query: str,
        results: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[Dict[str, Any]]:

        if not results:
            return []

        docs = [doc for doc, _ in results]
        passages = [doc.page_content for doc in docs]  # Get the retrieved chunk's text

        # Rerank with the compute_score() function
        pairs = [[query, p] for p in passages]
        scores = self.reranker.compute_score(pairs)

        # Build the return
        ranked: List[Dict[str, Any]] = []
        for (doc, retr_score), rr_score in zip(results, scores):
            item = {
                "id": getattr(doc, "id", None),
                "text": doc.page_content,
                "metadata": doc.metadata,
                "rerank_score": float(rr_score),
            }
            item["retriever_score"] = float(retr_score)
            ranked.append(item)

        # Sort by rerank_score and return the number specified in top_k
        ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]