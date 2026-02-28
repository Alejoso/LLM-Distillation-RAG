# Making the query work
import openai
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import argparse
from langchain_core.prompts import ChatPromptTemplate

# Evaluation metrics
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualPrecisionMetric



# Types
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

# Other .py files
from CreateDB.define_BGEM3_embeddings import BgeM3Embeddings
from QueryDB.reranker_BGE import Reranker

CHROMA_PATH  = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
embedding_model = BgeM3Embeddings()
reranker = Reranker()

# Get a list of string for using it in the metrics evaluators
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

def contextual_relevancy(query_text: str , actual_output: str , retrieval_context: List[str]):
    metric = ContextualRelevancyMetric(
        threshold=0.7, 
        model="gpt-5-nano",
        include_reason=True
    )

    test_case = LLMTestCase(
        input=query_text,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )

    return evaluate(test_cases=[test_case], metrics=[metric])

def contextual_recall(query_text:str , expected_output: str , actual_output: str , retrieval_context: List[str]):
    metric = ContextualRecallMetric(
        threshold=0.7, 
        model="gpt-5-nano",
        include_reason=True
    )
    test_case = LLMTestCase(
        input=query_text,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )
    return evaluate(test_cases=[test_case], metrics=[metric])

def contextual_precision(query_text:str , expected_output: str , actual_output: str , retrieval_context: List[str]):
    metric = ContextualPrecisionMetric(
        threshold=0.7, 
        model="gpt-5-nano",
        include_reason=True
    )
    test_case = LLMTestCase(
        input=query_text,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )
    return evaluate(test_cases=[test_case], metrics=[metric])
    
def main():
    # Define a parser for inputing the information on the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text" , type=str , help="The query text")
    args = parser.parse_args()
    query_text = args.query_text

    db = Chroma(persist_directory=CHROMA_PATH , embedding_function= embedding_model)

    # Do the similatiry and extract 10 results, then Rerank, getting only the 6 best
    results = db.similarity_search_with_relevance_scores(query_text , k = 10)
    ranked_chunks = reranker.rerank_similarity_results(query_text , results, 6)

    if len(ranked_chunks) == 0:
        print("Unable to find mathching results")
        return
        
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text , question = query_text)

    model = ChatOpenAI(
        model="gpt-5-nano"
    )
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source" , None) for doc, _ in results]
    formatted_response = f"Response: {response_text.text}\nSources: {sources}"
    print(formatted_response)

    formatted_chunks_for_metrics = ranked_to_retrieval_context(ranked_chunks)

    # relevancy_score = contextual_relevancy(query_text , response_text.text , formatted_chunks_for_metrics)
    # print(f"Contextual relevancy score {relevancy_score}")

    expected_response = "La Ley 288 de 1882 abrió un crédito suplemental de 80,000 pesos, imputable al Departamento de la Deuda Nacional, capítulo 55, artículo 184 del Presupuesto de Gastos de la vigencia 1881-1882."
    recall_score = contextual_recall(query_text , expected_response , response_text.text , formatted_chunks_for_metrics)
    print(f"Contextual recall score {recall_score}")

    precision_score = contextual_precision(query_text , expected_response , response_text.text , formatted_chunks_for_metrics)
    print(f"Contextual precision score {precision_score}")

if __name__ == "__main__":
    main()
