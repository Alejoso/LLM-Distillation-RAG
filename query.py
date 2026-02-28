import openai
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import argparse
from langchain_core.prompts import ChatPromptTemplate

from define_BGEM3_embeddings import BgeM3Embeddings
from reranker_BGE import Reranker
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



if __name__ == "__main__":
    main()
