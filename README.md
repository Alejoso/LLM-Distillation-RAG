
### Why this embedding model?

Here it says that *BGE (BAAI General Embedding)* is really good for multilingual contexts and its bets use case is for: "Use case: Enterprise RAG in finance or law, where precision in multilingual documents matters."

[Blog here](https://greennode.ai/blog/best-embedding-models-for-rag#bge-baai-general-embedding)

Hugging face page of the model: [Here](https://huggingface.co/BAAI/bge-m3)


### Creating the chunks

RecursiveCharacterTextSplitter.from_tiktoken_encoder

We defined that using the tiktoken encoder migth be a good aproach. This tokens differ from the tokenizer of the embedding model that we are using.
The number of tokens goes higher than the ones from BGE, so we wont have problems getting a bigger token size than the model lets us to embed when retriving the chunks. 

### How does the metadata look

{
  "metadatas": [
    {
      "puntaje_de_calidad": "100",
      "estatus_de_calidad": "HIGH",
      "estado": "Vigencia en Estudio",

      "tipo": "LEY",
      "subtipo": "LEY ORDINARIA",
      "numero": "1",
      "año": "1914",
      "entidad": "CONGRESO DE LA REPÚBLICA",
      "fecha_de_publicacion": "01/06/1914",
      "fuente": "DIARIO OFICIAL. AÑO L .N. 15205. 1, JUNIO, 1914. PÁG. 1.",

      "source": "dataCleanedExample/Laws/1555471.txt",
      "archivo_origen": "1555471.txt",

      "doc_index": 0, # Which doc it is, comming from 0 to n being 0 the first one
      "chunk_index_in_doc": 0, # Tells which chunk from a document is
      "start_index": 0, # When does the chunk starts (In case there is overlapping)
      "chunk_id_with_source_file": "1555471.txt:0", # Source with chunk id, for better human reading

      "n_tokens_bge": 606 # Number of tokens of the chunk
    }
  ]
}

### Reranking

https://huggingface.co/BAAI/bge-reranker-v2-m3

Implemented reranking for better context retrival