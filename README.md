
### Why this embedding model?

Here it says that *BGE (BAAI General Embedding)* is really good for multilingual contexts and its bets use case is for: "Use case: Enterprise RAG in finance or law, where precision in multilingual documents matters."

[Blog here](https://greennode.ai/blog/best-embedding-models-for-rag#bge-baai-general-embedding)

Hugging face page of the model: [Here](https://huggingface.co/BAAI/bge-m3)


### Creating the chunks

RecursiveCharacterTextSplitter.from_tiktoken_encoder

We defined that using the tiktoken encoder migth be a good aproach. This tokens differ from the tokenizer of the embedding model that we are using.
The number of tokens goes higher than the ones from BGE, so we wont have problems getting a bigger token size than the model lets us to embed when retriving the chunks. 