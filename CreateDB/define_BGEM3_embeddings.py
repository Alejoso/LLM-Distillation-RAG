from FlagEmbedding import BGEM3FlagModel
from langchain_core.embeddings import Embeddings


class BgeM3Embeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-m3", use_fp16=True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.tokenizer = self.model.tokenizer

    def transform_embeddings(self, out):
        return out.tolist()

    def embed_documents(
        self, texts
    ):  # This function have to be named exaactly like this (embed_documents and embed_query) because langchain searches for this specific names
        out = self.model.encode(
            texts,
            batch_size=12,  # How many chunks are going to be processed at a time. Higher number, faster but more memory is consumed
            max_length=8192,
        )["dense_vecs"]
        return self.transform_embeddings(out)

    def embed_query(self, text):
        out = self.model.encode(
            [text],  # Force to be a list
            max_length=8192,
        )["dense_vecs"]
        return self.transform_embeddings(out)[0]

    def count_tokens(self, text: str):
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        return len(ids)
