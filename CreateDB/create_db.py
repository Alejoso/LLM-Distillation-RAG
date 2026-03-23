from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
import shutil
import os
from pathlib import Path
import logging
from collections import defaultdict
from CreateDB.define_BGEM3_embeddings import BgeM3Embeddings

DATA_PATH = "../dataCleanedExample/Laws"
CHROMA_PATH = "../chroma"

embedding_model = BgeM3Embeddings()

# Base config for the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="create_db.log",
    filemode="w",
)


# Load all the .txt files
def load_documents():
    logging.info("Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents


# Helper for getting value after a : black space
def value_after_colon(s: str) -> str | None:
    if ": " not in s:
        return None
    return s.split(": ", 1)[1]


# Get metadata for chunks
def get_metadata_arguments(documents: list[Document]):
    logging.info("Getting metadata for each document...")
    metas = []

    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "")
        source_file = Path(source).name if source else None

        textMetadata = [l for l in doc.page_content.splitlines() if l.strip()]
        first5 = textMetadata[:12]

        # The first 11 lines of each document is important data which is going to be saved to the chunk metadata
        type = value_after_colon(first5[0])
        number = value_after_colon(first5[1])
        year = value_after_colon(first5[2])
        state = value_after_colon(first5[3])
        entity = value_after_colon(first5[4])
        subtype = value_after_colon(first5[5])
        expedition_date = value_after_colon(first5[6])
        publish_date = value_after_colon(first5[7])
        document_source = value_after_colon(first5[8])
        quality_score = value_after_colon(first5[9])
        quality_status = value_after_colon(first5[10])

        metas.append(
            {
                "tipo": type,
                "numero": number,
                "año": year,
                "estado": state,
                "entidad": entity,
                "subtipo": subtype,
                "fecha_de_expedicion": expedition_date,
                "fecha_de_publicacion": publish_date,
                "fuente": document_source,
                "puntaje_de_calidad": quality_score,
                "estatus_de_calidad": quality_status,
                "archivo_origen": source_file,
                "doc_index": i,
            }
        )

    return metas


# Add metadata to each document
def attach_metadata(documents: list[Document], metas: list[dict]):
    logging.info("Adding metadata...")

    if len(documents) != len(metas):
        logging.error(
            f"Documents and meta array have different sizes Chunk size: {len(documents)} Meta size: {len(metas)} "
        )
        raise ValueError(
            f"Documents and meta array have different sizes \nChunk size: {len(documents)} \nMeta size: {len(metas)}"
        )

    for doc, meta in zip(documents, metas):
        doc.metadata.update(meta)

    logging.info(f"This is how document metadata look: {documents[0].metadata}")
    return documents


# Method for adding information at the start and end of the chunks
def add_info_to_chunks(chunk, prefix, suffix):
    if prefix is None or prefix == "":
        prefix = "No hay titulo del articulo"
    if suffix is None or suffix == "":
        suffix = "No se tiene una firma"

    chunk.page_content = f"{prefix}\n{chunk.page_content}\n{suffix}"


# Split text for making the chunking (In thought of getting top k = 6)
def split_text(documents: list[Document]):
    logging.info("Creating chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 800 Tokens
        chunk_overlap=150,  # Each chunk is going to have an overlap of 150 tokens
        length_function=embedding_model.count_tokens,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # See the content of chunk number 10 on the logs
    logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    document = chunks[0]
    logging.info(
        "Example of chunk number 0:\n%s\n\n--- METADATA ---\n%s",
        document.page_content,
        document.metadata,
    )

    return chunks


def add_chunk_indexes(chunks: list[Document]):
    counters = defaultdict(int)

    for chunk in chunks:
        key = chunk.metadata.get(
            "source", chunk.metadata.get("archivo_origen", "unknown")
        )
        chunk.metadata["chunk_index_in_doc"] = counters[key]
        counters[key] += 1

        # ID with the name of the source 21398.txt:0
        chunk.metadata["chunk_id_with_source_file"] = (
            f"{Path(key).name}:{chunk.metadata['chunk_index_in_doc']}"
        )
        chunk.metadata["n_tokens_bge"] = embedding_model.count_tokens(
            chunk.page_content
        )

    return chunks


# Save the chunks to to the chroma db
def save_to_chroma(chunks: list[Document]):
    logging.info("Saving information to db...")
    # Delete the data base if there is one already created
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Save the chunks into a chroma database using OpeinAIEmbeddings
    Chroma.from_documents(chunks, embedding_model, persist_directory=CHROMA_PATH)

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def main():
    documents = load_documents()

    metas = get_metadata_arguments(documents)
    documents_with_meta = attach_metadata(documents, metas)

    chunks_with_no_index = split_text(documents_with_meta)
    chunks_with_index = add_chunk_indexes(chunks_with_no_index)

    save_to_chroma(chunks_with_index)

    # Test for looking at the db and how this thingy is saved
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    one = db._collection.peek(limit=1)
    print(one)


if __name__ == "__main__":
    main()
    # try:
    #     main()
    #     logging.info("Finished :)")
    # except Exception as e:
    #     logging.error(f"Something went bad :( \n{e}")
