import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from trustbit_rag_challenge.config import (
    CHROMA_DB_DIR,
    DEVICE,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    HNSW_CHROMA_SETTINGS,
    PROCESSED_DATA_DIR,
)
from trustbit_rag_challenge.logging_utils import setup_logging


def load_chunks_from_json(json_path: Path) -> list[Document]:
    """
    Load chunk records from a JSON file and convert them to LangChain Documents.

    Each chunk record is expected to be a dict containing at least "text".
    Any other keys are treated as metadata fields.

    Parameters
    ----------
    json_path : Path
        Path to the dataset.json file produced by the chunker
        (list of chunk dicts).

    Returns
    -------
    List[Document]
        A list of `langchain_core.documents.Document` objects ready for
        indexing.

    Raises
    ------
    json.JSONDecodeError
        If the JSON file is malformed.
    """
    with open(json_path, encoding="utf-8") as f:
        chunks = json.load(f)

    docs = []
    for chunk in chunks:
        metadata = chunk.copy()
        text = metadata.pop("text", None)
        metadata.pop("chunk_id", None)

        docs.append(Document(page_content=text, metadata=metadata))

    return docs


def main() -> None:
    """
    Index all available chunked documents into the Chroma vector store.

    Steps performed:
    - Initialize the HuggingFace embedding model with the configured device
    - Initialize or open the Chroma collection at CHROMA_DB_DIR
    - Iterate over processed document directories (PROCESSED_DATA_DIR)
    - For each `dataset.json`, load the chunks and add them to the collection
    - Persist vectors to disk
    """
    setup_logging()

    logger.info(f"Using device: {DEVICE} for model {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE},
    )

    if CHROMA_DB_DIR.exists():
        logger.warning(f"Removing old DB at {CHROMA_DB_DIR}...")
        shutil.rmtree(CHROMA_DB_DIR)

    vector_store = Chroma(
        collection_name="financial_reports",
        collection_metadata=HNSW_CHROMA_SETTINGS,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
        create_collection_if_not_exists=True,
    )

    doc_dirs = sorted([d for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir()])
    logger.info(f"Found {len(doc_dirs)} documents to index.")

    total_chunks = 0
    for doc_dir in doc_dirs:
        json_path = doc_dir / "dataset.json"
        if not json_path.exists():
            continue

        try:
            docs = load_chunks_from_json(json_path)
            if not docs:
                continue

            logger.info(f"Indexing {doc_dir.name} ({len(docs)} chunks)...")

            ids = vector_store.add_documents(docs)
            total_chunks += len(ids)

        except Exception as e:
            logger.error(f"Failed to index {doc_dir.name}: {e}")

    logger.success(f"Indexing complete! Total vectors: {total_chunks}")
    logger.info(f"DB Saved to: {CHROMA_DB_DIR}")
