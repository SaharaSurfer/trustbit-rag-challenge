import re
from typing import Any

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from trustbit_rag_challenge.config import (
    CHROMA_DB_DIR,
    COMPANY_MAPPING_PATH,
    DEVICE,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    HNSW_CHROMA_SETTINGS,
    RERANKER_MODEL_NAME,
)


class ChromaRetriever:
    """
    A retrieval system based on ChromaDB.

    This class handles connecting to the vector store, loading the
    company-to-SHA1 mapping, and performing similarity searches with
    metadata filtering.

    Attributes
    ----------
    vector_db_dir : Path
        Directory path where ChromaDB data is persisted.
    mapping_csv : Path
        Path to the CSV file containing 'company_name' to 'sha1' mapping.
    embeddings : HuggingFaceEmbeddings
        The embedding model instance used for query encoding.
    vector_store : Chroma
        The ChromaDB vector store instance.
    reranker: CrossEncoder
        The reranker model instance used for chunks reranking.
    company_map : Dict[str, str]
        A dictionary mapping company names to their PDF SHA1 hashes.
    """

    def __init__(self) -> None:
        """
        Initialize the ChromaRetriever.

        Sets up the embedding model, connects to the existing
        ChromaDB collection, and loads the company mapping into memory.
        """
        self.vector_db_dir = CHROMA_DB_DIR
        self.mapping_csv = COMPANY_MAPPING_PATH

        logger.info(
            f"Loading embedding model: {EMBEDDING_MODEL} on {DEVICE}..."
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE},
        )

        logger.info(f"Connecting to ChromaDB at {self.vector_db_dir}...")
        self.vector_store = Chroma(
            collection_name="financial_reports",
            collection_metadata=HNSW_CHROMA_SETTINGS,
            embedding_function=self.embeddings,
            persist_directory=str(self.vector_db_dir),
            create_collection_if_not_exists=False,
        )

        logger.info(f"Loading reranker model: {RERANKER_MODEL_NAME}...")
        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME, device=DEVICE, model_kwargs={"dtype": "auto"}
        )

        self.company_map = self._load_company_map()
        logger.info(f"Loaded mapping for {len(self.company_map)} companies.")

        self._bm25_cache: dict[
            str, tuple[BM25Okapi, list[str], list[dict]]
        ] = {}

    def _load_company_map(self) -> dict[str, str]:
        """
        Load and validate the company mapping CSV.

        Returns
        -------
        Dict[str, str]
            A dictionary where keys are company names and values are
            SHA1 strings.

        Raises
        ------
        FileNotFoundError
            If the mapping CSV file does not exist.
        ValueError
            If the CSV is missing required columns ('company_name', 'sha1').
        """
        if not self.mapping_csv.exists():
            logger.error(f"Mapping CSV not found at {self.mapping_csv}")
            raise FileNotFoundError(
                f"Mapping CSV not found at {self.mapping_csv}"
            )

        try:
            df = pd.read_csv(self.mapping_csv)
            if "company_name" not in df.columns or "sha1" not in df.columns:
                raise ValueError(
                    "CSV must contain 'company_name' and 'sha1' columns"
                )

            return dict(
                zip(
                    df["company_name"].astype(str).str.strip('"'),
                    df["sha1"].astype(str),
                    strict=True,
                )
            )

        except Exception as e:
            logger.error(f"Error loading company map: {e}")
            raise

    def _get_sha1_by_name(self, company_name: str) -> str | None:
        """
        Retrieve the SHA1 hash for a given company name.

        Parameters
        ----------
        company_name : str
            The exact name of the company as it appears in the CSV.

        Returns
        -------
        Optional[str]
            The SHA1 hash if found, otherwise None.
        """
        return self.company_map.get(company_name, None)

    def _get_bm25_index(
        self, sha1: str
    ) -> tuple[BM25Okapi, list[str], list[dict]] | None:
        """
        Load, build, and cache a BM25 index for a specific document.

        Parameters
        ----------
        sha1 : str
            The SHA1 hash identifier of the document. Used to locate the
            corresponding `dataset.json` file in `PROCESSED_DATA_DIR`.

        Returns
        -------
        tuple[BM25Okapi, list[str]] | None
            A tuple containing:
            1. The initialized `BM25Okapi` index object.
            2. The raw list of chunk documents.

        Returns `None` if the dataset file does not exist, is empty, or
        cannot be parsed.
        """

        if sha1 in self._bm25_cache:
            return self._bm25_cache[sha1]

        try:
            results = self.vector_store.get(where={"source": sha1})
        except Exception as e:
            logger.error(f"Failed to fetch docs from Chroma for BM25: {e}")
            return None

        raw_chunks = results.get("documents")
        metadatas = results.get("metadatas")
        if not raw_chunks or not metadatas:
            logger.warning(f"No documents found in DB for source: {sha1}")
            return None

        corpus = [re.findall(r"\w+", text.lower()) for text in raw_chunks]
        bm25 = BM25Okapi(corpus)

        self._bm25_cache[sha1] = (bm25, raw_chunks, metadatas)

        return bm25, raw_chunks, metadatas

    def _fetch_bm25_candidates(
        self, query: str, sha1: str, fetch_k: int
    ) -> list[Document]:
        """
        Retrieve candidate chunks using BM25 keyword search.

        Parameters
        ----------
        query : str
            The search query string.
        sha1 : str
            The SHA1 hash of the target document (to retrieve the cached index).
        fetch_k : int
            The maximum number of candidate documents to retrieve.

        Returns
        -------
        list[Document]
            A list of LangChain `Document` objects representing the top matching
            chunks. Returns an empty list if the index cannot be loaded or
            if no non-zero matches are found.
        """

        cached_data = self._get_bm25_index(sha1)
        if not cached_data:
            return []

        bm25, raw_chunks, metadatas = cached_data

        tokenized_query = re.findall(r"\w+", query.lower())
        scores = bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:fetch_k]

        bm25_docs = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            meta = metadatas[idx].copy()
            bm25_docs.append(
                Document(page_content=raw_chunks[idx], metadata=meta)
            )

        return bm25_docs

    def _fetch_vector_candidates(
        self, query: str, sha1: str, fetch_k: int
    ) -> list[Document]:
        """
        Retrieve candidate chunks using semantic similarity.

        Parameters
        ----------
        query : str
            The input search query (semantic question).
        sha1 : str
            The SHA1 hash of the target document. Used as a metadata filter
            (`{"source": sha1}`).
        fetch_k : int
            The number of candidate chunks to retrieve from the vector index.

        Returns
        -------
        list[Document]
            A list of LangChain `Document` objects. Returns an empty list if
            the vector search fails or raises an exception.
        """

        try:
            vector_docs_and_scores = (
                self.vector_store.similarity_search_with_score(
                    query, k=fetch_k, filter={"source": sha1}
                )
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        return [doc for doc, _ in vector_docs_and_scores]

    def retrieve(
        self, query: str, company_name: str, top_k: int = 5, fetch_k: int = 30
    ) -> list[dict[str, Any]]:
        """
        Execute the Hybrid Retrieval pipeline (Vector + BM25 + Reranking).

        This method orchestrates the retrieval process by:
        1. Fetching candidate chunks using Vector/Semantic search.
        2. Fetching candidate chunks using BM25/Keyword search.
        3. Merging and deduplicating the candidate pools based on content.
        4. Re-scoring the unique candidates using a Cross-Encoder (Reranker).
        5. Returning the top-k results with the highest semantic relevance.

        Parameters
        ----------
        query : str
            The user's question or search query.
        company_name : str
            The name of the target company. Used to resolve the specific
            document index (SHA1) via the company mapping.
        top_k : int, optional
            The number of final, reranked chunks to return to the LLM.
            Default is 5.
        fetch_k : int, optional
            The number of candidates to retrieve from *each* source
            (Vector and BM25) before merging. Default is 30.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries representing the most relevant chunks, sorted
            by reranker score (descending). Each dictionary contains:
            - 'text': str
            - 'score': float
            - 'page_index': int
            - 'source': str
        """

        target_sha1 = self._get_sha1_by_name(company_name)
        if not target_sha1:
            logger.warning(f"Company '{company_name}' not found.")
            return []

        # Fetch candidates
        vector_docs = self._fetch_vector_candidates(query, target_sha1, fetch_k)
        bm25_docs = self._fetch_bm25_candidates(query, target_sha1, fetch_k)

        # Merge & deduplicate
        unique_docs_map = {}
        for doc in bm25_docs:
            unique_docs_map[doc.page_content] = doc

        for doc in vector_docs:
            unique_docs_map[doc.page_content] = doc

        candidates = list(unique_docs_map.values())
        if not candidates:
            return []

        logger.debug(f"Pool: {len(candidates)} candidates ")

        # Reranking
        pairs = [[query, doc.page_content] for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)

        reranked_results: list[dict[str, Any]] = []
        for i, doc in enumerate(candidates):
            reranked_results.append(
                {"doc": doc, "score": float(rerank_scores[i])}
            )

        # Sort & cut
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        final_selection = reranked_results[:top_k]

        # Format
        clean_results = []
        for item in final_selection:
            doc = item["doc"]
            clean_results.append(
                {
                    "text": doc.page_content,
                    "score": round(item["score"], 4),
                    "page_index": doc.metadata.get("page_index"),
                    "source": doc.metadata.get("source"),
                }
            )

        return clean_results
