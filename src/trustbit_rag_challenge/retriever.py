from typing import Any

import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from sentence_transformers import CrossEncoder

from trustbit_rag_challenge.config import (
    CHROMA_DB_DIR,
    COMPANY_MAPPING_PATH,
    DEVICE,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
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
            embedding_function=self.embeddings,
            persist_directory=str(self.vector_db_dir),
        )

        logger.info(f"Loading reranker model: {RERANKER_MODEL_NAME}...")
        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME, device=DEVICE, model_kwargs={"dtype": "auto"}
        )

        self.company_map = self._load_company_map()
        logger.info(f"Loaded mapping for {len(self.company_map)} companies.")

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

    def get_sha1_by_name(self, company_name: str) -> str | None:
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

    def retrieve(
        self, query: str, company_name: str, top_k: int = 5, fetch_k: int = 30
    ) -> list[dict[str, Any]]:
        """
        Perform a semantic search for the query, filtered by the
        company's document.

        Parameters
        ----------
        query : str
            The user's question or search query.
        company_name : str
            The name of the company to restrict the search to.
        top_k : int, optional
            The number of chunks to retrieve, by default 5.
        fetch_k : int, optional
            The number of chunks to fetch, by default 30.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries representing the retrieved chunks.
            Each dictionary contains:
            - 'text': str
            - 'score': float (distance)
            - 'page_index': int
            - 'source': str (sha1)
        """
        search_kwargs: dict[str, Any] = {"k": fetch_k}

        target_sha1 = self.get_sha1_by_name(company_name)
        if target_sha1:
            search_kwargs["filter"] = {"source": target_sha1}
            logger.debug(
                f"Filtering by company: {company_name} ({target_sha1})"
            )
        else:
            logger.warning(
                f"Company '{company_name}' not found in mapping. "
                "Searching globally."
            )

        try:
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query, **search_kwargs
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        logger.debug(f"Reranking {len(docs_and_scores)} documents...")

        pairs = [[query, doc.page_content] for doc, _ in docs_and_scores]
        rerank_scores = self.reranker.predict(pairs)

        reranked_results: list[dict[str, Any]] = []
        for i, (doc, _) in enumerate(docs_and_scores):
            reranked_results.append(
                {"doc": doc, "score": float(rerank_scores[i])}
            )

        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        final_selection = reranked_results[:top_k]

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
