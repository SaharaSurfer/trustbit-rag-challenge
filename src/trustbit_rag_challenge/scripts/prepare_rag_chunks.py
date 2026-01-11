import json
import re
from pathlib import Path
from typing import Any

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from trustbit_rag_challenge.config import (
    CHUNK_OVERLAP,
    MARKDOWN_IMAGE_PATTERN,
    MARKER_PAGE_SEPARATOR_PATTERN,
    MAX_TOKENS_PER_CHUNK,
    PROCESSED_DATA_DIR,
)
from trustbit_rag_challenge.logging_utils import setup_logging


class CustomRAGChunker:
    """
    Token-aware Markdown chunker for Retrieval-Augmented Generation (RAG).

    The chunker performs the following steps:
    1. Splits a document into pages using explicit page markers.
    2. Splits each page by Markdown headers.
    3. Further splits header sections into token-limited chunks with overlap.
    4. Cleans chunk text (removes images, normalizes whitespace).
    5. Emits chunk records with metadata.
    """

    def __init__(self):
        """
        Initialize the chunker.

        Sets up:
        - Markdown header-based splitter (h1–h4)
        - Tokenizer (cl100k_base)
        - Recursive character splitter with token-aware length function
        """

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_TOKENS_PER_CHUNK,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=self.num_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def num_tokens(self, text: str) -> int:
        """
        Compute the number of tokens in a text string.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        int
            Number of tokens according to the tokenizer.
        """

        return len(self.tokenizer.encode(text))

    def clean_text(self, text: str) -> str:
        """
        Clean Markdown text.

        Operations:
        - Remove Markdown image references
        - Collapse excessive newlines
        - Strip leading/trailing whitespace

        Parameters
        ----------
        text : str
            Raw Markdown text.

        Returns
        -------
        str
            Cleaned text suitable for embedding.
        """

        text = MARKDOWN_IMAGE_PATTERN.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_by_pages(self, full_text: str) -> list[dict[str, Any]]:
        """
        Split a full document into pages using explicit page separators.

        Page separators are expected to be captured by
        MARKER_PAGE_SEPARATOR_PATTERN and include a page index.

        Parameters
        ----------
        full_text : str
            Full document text.

        Returns
        -------
        List[Dict[str, Any]]
            List of page dictionaries with keys:
            - "page_number": int
            - "text": str
        """

        parts = MARKER_PAGE_SEPARATOR_PATTERN.split(full_text)

        pages = []
        for i in range(1, len(parts), 2):
            page_idx = int(parts[i])
            page_content = parts[i + 1]

            if not page_content.strip():
                continue

            pages.append({"page_number": page_idx, "text": page_content})

        return pages

    def process_document(self, content_path: Path, doc_sha1: str) -> list[dict]:
        """
        Process a single Markdown document into RAG-ready chunks.

        Parameters
        ----------
        content_path : Path
            Path to the assembled Markdown file (content.md).
        doc_sha1 : str
            Stable document identifier (e.g., directory name or hash).

        Returns
        -------
        List[Dict]
            List of chunk records, each containing:
            - chunk_id
            - text
            - length_tokens
            - source
            - page_index
        """

        full_text = content_path.read_text(encoding="utf-8")
        raw_pages = self.split_by_pages(full_text)

        final_chunks: list[dict[str, Any]] = []
        for page in raw_pages:
            page_num = page["page_number"]
            page_text = page["text"]

            clean_page_text = self.clean_text(page_text)
            if not clean_page_text:
                continue

            header_splits = self.header_splitter.split_text(clean_page_text)
            sub_chunks = self.text_splitter.split_documents(header_splits)
            for chunk in sub_chunks:
                record = {
                    "chunk_id": len(final_chunks),
                    "text": chunk.page_content,
                    "length_tokens": self.num_tokens(chunk.page_content),
                    "source": doc_sha1,
                    "page_index": page_num,
                }
                final_chunks.append(record)

        return final_chunks


def main():
    """
    Chunk all processed documents and write chunk datasets to disk.

    For each document directory in PROCESSED_DATA_DIR:
    - Read full/content.md
    - Generate chunks
    - Write dataset.json containing chunk records

    Logs progress and failures without interrupting the full run.
    """

    setup_logging()

    chunker = CustomRAGChunker()

    doc_dirs = sorted([d for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir()])
    logger.info(f"Found {len(doc_dirs)} documents to chunk.")

    for doc_dir in doc_dirs:
        full_md_path = doc_dir / "full" / "content.md"

        if not full_md_path.exists():
            logger.warning(
                f"Skipping {doc_dir.name}: 'full/content.md' not found."
            )
            continue

        logger.info(f"Chunking {doc_dir.name}...")

        try:
            doc_chunks = chunker.process_document(full_md_path, doc_dir.name)
            logger.info(f"  -> Generated {len(doc_chunks)} chunks.")

            with open(doc_dir / "dataset.json", "w", encoding="utf-8") as f:
                json.dump(doc_chunks, f, ensure_ascii=False, indent=4)

        except Exception as e:
            logger.error(f"Failed to chunk {doc_dir.name}: {e}")
            continue

    logger.success("Done!")
