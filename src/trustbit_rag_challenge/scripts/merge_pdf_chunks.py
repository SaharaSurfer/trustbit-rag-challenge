import json
import shutil
from pathlib import Path
from typing import Any

from loguru import logger

from trustbit_rag_challenge.config import (
    CHUNK_DIR_PATTERN,
    MARKER_IMAGE_EXTENSION,
    PROCESSED_DATA_DIR,
)
from trustbit_rag_challenge.logging_utils import setup_logging


def chunk_sort_key(page_range: Path) -> int:
    """
    Sorting key for chunk directories.

    Extracts the starting page number from a chunk directory name
    formatted as "<start>-<end>" (e.g. "100-199" -> 100).

    Parameters
    ----------
    chunk_dir : Path
        Path to a chunk directory.

    Returns
    -------
    int
        Starting page number of the chunk.
    """
    return int(page_range.name.split("-")[0])


def merge_chunks(pdf_dir: Path) -> None:
    """
    Merge chunked outputs for a single PDF into a unified document.

    This function:
    - detects chunk directories named "<start>-<end>"
    - concatenates all chunk `content.md` files
    - merges JSON metadata (`table_of_contents` and `page_stats`)
    - copies all extracted images into a single `full/` directory

    The resulting structure is:

        processed/<pdf_stem>/full/
            ├── content.md
            ├── content_meta.json
            └── *.jpeg (or configured image extension)

    Parameters
    ----------
    pdf_dir : Path
        Path to the processed PDF directory containing chunk subdirectories.
    """
    chunk_dirs = sorted(
        [
            d
            for d in pdf_dir.iterdir()
            if d.is_dir() and CHUNK_DIR_PATTERN.fullmatch(d.name)
        ],
        key=chunk_sort_key,
    )

    if not chunk_dirs:
        logger.warning(f"No chunk directories found for {pdf_dir} — skipping")
        return

    result_dir = pdf_dir / "full"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True)

    logger.info(f"Gluing {len(chunk_dirs)} chunks for {pdf_dir}...")

    full_markdown = []
    full_metadata: dict[str, list[Any]] = {
        "table_of_contents": [],
        "page_stats": [],
    }

    for chunk_dir in chunk_dirs:
        # Markdown
        md_file = chunk_dir / "content.md"
        if not md_file.exists():
            logger.error(f"Missing content.md in {chunk_dir}")
            return

        full_markdown.append(md_file.read_text(encoding="utf-8"))

        # JSON metadata
        meta_file = chunk_dir / "content_meta.json"
        if not meta_file.exists():
            logger.error(f"Missing content_meta.json in {chunk_dir}")
            return

        with open(meta_file, encoding="utf-8") as f:
            metadata = json.load(f)

        full_metadata["table_of_contents"].extend(metadata["table_of_contents"])
        full_metadata["page_stats"].extend(metadata["page_stats"])

        # Images
        for img_file in chunk_dir.glob(f"*.{MARKER_IMAGE_EXTENSION}"):
            shutil.copy2(img_file, result_dir)

    (result_dir / "content.md").write_text(
        "".join(full_markdown), encoding="utf-8"
    )

    with open(result_dir / "content_meta.json", "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, ensure_ascii=False, indent=4)

    logger.success(f"Merged {pdf_dir} successfully into {result_dir}")


def main() -> None:
    """
    Assemble full documents for all processed PDFs.

    Iterates over directories in PROCESSED_DATA_DIR and merges
    chunked outputs where applicable.
    """
    setup_logging()

    for doc_dir in sorted(
        [d for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir()]
    ):
        merge_chunks(doc_dir)
