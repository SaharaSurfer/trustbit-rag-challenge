import os
import subprocess
import sys
import time

from pathlib import Path
from loguru import logger
from pypdf import PdfReader

from trustbit_rag_challenge.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNK_PAGES, PYTORCH_ENV
)


def get_page_count(pdf_path: Path) -> int:
    """
    Return the number of pages in a PDF.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    int
        Number of pages.

    Raises
    ------
    Exception
        If the PDF cannot be read.
    """
    return len(PdfReader(pdf_path).pages)


def chunk_already_processed(pdf_path: Path, page_range: str) -> bool:
    """
    Determine whether a specific page range of a PDF has already been processed.

    A chunk is considered processed if the expected output markdown file
    exists on disk:

        processed/<pdf_stem>/<page_range>/content.md

    Parameters
    ----------
    pdf_path : Path
        Path to the source PDF file.
    page_range : str
        Page range identifier (e.g. "0-99").

    Returns
    -------
    bool
        True if the output file for this chunk already exists,
        False otherwise.
    """
    output_path = (
        PROCESSED_DATA_DIR
        / pdf_path.stem
        / page_range
        / "content.md"
    )
    return output_path.exists()


def invoke_worker(pdf_path: Path, page_range: str | None) -> bool:
    """
    Invoke the single-PDF/single-chunk worker as a subprocess.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.
    page_range : str | None
        Page range to process (e.g. "0-99").

    Returns
    -------
    bool
        True if the worker exited with code 0, False otherwise.
    """
    env = os.environ.copy()
    env.update(PYTORCH_ENV)
    
    cmd = [
        sys.executable,
        "-m",
        "trustbit_rag_challenge.scripts.process_single",
        str(pdf_path),
    ]
    if page_range:
        cmd += ["--page-range", page_range]

    result = subprocess.run(cmd, env=env)

    # Cool down laptop
    time.sleep(3)

    return result.returncode == 0


def chunk_ranges(num_pages: int, chunk_size: int) -> list[str]:
    """
    Split page indices into contiguous ranges.

    Example:
    num_pages=120, chunk_size=50 ->
    ["0-49", "50-99", "100-119"]
    """
    ranges = []
    for start in range(0, num_pages, chunk_size):
        end = min(num_pages - 1, start + chunk_size - 1)
        ranges.append(f"{start}-{end}")
    
    return ranges


def main() -> None:
    """
    Main batch-processing entrypoint.

    Iterates over PDFs and processes them chunk-by-chunk.
    """
    logger.add(PROCESSED_DATA_DIR / "processing.log", rotation="10 MB")

    pdf_files = sorted(list(RAW_DATA_DIR.glob("*.pdf")))
    logger.info(f"Found {len(pdf_files)} PDFs to process.")

    for pdf_path in pdf_files:
        try:
            page_count = get_page_count(pdf_path)
            
        except Exception as e:
            logger.error("Failed to read {}: {}", pdf_path.name, e)
            continue

        logger.info(f"{pdf_path.name}: {page_count} pages")

        for page_range in chunk_ranges(page_count, CHUNK_PAGES):
            if chunk_already_processed(pdf_path, page_range):
                logger.debug(
                    "Skipping already processed chunk {} for {}",
                    page_range,
                    pdf_path.name
                )
                continue

            logger.info(
                "Spawning worker for {} | CHUNK: {}",
                pdf_path.name,
                page_range
            )
            
            status_ok = invoke_worker(pdf_path, page_range)
            if not status_ok:
                logger.error(
                    "Chunk {} FAILED for {}, continuing to next chunk",
                    page_range,
                    pdf_path
                )
            else:
                logger.success(
                    "Chunk {} SUCCEEDED for {}, continuing to next chunk",
                    page_range,
                    pdf_path
                )
