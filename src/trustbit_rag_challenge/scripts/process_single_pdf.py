import sys
import argparse
import gc
import torch

from pathlib import Path
from loguru import logger
from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from marker.output import save_output

from trustbit_rag_challenge.config import PROCESSED_DATA_DIR, MARKER_BASE_CONFIG


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing:
        - pdf_path (str): path to the PDF file
        - page_range (str | None): page range to process
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=str)
    parser.add_argument(
        "--page-range",
        type=str,
        default=None,
        help='Comma separated pages/ranges, e.g. "0,5-10,20" (optional)'
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entrypoint for processing a small single PDF (or chunk).

    This function:
    - Configures Marker
    - Creates output directories
    - Runs conversion
    - Ensures GPU memory is freed before process exit

    Returns
    -------
    int
        Exit code:
        - 0 on success or if output already exists
        - 1 on processing failure
    """
    args = parse_args()
    pdf_path = Path(args.pdf_path)
    page_range = args.page_range

    marker_config = MARKER_BASE_CONFIG.copy()
    if page_range:
        marker_config["page_range"] = page_range

    # Output directory layout:
    # processed/<pdf_stem>/<page_range>/content.md
    # or
    # processed/<pdf_stem>/content.md
    doc_output_dir = PROCESSED_DATA_DIR / pdf_path.stem
    if page_range:
        doc_output_dir /= page_range
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing pages {page_range}...")

    # Explicit initialization for safe cleanup in `finally`
    converter = None
    rendered = None

    try:
        converter = PdfConverter(
            config=ConfigParser(marker_config).generate_config_dict(),
            artifact_dict=create_model_dict(),
        )
        rendered = converter(str(pdf_path))
        save_output(rendered, str(doc_output_dir), "content")
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR processing {pdf_path.name}: {e}")
        return 1

    else:
        return 0
    
    finally:
        del converter
        del rendered
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
