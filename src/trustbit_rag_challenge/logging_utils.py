import sys
from pathlib import Path

from loguru import logger

from trustbit_rag_challenge.config import LOGS_DIR


def setup_logging():
    script_called = Path(sys.argv[0]).stem
    log_file = LOGS_DIR / f"{script_called}.log"

    logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")
