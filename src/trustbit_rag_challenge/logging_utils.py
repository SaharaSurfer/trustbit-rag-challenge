import sys
from pathlib import Path

from loguru import logger

from trustbit_rag_challenge.config import LOGS_DIR


def setup_logging():
    """
    Configure the Loguru logger to output logs to a file.

    The log filename is generated dynamically based on the entry point script's
    name. For example, running `python main.py` will create a log file named
    `main.log` in the logs directory.

    Returns
    -------
    None
    """

    script_called = Path(sys.argv[0]).stem
    log_file = LOGS_DIR / f"{script_called}.log"

    logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")
