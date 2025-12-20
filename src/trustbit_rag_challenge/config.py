from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
QUESTIONS_PATH = DATA_DIR / "questions.json"

RAW_DATA_DIR = DATA_DIR / "pdfs"
if not RAW_DATA_DIR.exists() or not RAW_DATA_DIR.is_dir():
    raise FileExistsError(
        f"Raw data directory not found: {RAW_DATA_DIR}\n"
        "Please place the dataset there."
    )

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------- #

CHUNK_PAGES = 100
PYTORCH_ENV = {
    "PYTORCH_ALLOC_CONF": "expandable_segments:True"
}

MARKER_BASE_CONFIG: dict[str, object] = {
    "output_format": "markdown",
    "paginate_output": True,

    # GPU / performance
    "disable_multiprocessing": True,
    "detection_batch_size": 4,
    "recognition_batch_size": 32,
}