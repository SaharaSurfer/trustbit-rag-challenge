import re
from pathlib import Path

# -------------------- Path related --------------------- #

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

QUESTIONS_PATH = DATA_DIR / "questions.json"
COMPANY_MAPPING_PATH = DATA_DIR / "subset.csv"

RAW_DATA_DIR = DATA_DIR / "pdfs"
if not RAW_DATA_DIR.exists() or not RAW_DATA_DIR.is_dir():
    raise FileExistsError(
        f"Raw data directory not found: {RAW_DATA_DIR}\n"
        "Please place the dataset there."
    )

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_DIR_PATTERN = re.compile(r"^\d+-\d+$")

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ------------- Marker-pdf (OCR & parsing) -------------- #

CHUNK_PAGES = 100
PYTORCH_ENV = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}

MARKER_BASE_CONFIG: dict[str, object] = {
    "output_format": "markdown",
    "paginate_output": True,
    # GPU / performance
    "disable_multiprocessing": True,
    "detection_batch_size": 4,
    "recognition_batch_size": 32,
}

MARKER_IMAGE_EXTENSION = "jpeg"
MARKER_PAGE_SEPARATOR_PATTERN = re.compile(r"\n\n\{(\d+)\}-{48}\n\n")
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")

# ----------------- RAG pipeline config ----------------- #

DEVICE = "cuda"

MAX_TOKENS_PER_CHUNK = 1000
CHUNK_OVERLAP = 100

CHROMA_DB_DIR = DATA_DIR / "chroma_db"
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

HNSW_CHROMA_SETTINGS = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 256,
    "hnsw:M": 128,
    "hnsw:search_ef": 256,
}

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_NORMALIZE = True

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

OPENAI_MODEL = "gpt-4o-mini"

# -------------------- Challenge API -------------------- #

TEAM_EMAIL = "st119029@student.spbu.ru"
SURNAME = "Romanov"
SUBMISSION_NAME = f"{SURNAME}_v6"

SERVER_BASE_URL = "http://5.35.3.130:800"
SUBMISSION_URL = f"{SERVER_BASE_URL}/submit"
LEADERBOARD_URL = f"{SERVER_BASE_URL}/round_two/validate"
