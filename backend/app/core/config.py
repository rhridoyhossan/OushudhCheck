from pathlib import Path
import os
import sqlite3

# Locate OushudhCheck root and the read-only medicines.db path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
ROOT_DIR = PROJECT_ROOT

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DB_PATH = Path(os.getenv("MEDICINES_DB_PATH", PROJECT_ROOT / "data" / "processed" / "medicines.db"))
YOLO_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", PROJECT_ROOT / "data" / "processed" / "yolo_strips.onnx"))

# Fuzzy matching & OCR thresholds
MIN_FUZZY_SCORE = float(os.getenv("MIN_FUZZY_SCORE", "72.0"))
MAX_STRIPS_PER_IMAGE = int(os.getenv("MAX_STRIPS_PER_IMAGE", "6"))
OCR_LANG = os.getenv("OCR_LANG", "en")


def get_readonly_db_connection() -> sqlite3.Connection:
    """
    Opens medicines.db strictly in READ-ONLY mode using SQLite URI mode (?mode=ro).
    Never modifies or touches the database file on disk.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found at {DB_PATH}")

    db_uri = f"{DB_PATH.as_uri()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
