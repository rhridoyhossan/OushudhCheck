from pathlib import Path
import os
import sqlite3

# Locate OushudhCheck root and resolve paths safely (even when .env uses relative paths)
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
ROOT_DIR = PROJECT_ROOT

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def _resolve_project_path(env_var: str, default_rel: Path) -> Path:
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return (PROJECT_ROOT / default_rel).resolve()
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


DB_PATH = _resolve_project_path("MEDICINES_DB_PATH", Path("data") / "processed" / "medicines.db")
YOLO_MODEL_PATH = _resolve_project_path("YOLO_MODEL_PATH", Path("data") / "processed" / "yolo_strips.onnx")

# Fuzzy matching & OCR thresholds
MIN_FUZZY_SCORE = float(os.getenv("MIN_FUZZY_SCORE", "72.0"))
MAX_STRIPS_PER_IMAGE = int(os.getenv("MAX_STRIPS_PER_IMAGE", "6"))
OCR_LANG = os.getenv("OCR_LANG", "en")

# YOLO26 ONNX Strip Detector runtime controls (env-tunable for local/Docker/onsite judge tweaks)
YOLO_IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.22"))
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.50"))
YOLO_CROP_PAD_RATIO = float(os.getenv("YOLO_CROP_PAD_RATIO", "0.06"))
YOLO_MAX_PROPOSALS = int(os.getenv("YOLO_MAX_PROPOSALS", str(max(MAX_STRIPS_PER_IMAGE * 2, 12))))
YOLO_PIPELINE_MODE = os.getenv("YOLO_PIPELINE_MODE", "hybrid_fusion").strip().lower()
YOLO_ORT_THREADS = int(
    os.getenv("YOLO_ORT_THREADS", str(min(4, max(1, (os.cpu_count() or 4) // 2))))
)
YOLO_ENABLE_CROP_RESCUE = os.getenv("YOLO_ENABLE_CROP_RESCUE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
YOLO_RESCUE_CONF_MIN = float(os.getenv("YOLO_RESCUE_CONF_MIN", "0.40"))
YOLO_RESCUE_MAX_CROPS = int(os.getenv("YOLO_RESCUE_MAX_CROPS", "2"))
YOLO_REQUIRE_REAL_MODEL = os.getenv("YOLO_REQUIRE_REAL_MODEL", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def get_readonly_db_connection() -> sqlite3.Connection:
    """
    Opens medicines.db strictly in READ-ONLY mode using SQLite URI mode (?mode=ro).
    Never modifies or touches the database file on disk.
    """
    resolved = DB_PATH.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Database file not found at {resolved}")

    db_uri = f"{resolved.as_uri()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
