from pathlib import Path
import sqlite3

# Locate OushudhCheck root and the read-only medicines.db path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "medicines.db"


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
