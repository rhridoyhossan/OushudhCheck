from pathlib import Path
import csv
import re
import sqlite3

ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "data" / "raw" / "master_medicine_bd_clean.csv"
DB_PATH = ROOT_DIR / "data" / "processed" / "medicines.db"

NULL_TOKENS = {"", "nan", "NaN", "None", "null", "NULL", "N/A", "n/a", "-"}

BRAND_COLS = [
    "brand_id",
    "brand_name",
    "type",
    "dosage_form",
    "manufacturer",
    "package_container",
    "brand_lower",
    "generic_lower",
    "beginner_summary",
]

GENERIC_COLS = [
    "generic_lower",
    "generic",
    "drug_class",
    "indication",
    "indication_description",
    "therapeutic_class_description",
    "pharmacology_description",
    "dosage_description",
    "interaction_description",
    "contraindications_description",
    "side_effects_description",
    "pregnancy_and_lactation_description",
    "precautions_description",
    "storage_conditions_description",
]

_READ_MORE_RE = re.compile(r"^.*?\.\.\.\s*Read more\s+")


def _clean(val: str | None) -> str | None:
    if val is None:
        return None
    s = val.strip()
    return None if s in NULL_TOKENS else s


def build_database() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    brands_dedup: dict[tuple[str, str, str, str], tuple] = {}
    generics_map: dict[str, dict[str, str | None]] = {}

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.strip(): _clean(v) for k, v in raw_row.items() if k is not None}
            brand_name = row.get("brand_name")
            generic = row.get("generic")
            brand_id_raw = row.get("brand_id")
            if not brand_name or not generic or not brand_id_raw:
                continue

            if row.get("indication_description"):
                row["indication_description"] = _READ_MORE_RE.sub("", row["indication_description"])

            brand_id = int(float(brand_id_raw))
            brand_lower = brand_name.lower()
            generic_lower = generic.lower()
            dosage_form = row.get("dosage_form")
            manufacturer = row.get("manufacturer")

            dedup_key = (
                brand_lower,
                generic_lower,
                (dosage_form or "").lower(),
                (manufacturer or "").lower(),
            )
            if dedup_key not in brands_dedup:
                brands_dedup[dedup_key] = (
                    brand_id,
                    brand_name,
                    row.get("type"),
                    dosage_form,
                    manufacturer,
                    row.get("package_container"),
                    brand_lower,
                    generic_lower,
                    None,
                )

            if generic_lower not in generics_map:
                generics_map[generic_lower] = {
                    "generic_lower": generic_lower,
                    **{col: row.get(col) for col in GENERIC_COLS[1:]},
                }
            else:
                existing = generics_map[generic_lower]
                for col in GENERIC_COLS[1:]:
                    if existing[col] is None and row.get(col) is not None:
                        existing[col] = row.get(col)

    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE brands (
                brand_id INTEGER PRIMARY KEY,
                brand_name TEXT NOT NULL,
                type TEXT,
                dosage_form TEXT,
                manufacturer TEXT,
                package_container TEXT,
                brand_lower TEXT NOT NULL,
                generic_lower TEXT NOT NULL,
                beginner_summary TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE generics (
                generic_lower TEXT PRIMARY KEY,
                generic TEXT NOT NULL,
                drug_class TEXT,
                indication TEXT,
                indication_description TEXT,
                therapeutic_class_description TEXT,
                pharmacology_description TEXT,
                dosage_description TEXT,
                interaction_description TEXT,
                contraindications_description TEXT,
                side_effects_description TEXT,
                pregnancy_and_lactation_description TEXT,
                precautions_description TEXT,
                storage_conditions_description TEXT
            )
            """
        )

        cur.executemany(
            f"INSERT OR IGNORE INTO brands ({', '.join(BRAND_COLS)}) VALUES ({', '.join('?' for _ in BRAND_COLS)})",
            brands_dedup.values(),
        )
        cur.executemany(
            f"INSERT OR IGNORE INTO generics ({', '.join(GENERIC_COLS)}) VALUES ({', '.join('?' for _ in GENERIC_COLS)})",
            ([g[col] for col in GENERIC_COLS] for g in generics_map.values()),
        )

        cur.execute("CREATE UNIQUE INDEX idx_brands_id ON brands(brand_id)")
        cur.execute("CREATE INDEX idx_brands_lower ON brands(brand_lower)")
        cur.execute("CREATE INDEX idx_brands_generic ON brands(generic_lower)")
        cur.execute("CREATE UNIQUE INDEX idx_generics_lower ON generics(generic_lower)")

        cur.execute(
            """
            CREATE VIEW medicines AS
            SELECT
                b.brand_id,
                b.brand_name,
                b.type,
                b.dosage_form,
                g.generic,
                b.manufacturer,
                b.package_container,
                g.drug_class,
                g.indication,
                g.indication_description,
                g.therapeutic_class_description,
                g.pharmacology_description,
                g.dosage_description,
                g.interaction_description,
                g.contraindications_description,
                g.side_effects_description,
                g.pregnancy_and_lactation_description,
                g.precautions_description,
                g.storage_conditions_description,
                b.brand_lower,
                b.generic_lower,
                b.beginner_summary
            FROM brands b
            LEFT JOIN generics g ON b.generic_lower = g.generic_lower
            """
        )

        cur.execute(
            """
            CREATE VIRTUAL TABLE medicines_fts USING fts5(
                brand_id UNINDEXED,
                brand_name,
                generic,
                dosage_form,
                manufacturer,
                tokenize='unicode61'
            )
            """
        )
        cur.execute(
            """
            INSERT INTO medicines_fts (brand_id, brand_name, generic, dosage_form, manufacturer)
            SELECT brand_id, brand_name, generic, COALESCE(dosage_form, ''), COALESCE(manufacturer, '')
            FROM medicines
            """
        )

        conn.commit()
        conn.execute("VACUUM")

    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"Built {DB_PATH} ({size_mb:.2f} MB)")
    print(f"Brands (deduplicated): {len(brands_dedup):,} | Generics: {len(generics_map):,}")


if __name__ == "__main__":
    build_database()
