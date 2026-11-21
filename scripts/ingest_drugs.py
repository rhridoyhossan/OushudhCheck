from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

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


def build_database() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    df = df.replace(list(NULL_TOKENS), np.nan)

    if "indication_description" in df.columns:
        df["indication_description"] = df["indication_description"].str.replace(
            r"^.*?\.\.\.\s*Read more\s+", "", regex=True
        )

    df = df.dropna(subset=["brand_name", "generic"]).copy()
    df["brand_id"] = df["brand_id"].astype(int)

    df["brand_lower"] = df["brand_name"].str.lower()
    df["generic_lower"] = df["generic"].str.lower()

    brands_df = df[BRAND_COLS].copy()
    generics_df = df[GENERIC_COLS].groupby("generic_lower", as_index=False).first()

    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        brands_df.to_sql("brands", conn, index=False)
        generics_df.to_sql("generics", conn, index=False)

        cur = conn.cursor()
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
                b.generic_lower
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
    print(
        f"Brands: {len(brands_df):,} | Generics: {len(generics_df):,} | "
        f"View 'medicines' columns ({len(df.columns)}): {list(df.columns)}"
    )


if __name__ == "__main__":
    build_database()
