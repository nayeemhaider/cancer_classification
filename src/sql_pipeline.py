import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/cancer.db")
RAW_CSV = Path("data/cancer-risk-factors.csv")
SQL_DIR = Path("sql")
CLEANED_CSV = Path("data/cleaned_features.csv")


def run_sql_file(conn, sql_file: Path):
    sql = sql_file.read_text(encoding="utf-8")
    # executescript handles multi-statement files correctly (SQLite built-in)
    # It auto-commits before running, so we use it for DDL/DML files.
    # For SELECT-only explore files we skip silently.
    try:
        conn.executescript(sql)
    except sqlite3.OperationalError as e:
        print(f"  Warning in {sql_file.name}: {e}")


def load_csv_to_sqlite(conn):
    df = pd.read_csv(RAW_CSV, encoding="utf-8")

    # Normalise column names to lowercase with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df.to_sql("raw_cancer", conn, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into raw_cancer table")
    return df


def export_features(conn):
    df = pd.read_sql("SELECT * FROM features", conn)
    df.to_csv(CLEANED_CSV, index=False, encoding="utf-8")
    print(f"Exported {len(df)} rows to {CLEANED_CSV}")
    return df


def run_pipeline():
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    print("Step 1: Loading CSV into SQLite...")
    load_csv_to_sqlite(conn)

    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        print(f"Step: Running {sql_file.name}...")
        run_sql_file(conn, sql_file)

    print("Step 4: Exporting feature table to CSV...")
    features_df = export_features(conn)

    conn.close()
    return features_df


if __name__ == "__main__":
    run_pipeline()
