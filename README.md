# Cancer Type Classification — SQL · ML · MLflow · FastAPI

End-to-end cancer type classification project. The distinguishing feature is that
all data cleaning, EDA, and feature engineering are done **entirely in SQL** (SQLite)
before any Python ML code runs. This makes the data pipeline transparent, auditable,
and easy to replace with Postgres or any other SQL database later.

---

## Project Structure

```
cancer-classification/
├── data/
│   ├── cancer-risk-factors.csv   # raw source data
│   ├── cancer.db                 # SQLite database (created by pipeline)
│   └── cleaned_features.csv      # exported feature table
├── sql/
│   ├── 01_explore.sql            # EDA queries — run these to understand the data
│   ├── 02_clean.sql              # deduplication, null removal, outlier clamping
│   └── 03_features.sql           # derived feature engineering
├── src/
│   ├── sql_pipeline.py           # loads CSV → SQLite, runs SQL files, exports CSV
│   ├── train.py                  # model training with MLflow tracking
│   └── api.py                    # FastAPI application
├── frontend/
│   └── index.html                # standalone HTML frontend
├── tests/
│   └── test_pipeline.py          # pytest suite for SQL pipeline + API
└── models/                       # saved model artefacts (created by train.py)
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the SQL pipeline (loads CSV, cleans data, engineers features)
python -m src.sql_pipeline

# 3. Train models (RandomForest vs XGBoost, tracked in MLflow)
python -m src.train

# 4. Start the API
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# 5. Open the frontend
open frontend/index.html

# 6. View MLflow experiments
mlflow ui
# → http://localhost:5000

# 7. Run tests
pytest tests/ -v
```

---

## The SQL Pipeline — What You're Learning

### Why SQL for data cleaning?

In real projects, data often lives in a database from the start. Cleaning it in SQL
means the transformations are:
- **Repeatable** — re-run the SQL anytime
- **Auditable** — anyone with basic SQL can read what happened
- **Portable** — swap SQLite for Postgres with minimal changes

### What each SQL file does

`01_explore.sql`
Mirrors what you'd do in pandas `.describe()` and `.value_counts()`, but in SQL.
Run these queries manually in any SQLite client to build intuition about the data.

`02_clean.sql`
- Removes duplicate `patient_id` using `GROUP BY + MIN(rowid)` — SQL's idiomatic dedup
- `DELETE WHERE` removes impossible ages and BMI values
- `UPDATE SET MAX(0, MIN(10, col))` clamps scores to valid range — cleaner than multiple WHERE clauses
- `ALTER TABLE + CASE WHEN` adds an integer-encoded version of `risk_level`

`03_features.sql`
- `CREATE TABLE AS SELECT` materialises the feature table in one query
- Arithmetic expressions directly in SQL: `(smoking + alcohol_use + ...)` = `lifestyle_burden`
- `CASE WHEN` creates BMI category bins — same logic you'd write in pandas `.cut()`
- The final `GROUP BY` query is the EDA output — you can paste these results into a report

---

## Algorithm Choice

Two models are trained and compared:

**RandomForest** — Good baseline. Handles mixed numeric features well, robust to
outliers, and gives decent feature importance. With 5 balanced-ish classes, it tends
to perform well without much tuning.

**XGBoost** — Usually wins on tabular data. Gradient boosting handles the interaction
between features (e.g. high smoking + high air pollution → lung cancer) better than
a random forest in most cases. With the `class_weight="balanced"` equivalent in XGB
(`scale_pos_weight` or sample weights), it's competitive on imbalanced multiclass too.

The training script picks the winner by weighted F1 and saves it.

---

## Evaluation Metrics

| Metric | Why it's here |
|---|---|
| Accuracy | Easy to communicate, but misleading on imbalanced classes |
| Weighted F1 | Accounts for class imbalance — main selection metric |
| ROC-AUC (OvR) | One-vs-Rest AUC across all 5 classes — measures ranking quality |
| Per-class Precision/Recall | Tells you which cancer types are hardest to classify |
| Confusion Matrix | Shows exactly where the model confuses one type for another |

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Predict cancer type for a patient |
| GET | `/model/info` | Model name + evaluation metrics |
| GET | `/model/confusion-matrix` | Confusion matrix + class names |
| GET | `/data/distribution` | Cancer type counts (SQL GROUP BY) |
| GET | `/data/risk-by-cancer` | Avg risk scores per cancer type |
| GET | `/data/age-distribution` | Patient age group breakdown |
| GET | `/data/feature-averages` | Avg BMI, age, protective score per type |
| GET | `/data/cleaned-sample` | First N rows of cleaned feature table |
