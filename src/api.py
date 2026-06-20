import sqlite3
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Cancer Type Classification API",
    description="Predict cancer type from patient risk factors. Data cleaning done via SQL.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_DIR = Path("models")
DB_PATH = Path("data/cancer.db")

model = joblib.load(MODEL_DIR / "model.pkl")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))

FEATURE_COLS = metadata["feature_cols"]


def get_db():
    return sqlite3.connect(DB_PATH)


class PatientInput(BaseModel):
    age: int = Field(..., ge=18, le=110)
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    bmi: float = Field(..., ge=10.0, le=70.0)
    smoking: int = Field(..., ge=0, le=10)
    alcohol_use: int = Field(..., ge=0, le=10)
    physical_activity: int = Field(..., ge=0, le=10)
    fruit_veg_intake: int = Field(..., ge=0, le=10)
    diet_red_meat: int = Field(..., ge=0, le=10)
    diet_salted_processed: int = Field(..., ge=0, le=10)
    air_pollution: int = Field(..., ge=0, le=10)
    occupational_hazards: int = Field(..., ge=0, le=10)
    obesity: int = Field(..., ge=0, le=10)
    family_history: int = Field(..., ge=0, le=1)
    brca_mutation: int = Field(..., ge=0, le=1)
    h_pylori_infection: int = Field(..., ge=0, le=1)
    calcium_intake: int = Field(..., ge=0, le=10)
    physical_activity_level: int = Field(..., ge=0, le=10)
    overall_risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level_encoded: int = Field(..., ge=0, le=2, description="0=Low, 1=Medium, 2=High")


def compute_derived_features(p: PatientInput) -> dict:
    lifestyle_burden = p.smoking + p.alcohol_use + p.diet_red_meat + p.diet_salted_processed
    protective_score = p.fruit_veg_intake + p.physical_activity
    env_exposure = p.air_pollution + p.occupational_hazards

    if p.bmi < 18.5:
        bmi_category = 0
    elif p.bmi < 25:
        bmi_category = 1
    elif p.bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3

    is_senior = 1 if p.age >= 65 else 0

    return {
        **p.model_dump(),
        "lifestyle_burden": lifestyle_burden,
        "protective_score": protective_score,
        "env_exposure": env_exposure,
        "bmi_category": bmi_category,
        "is_senior": is_senior
    }


@app.get("/")
def root():
    return {"message": "Cancer Classification API is running", "model": metadata["best_model"]}


@app.post("/predict")
def predict(patient: PatientInput):
    features = compute_derived_features(patient)
    row = pd.DataFrame([[features[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)

    prediction_idx = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    predicted_class = label_encoder.inverse_transform([prediction_idx])[0]

    prob_dict = {
        label_encoder.classes_[i]: round(float(probabilities[i]), 4)
        for i in range(len(label_encoder.classes_))
    }

    return {
        "predicted_cancer_type": predicted_class,
        "confidence": round(float(probabilities[prediction_idx]), 4),
        "all_probabilities": prob_dict
    }


@app.get("/model/info")
def model_info():
    return {
        "best_model": metadata["best_model"],
        "classes": metadata["classes"],
        "metrics": {
            k: v for k, v in metadata["metrics"].items()
            if k in ("accuracy", "f1_weighted", "roc_auc_ovr")
        }
    }


@app.get("/data/distribution")
def cancer_distribution():
    conn = get_db()
    rows = conn.execute(
        "SELECT cancer_type, COUNT(*) as count FROM features GROUP BY cancer_type ORDER BY count DESC"
    ).fetchall()
    conn.close()
    return [{"cancer_type": r[0], "count": r[1]} for r in rows]


@app.get("/data/risk-by-cancer")
def risk_by_cancer():
    conn = get_db()
    rows = conn.execute("""
        SELECT cancer_type,
               ROUND(AVG(overall_risk_score), 4) AS avg_risk,
               ROUND(AVG(smoking), 2)             AS avg_smoking,
               ROUND(AVG(lifestyle_burden), 2)    AS avg_lifestyle_burden,
               COUNT(*) AS n
        FROM features
        GROUP BY cancer_type
        ORDER BY avg_risk DESC
    """).fetchall()
    conn.close()
    cols = ["cancer_type", "avg_risk", "avg_smoking", "avg_lifestyle_burden", "n"]
    return [dict(zip(cols, r)) for r in rows]


@app.get("/data/age-distribution")
def age_distribution():
    conn = get_db()
    rows = conn.execute("""
        SELECT
            CASE
                WHEN age < 40 THEN 'Under 40'
                WHEN age BETWEEN 40 AND 54 THEN '40-54'
                WHEN age BETWEEN 55 AND 69 THEN '55-69'
                ELSE '70+'
            END AS age_group,
            COUNT(*) AS count
        FROM features
        GROUP BY age_group
        ORDER BY count DESC
    """).fetchall()
    conn.close()
    return [{"age_group": r[0], "count": r[1]} for r in rows]


@app.get("/data/feature-averages")
def feature_averages():
    conn = get_db()
    rows = conn.execute("""
        SELECT cancer_type,
               ROUND(AVG(age), 1)              AS avg_age,
               ROUND(AVG(bmi), 2)              AS avg_bmi,
               ROUND(AVG(protective_score), 2) AS avg_protective,
               ROUND(AVG(env_exposure), 2)     AS avg_env_exposure
        FROM features
        GROUP BY cancer_type
    """).fetchall()
    conn.close()
    cols = ["cancer_type", "avg_age", "avg_bmi", "avg_protective", "avg_env_exposure"]
    return [dict(zip(cols, r)) for r in rows]


@app.get("/data/cleaned-sample")
def cleaned_sample(limit: int = 20):
    conn = get_db()
    df = pd.read_sql(f"SELECT * FROM features LIMIT {limit}", conn)
    conn.close()
    return df.to_dict(orient="records")


@app.get("/model/confusion-matrix")
def get_confusion_matrix():
    return {
        "matrix": metadata["metrics"]["confusion_matrix"],
        "class_names": metadata["metrics"]["class_names"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
