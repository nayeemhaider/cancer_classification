import sqlite3
import json
import pytest
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

# These tests assume the SQL pipeline has been run before the test suite.
DB_PATH = Path("data/cancer.db")
MODEL_DIR = Path("models")


def get_db():
    return sqlite3.connect(DB_PATH)


@pytest.fixture(scope="session")
def client():
    from src.api import app
    return TestClient(app)


@pytest.fixture(scope="session")
def db():
    conn = get_db()
    yield conn
    conn.close()


# SQL pipeline tests

def test_raw_table_exists(db):
    count = db.execute("SELECT COUNT(*) FROM raw_cancer").fetchone()[0]
    assert count > 0, "raw_cancer table should have rows"


def test_cleaned_table_exists(db):
    count = db.execute("SELECT COUNT(*) FROM cleaned_cancer").fetchone()[0]
    assert count > 0


def test_features_table_has_derived_cols(db):
    row = db.execute("SELECT lifestyle_burden, protective_score, env_exposure, bmi_category, is_senior FROM features LIMIT 1").fetchone()
    assert row is not None, "features table should have derived columns"


def test_no_nulls_in_cancer_type(db):
    nulls = db.execute("SELECT COUNT(*) FROM features WHERE cancer_type IS NULL").fetchone()[0]
    assert nulls == 0


def test_score_clamp_range(db):
    out_of_range = db.execute("""
        SELECT COUNT(*) FROM features
        WHERE smoking < 0 OR smoking > 10
           OR alcohol_use < 0 OR alcohol_use > 10
    """).fetchone()[0]
    assert out_of_range == 0, "Clamped columns should be in [0, 10]"


def test_five_cancer_types(db):
    types = db.execute("SELECT DISTINCT cancer_type FROM features").fetchall()
    assert len(types) == 5


def test_lifestyle_burden_formula(db):
    row = db.execute("""
        SELECT smoking + alcohol_use + diet_red_meat + diet_salted_processed AS expected,
               lifestyle_burden
        FROM features LIMIT 1
    """).fetchone()
    assert row[0] == row[1], "lifestyle_burden should equal sum of its components"


# API tests

def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "model" in resp.json()


def test_predict_valid(client):
    payload = {
        "age": 65, "gender": 1, "bmi": 26.5,
        "smoking": 7, "alcohol_use": 3, "physical_activity": 4,
        "fruit_veg_intake": 5, "diet_red_meat": 6, "diet_salted_processed": 4,
        "air_pollution": 8, "occupational_hazards": 3, "obesity": 6,
        "family_history": 0, "brca_mutation": 0, "h_pylori_infection": 0,
        "calcium_intake": 5, "physical_activity_level": 4,
        "overall_risk_score": 0.52, "risk_level_encoded": 1
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_cancer_type" in data
    assert "confidence" in data
    assert "all_probabilities" in data
    assert len(data["all_probabilities"]) == 5


def test_predict_returns_valid_class(client):
    payload = {
        "age": 55, "gender": 0, "bmi": 22.0,
        "smoking": 2, "alcohol_use": 1, "physical_activity": 8,
        "fruit_veg_intake": 9, "diet_red_meat": 2, "diet_salted_processed": 1,
        "air_pollution": 2, "occupational_hazards": 1, "obesity": 2,
        "family_history": 1, "brca_mutation": 1, "h_pylori_infection": 0,
        "calcium_intake": 8, "physical_activity_level": 9,
        "overall_risk_score": 0.25, "risk_level_encoded": 0
    }
    resp = client.post("/predict", json=payload)
    valid_classes = {"Breast", "Lung", "Colon", "Prostate", "Skin"}
    assert resp.json()["predicted_cancer_type"] in valid_classes


def test_distribution_endpoint(client):
    resp = client.get("/data/distribution")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 5
    total = sum(d["count"] for d in data)
    assert total > 1000


def test_risk_by_cancer_endpoint(client):
    resp = client.get("/data/risk-by-cancer")
    assert resp.status_code == 200
    assert len(resp.json()) == 5


def test_age_distribution_endpoint(client):
    resp = client.get("/data/age-distribution")
    assert resp.status_code == 200
    groups = [d["age_group"] for d in resp.json()]
    assert len(groups) > 0


def test_model_info_endpoint(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    info = resp.json()
    assert "best_model" in info
    assert "metrics" in info


def test_confusion_matrix_endpoint(client):
    resp = client.get("/model/confusion-matrix")
    assert resp.status_code == 200
    data = resp.json()
    assert "matrix" in data
    assert "class_names" in data
    assert len(data["matrix"]) == 5


def test_cleaned_sample_endpoint(client):
    resp = client.get("/data/cleaned-sample?limit=5")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 5
    assert "cancer_type" in rows[0]
    assert "lifestyle_burden" in rows[0]
