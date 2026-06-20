import sqlite3
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

DB_PATH    = Path("data/cancer.db")
MODEL_DIR  = Path("models")
MLFLOW_DIR = Path("mlruns")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "age", "gender", "bmi", "smoking", "alcohol_use",
    "physical_activity", "fruit_veg_intake", "diet_red_meat",
    "diet_salted_processed", "air_pollution", "occupational_hazards",
    "obesity", "family_history", "brca_mutation", "h_pylori_infection",
    "calcium_intake", "physical_activity_level", "overall_risk_score",
    "risk_level_encoded", "lifestyle_burden", "protective_score",
    "env_exposure", "bmi_category", "is_senior"
]
TARGET = "cancer_type"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM features", conn)
    conn.close()
    return df


def encode_target(df):
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[TARGET])
    return df, le


def evaluate_model(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    auc  = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    return {
        "accuracy":                round(acc, 4),
        "f1_weighted":             round(f1, 4),
        "roc_auc_ovr":             round(auc, 4),
        "classification_report":   report,
        "confusion_matrix":        cm.tolist(),
        "class_names":             le.classes_.tolist(),
    }


def log_per_class_metrics(report, classes):
    for cls in classes:
        if cls not in report:
            continue
        mlflow.log_metric(f"{cls}_precision", round(report[cls]["precision"], 4))
        mlflow.log_metric(f"{cls}_recall",    round(report[cls]["recall"],    4))
        mlflow.log_metric(f"{cls}_f1",        round(report[cls]["f1-score"],  4))


def log_confusion_matrix_artifact(cm, classes):
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    tmp = Path("models/confusion_matrix.csv")
    cm_df.to_csv(tmp, encoding="utf-8")
    mlflow.log_artifact(str(tmp), artifact_path="evaluation")


def log_cv_fold_scores(cv_scores):
    for i, score in enumerate(cv_scores, start=1):
        mlflow.log_metric("cv_f1", round(score, 4), step=i)


def train():
    # Use a local file URI so mlflow ui always finds the same backend
    db_path = Path("mlflow.db").resolve()
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("cancer-type-classification")

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"  mlflow ui --backend-store-uri {tracking_uri}")

    df, le = encode_target(load_data())
    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
        print(f"  CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, le)

            # Core params
            mlflow.log_params(model.get_params())

            # Aggregate metrics
            mlflow.log_metric("accuracy",    metrics["accuracy"])
            mlflow.log_metric("f1_weighted", metrics["f1_weighted"])
            mlflow.log_metric("roc_auc_ovr", metrics["roc_auc_ovr"])
            mlflow.log_metric("cv_f1_mean",  round(float(cv_scores.mean()), 4))
            mlflow.log_metric("cv_f1_std",   round(float(cv_scores.std()),  4))

            # Per-class metrics (shows up as separate metric rows in the UI)
            log_per_class_metrics(metrics["classification_report"], le.classes_)

            # Fold-by-fold scores (shows up as a chart in the UI — step=fold number)
            log_cv_fold_scores(cv_scores)

            # Confusion matrix as a downloadable CSV artifact
            log_confusion_matrix_artifact(metrics["confusion_matrix"], le.classes_)

            # Log the model itself
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(model, name="model")

            print(f"  Test Accuracy : {metrics['accuracy']}")
            print(f"  Test F1       : {metrics['f1_weighted']}")
            print(f"  ROC-AUC (OvR) : {metrics['roc_auc_ovr']}")

        results[name] = {
            "model":    model,
            "metrics":  metrics,
            "cv_mean":  cv_scores.mean(),
        }

    best_name = max(results, key=lambda k: results[k]["metrics"]["f1_weighted"])
    best = results[best_name]
    print(f"\nBest model: {best_name} (F1={best['metrics']['f1_weighted']})")

    joblib.dump(best["model"], MODEL_DIR / "model.pkl")
    joblib.dump(le,            MODEL_DIR / "label_encoder.pkl")

    metadata = {
        "best_model":   best_name,
        "feature_cols": FEATURE_COLS,
        "classes":      le.classes_.tolist(),
        "metrics":      best["metrics"],
        "mlflow_uri":   tracking_uri,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nClassification Report:")
    print(classification_report(
        y_test, best["model"].predict(X_test), target_names=le.classes_
    ))

    print("\nConfusion Matrix:")
    cm = np.array(best["metrics"]["confusion_matrix"])
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

    print(f"\nTo open the MLflow UI run:")
    print(f"  mlflow ui --backend-store-uri {tracking_uri}")
    print(f"  then go to http://localhost:5000")

    return best["model"], le, metadata


if __name__ == "__main__":
    train()