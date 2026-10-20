import io
import logging
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from threading import Lock

import joblib
import matplotlib
import mlflow
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

for folder in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/45/"
    "heart+disease.zip"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target",
]

# Use a database-backed store for modern MLflow versions.
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').resolve()}")
mlflow.set_experiment("Heart-Disease-Classification")

def prepare_data():
    """Download the public Cleveland dataset if not already present."""
    output = PROCESSED_DIR / "heart_disease.csv"
    if output.exists():
        print(f"Using existing dataset: {output}")
        return output

    print("Downloading UCI Heart Disease dataset...")
    request = urllib.request.Request(
        DATA_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()

    with zipfile.ZipFile(io.BytesIO(content)) as archive, archive.open("processed.cleveland.data") as source:
            df = pd.read_csv(
                source, header=None, names=COLUMNS, na_values="?"
            )

    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(output, index=False)
    print(f"Prepared dataset: {output}")
    return output

def train_pipeline():
    """Explore data, train models, track metrics and save best model."""
    df = pd.read_csv(prepare_data())

    print("\n--- Dataset overview ---")
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Target distribution:\n", df["target"].value_counts().sort_index())
    print("\nDescriptive statistics:\n", df.describe().round(2))

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target")
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "target_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Feature Correlations")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_heatmap.png")
    plt.close()

    X = df.drop(columns="target")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200, random_state=42,
                class_weight="balanced"
            )),
        ]),
    }

    results = []
    best_auc, best_model, best_name = -1, None, None
    best_predictions = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
            mlflow.log_param("model_type", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_metrics(metrics)

            results.append({"model": name, **metrics})
            print(f"\n--- {name} ---")
            for key, value in metrics.items():
                print(f"{key}: {value:.3f}")
            print(classification_report(
                y_test, predictions, zero_division=0
            ))

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_model = model
                best_name = name
                best_predictions = predictions

    pd.DataFrame(results).to_csv(
        REPORT_DIR / "model_metrics.csv", index=False
    )
    joblib.dump(best_model, MODEL_DIR / "heart_disease_model.joblib")
    (MODEL_DIR / "model_info.txt").write_text(
        f"Selected model: {best_name}\n"
        f"Test ROC-AUC: {best_auc:.4f}\n"
        "Educational use only; not for clinical diagnosis.\n",
        encoding="utf-8",
    )

    cm = confusion_matrix(y_test, best_predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png")
    plt.close()

    print("\nModel comparison:\n", pd.DataFrame(results).round(3).to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print("Model and metrics saved.")

def load_model():
    path = MODEL_DIR / "heart_disease_model.joblib"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run training first.",
        )
    return joblib.load(path)

app = FastAPI(
    title="Heart Disease ML API",
    description="Educational demonstration only; not for clinical use.",
    version="1.0.0",
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart_disease_api")
_metrics_lock = Lock()
_request_counts = Counter()
_request_duration_seconds = Counter()


@app.middleware("http")
async def collect_request_metrics(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        method = request.method
        path = request.url.path
        with _metrics_lock:
            _request_counts[(method, path, status_code)] += 1
            _request_duration_seconds[(method, path)] += elapsed
        logger.info(
            "method=%s path=%s status=%s duration_seconds=%.6f",
            method, path, status_code, elapsed,
        )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = [
        "# HELP api_requests_total Total API requests by method, path and status.",
        "# TYPE api_requests_total counter",
        "# HELP api_request_duration_seconds_total Total request duration in seconds.",
        "# TYPE api_request_duration_seconds_total counter",
    ]
    with _metrics_lock:
        for (method, path, status), count in sorted(_request_counts.items()):
            lines.append(
                f'api_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )
        for (method, path), duration in sorted(_request_duration_seconds.items()):
            lines.append(
                f'api_request_duration_seconds_total{{method="{method}",path="{path}"}} {duration:.6f}'
            )
    return "\n".join(lines) + "\n"

class HeartDiseaseInput(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: float = Field(ge=3, le=7)

@app.get("/")
def root():
    return {
        "message": "Heart Disease ML API is running",
        "docs": "/docs",
        "purpose": "Educational demonstration only",
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: HeartDiseaseInput):
    model = load_model()
    features = pd.DataFrame([data.model_dump()], columns=COLUMNS[:-1])
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    return {
        "prediction": prediction,
        "probability_class_1": round(probability, 4),
        "model": "Selected baseline model",
        "note": "Educational output only; not a medical diagnosis.",
    }

def run_self_tests():
    """Run basic automated checks for the API."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Heart Disease ML API is running"
    print("PASS: Root endpoint")

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("PASS: Health endpoint")

    sample = {
        "age": 55, "sex": 1, "cp": 3, "trestbps": 130,
        "chol": 250, "fbs": 0, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3,
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction"] in (0, 1)
    assert 0 <= result["probability_class_1"] <= 1
    print("PASS: Prediction endpoint")

    response = client.post("/predict", json={**sample, "age": 500})
    assert response.status_code == 422
    print("PASS: Input validation")
    print("All self-tests passed.")

if __name__ == "__main__":
    train_pipeline()