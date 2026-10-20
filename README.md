# Heart Disease MLOps Platform

Educational MLOps project using the UCI Heart Disease dataset.

## Features
- Data preprocessing and EDA
- Logistic Regression and Random Forest
- MLflow experiment tracking
- FastAPI prediction API
- Docker and Kubernetes deployment
- Request logging and Prometheus-format metrics
- Automated tests and GitHub Actions

## Run tests
python3 -m pip install -r requirements.txt
python3 -m pytest -q

## Kubernetes API
Use minikube service heart-disease-api --url to obtain the current service URL.
Test /health, /predict, and /metrics on that URL.

*Disclaimer:* Educational use only; not a medical diagnosis.
