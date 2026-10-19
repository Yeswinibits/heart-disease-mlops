from fastapi.testclient import TestClient

from heart_disease_mlops import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    payload = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability_class_1"] <= 1


def test_invalid_age():
    payload = {
        "age": 500,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
