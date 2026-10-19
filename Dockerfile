FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY heart_disease_mlops.py .
COPY artifacts/ ./artifacts/

RUN pip install --no-cache-dir \
    pandas numpy scikit-learn matplotlib seaborn \
    mlflow fastapi uvicorn pydantic joblib httpx2

EXPOSE 8000

CMD ["uvicorn", "heart_disease_mlops:app", "--host", "0.0.0.0", "--port", "8000"]
