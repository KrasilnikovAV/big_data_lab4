FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY config.ini .
COPY scenario.json .
COPY secrets ./secrets
COPY src ./src
COPY data ./data
COPY scripts ./scripts

RUN python -m bbc_news.train \
    --config config.ini \
    --output-model artifacts/model.joblib \
    --metrics artifacts/metrics.json \
    --submission artifacts/submission.csv

EXPOSE 8000

CMD ["uvicorn", "bbc_news.api:app", "--host", "0.0.0.0", "--port", "8000"]
