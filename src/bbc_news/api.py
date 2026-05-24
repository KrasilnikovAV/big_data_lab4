from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .kafka import (
    NullPredictionProducer,
    PredictionProducer,
    build_prediction_events,
    build_prediction_producer,
    prediction_event_to_record,
)
from .predict import DEFAULT_PREDICTION_SERVICE, PredictionModel, load_model
from .storage import (
    NullPredictionStore,
    PredictionLogRecord,
    PredictionStore,
    build_prediction_store,
)

MODEL: PredictionModel | None = None
PREDICTION_STORE: PredictionStore = NullPredictionStore()
PREDICTION_PRODUCER: PredictionProducer = NullPredictionProducer()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL, PREDICTION_STORE, PREDICTION_PRODUCER
    try:
        MODEL = load_model("artifacts/model.joblib")
    except FileNotFoundError:
        MODEL = None
    PREDICTION_STORE = build_prediction_store()
    PREDICTION_STORE.ensure_ready()
    PREDICTION_PRODUCER = build_prediction_producer()
    try:
        yield
    finally:
        PREDICTION_PRODUCER.close()


app = FastAPI(title="BBC News Classifier API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    encoding: Literal["plain", "base64"] = "plain"

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        if any(str(text).strip() == "" for text in texts):
            raise ValueError("Texts must not contain blank values.")
        return texts


class PredictResponse(BaseModel):
    predictions: list[str]


class HealthResponse(BaseModel):
    status: str
    database: str
    kafka: str


class PredictionLogResponse(BaseModel):
    request_id: str
    row_position: int
    input_text: str
    predicted_label: str
    created_at: str


class PredictionHistoryResponse(BaseModel):
    records: list[PredictionLogResponse]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    payload = DEFAULT_PREDICTION_SERVICE.health_status(MODEL)
    payload["database"] = PREDICTION_STORE.health_status()
    payload["kafka"] = PREDICTION_PRODUCER.health_status()
    return HealthResponse(**payload)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        prepared_texts = DEFAULT_PREDICTION_SERVICE.decoder.decode(
            payload.texts,
            encoding=payload.encoding,
        )
        predictions = DEFAULT_PREDICTION_SERVICE.predict(
            MODEL,
            prepared_texts,
            encoding="plain",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    events = build_prediction_events(
        prepared_texts,
        predictions,
        model_version=_resolve_model_version(),
    )
    try:
        if PREDICTION_PRODUCER.enabled:
            PREDICTION_PRODUCER.publish_predictions(events)
        else:
            PREDICTION_STORE.save_prediction_records(
                [prediction_event_to_record(event) for event in events]
            )
    except Exception as exc:
        target = "Kafka" if PREDICTION_PRODUCER.enabled else "ClickHouse"
        raise HTTPException(
            status_code=503,
            detail=f"Prediction was not delivered to {target}.",
        ) from exc

    return PredictResponse(predictions=predictions)


@app.get("/predictions", response_model=PredictionHistoryResponse)
def get_recent_predictions(limit: int = 10) -> PredictionHistoryResponse:
    records = PREDICTION_STORE.fetch_recent_predictions(limit=limit)
    return PredictionHistoryResponse(
        records=[PredictionLogResponse(**_serialize_record(record)) for record in records]
    )


def _serialize_record(record: PredictionLogRecord) -> dict[str, str | int]:
    return {
        "request_id": record.request_id,
        "row_position": record.row_position,
        "input_text": record.input_text,
        "predicted_label": record.predicted_label,
        "created_at": record.created_at,
    }


def _resolve_model_version() -> str:
    settings = getattr(PREDICTION_PRODUCER, "settings", None)
    return str(getattr(settings, "model_version", "lab4"))
