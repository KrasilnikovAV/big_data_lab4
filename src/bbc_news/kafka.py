from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Protocol

from .storage import PredictionLogRecord

DEFAULT_KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"
DEFAULT_KAFKA_PREDICTIONS_TOPIC = "prediction-results"
DEFAULT_KAFKA_CONSUMER_GROUP = "bbc-news-prediction-consumers"
DEFAULT_MODEL_VERSION = "lab4"
TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class KafkaSettings:
    enabled: bool = False
    bootstrap_servers: str = DEFAULT_KAFKA_BOOTSTRAP_SERVERS
    predictions_topic: str = DEFAULT_KAFKA_PREDICTIONS_TOPIC
    consumer_group: str = DEFAULT_KAFKA_CONSUMER_GROUP
    producer_client_id: str = "bbc-news-api-producer"
    consumer_client_id: str = "bbc-news-consumer"
    model_version: str = DEFAULT_MODEL_VERSION
    request_timeout_ms: int = 10000
    max_block_ms: int = 10000
    auto_offset_reset: str = "earliest"


@dataclass(frozen=True)
class PredictionResultEvent:
    request_id: str
    row_position: int
    input_text: str
    predicted_label: str
    created_at: str
    model_version: str


class PredictionProducer(Protocol):
    @property
    def enabled(self) -> bool:
        ...

    def health_status(self) -> str:
        ...

    def publish_predictions(self, events: Iterable[PredictionResultEvent]) -> None:
        ...

    def close(self) -> None:
        ...


def load_kafka_settings() -> KafkaSettings:
    return KafkaSettings(
        enabled=_env_bool("KAFKA_ENABLED", default=False),
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_KAFKA_BOOTSTRAP_SERVERS),
        predictions_topic=os.getenv("KAFKA_PREDICTIONS_TOPIC", DEFAULT_KAFKA_PREDICTIONS_TOPIC),
        consumer_group=os.getenv("KAFKA_CONSUMER_GROUP", DEFAULT_KAFKA_CONSUMER_GROUP),
        producer_client_id=os.getenv("KAFKA_PRODUCER_CLIENT_ID", "bbc-news-api-producer"),
        consumer_client_id=os.getenv("KAFKA_CONSUMER_CLIENT_ID", "bbc-news-consumer"),
        model_version=os.getenv("MODEL_VERSION", DEFAULT_MODEL_VERSION),
        request_timeout_ms=_env_int("KAFKA_REQUEST_TIMEOUT_MS", default=10000),
        max_block_ms=_env_int("KAFKA_MAX_BLOCK_MS", default=10000),
        auto_offset_reset=os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest"),
    )


def build_prediction_events(
    texts: list[str],
    predictions: list[str],
    model_version: str = DEFAULT_MODEL_VERSION,
    request_id: str | None = None,
    created_at: datetime | None = None,
) -> list[PredictionResultEvent]:
    if len(texts) != len(predictions):
        raise ValueError("Texts and predictions must have the same length.")

    resolved_request_id = request_id or str(uuid.uuid4())
    resolved_created_at = created_at or datetime.now(timezone.utc).replace(microsecond=0)
    created_at_text = _format_utc_timestamp(resolved_created_at)

    return [
        PredictionResultEvent(
            request_id=resolved_request_id,
            row_position=row_position,
            input_text=text,
            predicted_label=prediction,
            created_at=created_at_text,
            model_version=model_version,
        )
        for row_position, (text, prediction) in enumerate(zip(texts, predictions, strict=True))
    ]


def prediction_event_to_record(event: PredictionResultEvent) -> PredictionLogRecord:
    return PredictionLogRecord(
        request_id=event.request_id,
        row_position=event.row_position,
        input_text=event.input_text,
        predicted_label=event.predicted_label,
        created_at=event.created_at,
    )


def deserialize_prediction_event(payload: object) -> PredictionResultEvent:
    if not isinstance(payload, dict):
        raise ValueError("Kafka prediction payload must be a JSON object.")

    try:
        return PredictionResultEvent(
            request_id=str(payload["request_id"]),
            row_position=int(payload["row_position"]),
            input_text=str(payload["input_text"]),
            predicted_label=str(payload["predicted_label"]),
            created_at=str(payload["created_at"]),
            model_version=str(payload.get("model_version", DEFAULT_MODEL_VERSION)),
        )
    except KeyError as exc:
        raise ValueError(f"Kafka prediction payload is missing field: {exc.args[0]}") from exc


def prediction_event_to_payload(event: PredictionResultEvent) -> dict[str, str | int]:
    return asdict(event)


def build_prediction_producer(settings: KafkaSettings | None = None) -> PredictionProducer:
    resolved_settings = settings or load_kafka_settings()
    if not resolved_settings.enabled:
        return NullPredictionProducer()
    return KafkaPredictionProducer(resolved_settings)


class NullPredictionProducer:
    @property
    def enabled(self) -> bool:
        return False

    def health_status(self) -> str:
        return "disabled"

    def publish_predictions(self, events: Iterable[PredictionResultEvent]) -> None:
        return None

    def close(self) -> None:
        return None


class KafkaPredictionProducer:
    def __init__(
        self,
        settings: KafkaSettings,
        producer_factory: Callable[[], object] | None = None,
    ) -> None:
        self.settings = settings
        self._producer_factory = producer_factory or self._create_producer
        self._producer: object | None = None

    @property
    def enabled(self) -> bool:
        return True

    def health_status(self) -> str:
        return "enabled"

    def publish_predictions(self, events: Iterable[PredictionResultEvent]) -> None:
        resolved_events = list(events)
        if not resolved_events:
            return

        producer = self._get_producer()
        futures = []
        for event in resolved_events:
            futures.append(
                producer.send(
                    self.settings.predictions_topic,
                    value=prediction_event_to_payload(event),
                    key=event.request_id.encode("utf-8"),
                )
            )

        for future in futures:
            future.get(timeout=self.settings.request_timeout_ms / 1000)
        producer.flush(timeout=self.settings.request_timeout_ms / 1000)

    def close(self) -> None:
        if self._producer is not None:
            self._producer.close(timeout=self.settings.request_timeout_ms / 1000)
            self._producer = None

    def _get_producer(self):
        if self._producer is None:
            self._producer = self._producer_factory()
        return self._producer

    def _create_producer(self):
        from kafka import KafkaProducer

        return KafkaProducer(
            bootstrap_servers=split_bootstrap_servers(self.settings.bootstrap_servers),
            client_id=self.settings.producer_client_id,
            value_serializer=lambda payload: json.dumps(payload, ensure_ascii=False).encode(
                "utf-8"
            ),
            retries=3,
            request_timeout_ms=self.settings.request_timeout_ms,
            max_block_ms=self.settings.max_block_ms,
        )


def split_bootstrap_servers(value: str) -> list[str]:
    servers = [server.strip() for server in value.split(",") if server.strip()]
    if not servers:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS must contain at least one host:port pair.")
    return servers


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _format_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()
