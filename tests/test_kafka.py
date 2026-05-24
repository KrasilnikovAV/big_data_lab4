from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bbc_news.consumer import KafkaPredictionConsumerService
from bbc_news.kafka import (
    KafkaPredictionProducer,
    KafkaSettings,
    build_prediction_events,
    deserialize_prediction_event,
)


class FakeFuture:
    def get(self, timeout: float):
        return None


class FakeKafkaProducer:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.flushed = False
        self.closed = False

    def send(self, topic: str, value: dict, key: bytes):
        self.sent.append({"topic": topic, "value": value, "key": key})
        return FakeFuture()

    def flush(self, timeout: float) -> None:
        self.flushed = True

    def close(self, timeout: float) -> None:
        self.closed = True


class FakeStore:
    def __init__(self) -> None:
        self.ready = False
        self.records = []

    def ensure_ready(self) -> None:
        self.ready = True

    def health_status(self) -> str:
        return "ok"

    def save_predictions(self, texts: list[str], predictions: list[str]) -> None:
        raise AssertionError("Consumer must save records from Kafka payloads.")

    def save_prediction_records(self, records: list) -> None:
        self.records.extend(records)

    def fetch_recent_predictions(self, limit: int = 10) -> list:
        return self.records[:limit]


def test_build_prediction_events_uses_one_request_id_for_batch() -> None:
    created_at = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)

    events = build_prediction_events(
        ["one", "two"],
        ["sport", "business"],
        model_version="lab4-test",
        request_id="request-1",
        created_at=created_at,
    )

    assert [event.request_id for event in events] == ["request-1", "request-1"]
    assert [event.row_position for event in events] == [0, 1]
    assert events[0].created_at == "2026-05-24T12:00:00+00:00"
    assert events[1].predicted_label == "business"
    assert events[1].model_version == "lab4-test"


def test_build_prediction_events_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        build_prediction_events(["one"], ["sport", "business"])


def test_kafka_prediction_producer_sends_json_payloads() -> None:
    fake_client = FakeKafkaProducer()
    settings = KafkaSettings(
        enabled=True,
        predictions_topic="prediction-results",
        request_timeout_ms=1000,
    )
    producer = KafkaPredictionProducer(settings, producer_factory=lambda: fake_client)
    events = build_prediction_events(
        ["market update"],
        ["business"],
        request_id="request-1",
        created_at=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
    )

    producer.publish_predictions(events)
    producer.close()

    assert fake_client.sent == [
        {
            "topic": "prediction-results",
            "value": {
                "request_id": "request-1",
                "row_position": 0,
                "input_text": "market update",
                "predicted_label": "business",
                "created_at": "2026-05-24T12:00:00+00:00",
                "model_version": "lab4",
            },
            "key": b"request-1",
        }
    ]
    assert fake_client.flushed is True
    assert fake_client.closed is True


def test_consumer_service_stores_prediction_payload() -> None:
    store = FakeStore()
    service = KafkaPredictionConsumerService(
        settings=KafkaSettings(enabled=True),
        store=store,
    )

    processed = service.run_payloads(
        [
            {
                "request_id": "request-1",
                "row_position": 0,
                "input_text": "team won",
                "predicted_label": "sport",
                "created_at": "2026-05-24T12:00:00+00:00",
                "model_version": "lab4",
            }
        ]
    )

    assert processed == 1
    assert store.records[0].request_id == "request-1"
    assert store.records[0].input_text == "team won"
    assert store.records[0].predicted_label == "sport"


def test_deserialize_prediction_event_requires_json_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        deserialize_prediction_event(["not", "an", "object"])
