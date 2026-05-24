from __future__ import annotations

import argparse
import json
import logging
from typing import Iterable

from .kafka import (
    KafkaSettings,
    deserialize_prediction_event,
    load_kafka_settings,
    prediction_event_to_record,
)
from .storage import ClickHousePredictionStore, PredictionStore, load_clickhouse_settings

LOGGER = logging.getLogger(__name__)


class KafkaPredictionConsumerService:
    def __init__(
        self,
        settings: KafkaSettings | None = None,
        store: PredictionStore | None = None,
    ) -> None:
        self.settings = settings or load_kafka_settings()
        self.store = store

    def handle_payload(self, payload: object) -> None:
        event = deserialize_prediction_event(payload)
        self._get_store().save_prediction_records([prediction_event_to_record(event)])
        LOGGER.info(
            "Stored prediction event request_id=%s row_position=%s",
            event.request_id,
            event.row_position,
        )

    def run(self, max_messages: int = 0) -> int:
        if not self.settings.enabled:
            raise RuntimeError("Kafka consumer requires KAFKA_ENABLED=true.")

        self._get_store().ensure_ready()
        consumer = self._create_consumer()
        processed = 0

        try:
            for message in consumer:
                try:
                    self.handle_payload(message.value)
                    consumer.commit()
                    processed += 1
                except Exception:
                    LOGGER.exception("Failed to process Kafka prediction message.")

                if max_messages > 0 and processed >= max_messages:
                    break
        finally:
            consumer.close()

        return processed

    def run_payloads(self, payloads: Iterable[object]) -> int:
        processed = 0
        for payload in payloads:
            self.handle_payload(payload)
            processed += 1
        return processed

    @staticmethod
    def _build_clickhouse_store() -> PredictionStore:
        store = ClickHousePredictionStore(load_clickhouse_settings())
        store.ensure_ready()
        return store

    def _get_store(self) -> PredictionStore:
        if self.store is None:
            self.store = self._build_clickhouse_store()
        return self.store

    def _create_consumer(self):
        from kafka import KafkaConsumer

        return KafkaConsumer(
            self.settings.predictions_topic,
            bootstrap_servers=_split_bootstrap_servers(self.settings.bootstrap_servers),
            group_id=self.settings.consumer_group,
            client_id=self.settings.consumer_client_id,
            auto_offset_reset=self.settings.auto_offset_reset,
            enable_auto_commit=False,
            value_deserializer=lambda payload: json.loads(payload.decode("utf-8")),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume prediction results from Kafka.")
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Stop after this many messages. Use 0 to run forever.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    service = KafkaPredictionConsumerService()
    service.run(max_messages=max(0, args.max_messages))
    return 0


def _split_bootstrap_servers(value: str) -> list[str]:
    servers = [server.strip() for server in value.split(",") if server.strip()]
    if not servers:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS must contain at least one host:port pair.")
    return servers


if __name__ == "__main__":
    raise SystemExit(main())
