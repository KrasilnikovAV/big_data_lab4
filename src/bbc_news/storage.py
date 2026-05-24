from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import clickhouse_connect

from .secrets import load_secret_value

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_CLICKHOUSE_HOST = "clickhouse"
FALLBACK_CLICKHOUSE_HOST = "localhost"
DEFAULT_CLICKHOUSE_PORT = 8123
DEFAULT_CLICKHOUSE_DATABASE = "bbc_news"
DEFAULT_CLICKHOUSE_PREDICTIONS_TABLE = "prediction_logs"
DEFAULT_CLICKHOUSE_DATASET_TABLE = "dataset_rows"


@dataclass(frozen=True)
class ClickHouseSettings:
    host: str
    port: int
    username: str
    password: str
    database: str
    predictions_table: str = "prediction_logs"
    dataset_table: str = "dataset_rows"
    secure: bool = False


@dataclass(frozen=True)
class PredictionLogRecord:
    request_id: str
    row_position: int
    input_text: str
    predicted_label: str
    created_at: str


@dataclass(frozen=True)
class DatasetRecord:
    article_id: str
    text: str
    category: str | None = None


class PredictionStore(Protocol):
    def ensure_ready(self) -> None:
        ...

    def health_status(self) -> str:
        ...

    def save_predictions(self, texts: list[str], predictions: list[str]) -> None:
        ...

    def save_prediction_records(self, records: list[PredictionLogRecord]) -> None:
        ...

    def fetch_recent_predictions(self, limit: int = 10) -> list[PredictionLogRecord]:
        ...


def _require_secret(
    name: str,
    vault_file: str | Path | None = None,
    password_file: str | Path | None = None,
) -> str:
    value = load_secret_value(name, default="", vault_file=vault_file, password_file=password_file) or ""
    if not value:
        raise ValueError(f"Secret '{name}' must be present in Ansible Vault.")
    return value


def _validate_identifier(value: str, field_name: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must match pattern {IDENTIFIER_PATTERN.pattern!r}, got {value!r}."
        )
    return value


def load_clickhouse_settings(
    vault_file: str | Path | None = None,
    password_file: str | Path | None = None,
) -> ClickHouseSettings:
    return ClickHouseSettings(
        host=DEFAULT_CLICKHOUSE_HOST,
        port=DEFAULT_CLICKHOUSE_PORT,
        username=_require_secret("CLICKHOUSE_USER", vault_file=vault_file, password_file=password_file),
        password=_require_secret(
            "CLICKHOUSE_PASSWORD",
            vault_file=vault_file,
            password_file=password_file,
        ),
        database=_validate_identifier(DEFAULT_CLICKHOUSE_DATABASE, "database"),
        predictions_table=DEFAULT_CLICKHOUSE_PREDICTIONS_TABLE,
        dataset_table=DEFAULT_CLICKHOUSE_DATASET_TABLE,
        secure=False,
    )


def build_prediction_store() -> PredictionStore:
    try:
        settings = load_clickhouse_settings()
        return ClickHousePredictionStore(settings)
    except Exception:
        return NullPredictionStore()


def create_clickhouse_client(settings: ClickHouseSettings):
    last_error: Exception | None = None
    for host in dict.fromkeys((settings.host, FALLBACK_CLICKHOUSE_HOST)):
        for database in dict.fromkeys((settings.database, "default")):
            try:
                return clickhouse_connect.get_client(
                    host=host,
                    port=settings.port,
                    username=settings.username,
                    password=settings.password,
                    database=database,
                    secure=settings.secure,
                )
            except Exception as exc:
                last_error = exc

    assert last_error is not None
    raise last_error


class NullPredictionStore:
    def ensure_ready(self) -> None:
        return None

    def health_status(self) -> str:
        return "disabled"

    def save_predictions(self, texts: list[str], predictions: list[str]) -> None:
        return None

    def save_prediction_records(self, records: list[PredictionLogRecord]) -> None:
        return None

    def fetch_recent_predictions(self, limit: int = 10) -> list[PredictionLogRecord]:
        return []


class ClickHousePredictionStore:
    def __init__(self, settings: ClickHouseSettings, client=None) -> None:
        self.settings = settings
        self.client = client or create_clickhouse_client(settings)

    @property
    def _predictions_table_ref(self) -> str:
        return f"{self.settings.database}.{self.settings.predictions_table}"

    @property
    def _dataset_table_ref(self) -> str:
        return f"{self.settings.database}.{self.settings.dataset_table}"

    def ensure_ready(self) -> None:
        self.client.command(f"CREATE DATABASE IF NOT EXISTS {self.settings.database}")
        self.client.command(
            f"""
            CREATE TABLE IF NOT EXISTS {self._predictions_table_ref} (
                request_id UUID,
                created_at DateTime('UTC'),
                row_position UInt32,
                input_text String,
                predicted_label LowCardinality(String)
            )
            ENGINE = MergeTree
            ORDER BY (created_at, request_id, row_position)
            """
        )
        self.client.command(
            f"""
            CREATE TABLE IF NOT EXISTS {self._dataset_table_ref} (
                dataset_split LowCardinality(String),
                article_id String,
                text String,
                category Nullable(String),
                loaded_at DateTime('UTC')
            )
            ENGINE = MergeTree
            ORDER BY (dataset_split, article_id)
            """
        )

    def health_status(self) -> str:
        self.client.query("SELECT 1")
        return "ok"

    def save_predictions(self, texts: list[str], predictions: list[str]) -> None:
        if len(texts) != len(predictions):
            raise ValueError("Texts and predictions must have the same length.")

        created_at = datetime.now(timezone.utc).replace(microsecond=0)
        request_id = str(uuid.uuid4())
        records = [
            PredictionLogRecord(
                request_id=request_id,
                created_at=_format_timestamp(created_at),
                row_position=row_position,
                input_text=text,
                predicted_label=prediction,
            )
            for row_position, (text, prediction) in enumerate(zip(texts, predictions, strict=True))
        ]
        self.save_prediction_records(records)

    def save_prediction_records(self, records: list[PredictionLogRecord]) -> None:
        if not records:
            return

        rows = [
            [
                record.request_id,
                _parse_timestamp(record.created_at),
                record.row_position,
                record.input_text,
                record.predicted_label,
            ]
            for record in records
        ]
        self.client.insert(
            self._predictions_table_ref,
            rows,
            column_names=[
                "request_id",
                "created_at",
                "row_position",
                "input_text",
                "predicted_label",
            ],
        )

    def fetch_recent_predictions(self, limit: int = 10) -> list[PredictionLogRecord]:
        safe_limit = max(1, min(int(limit), 100))
        query = f"""
            SELECT request_id, row_position, input_text, predicted_label, created_at
            FROM {self._predictions_table_ref}
            ORDER BY created_at DESC, row_position DESC
            LIMIT {safe_limit}
        """
        rows = self.client.query(query).result_set
        return [
            PredictionLogRecord(
                request_id=str(row[0]),
                row_position=int(row[1]),
                input_text=str(row[2]),
                predicted_label=str(row[3]),
                created_at=_format_timestamp(row[4]),
            )
            for row in rows
        ]

    def insert_dataset_rows(self, dataset_split: str, records: list[DatasetRecord]) -> int:
        normalized_split = dataset_split.strip().lower()
        if normalized_split not in {"train", "test"}:
            raise ValueError("dataset_split must be either 'train' or 'test'.")
        if not records:
            return 0

        loaded_at = datetime.now(timezone.utc).replace(microsecond=0)
        rows = [
            [normalized_split, record.article_id, record.text, record.category, loaded_at]
            for record in records
        ]
        self.client.insert(
            self._dataset_table_ref,
            rows,
            column_names=["dataset_split", "article_id", "text", "category", "loaded_at"],
        )
        return len(rows)


def _format_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)
