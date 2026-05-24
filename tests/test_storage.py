from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from ansible.parsing.vault import VaultLib, VaultSecret

from bbc_news.storage import (
    ClickHousePredictionStore,
    ClickHouseSettings,
    PredictionLogRecord,
    DEFAULT_CLICKHOUSE_DATABASE,
    DEFAULT_CLICKHOUSE_HOST,
    DEFAULT_CLICKHOUSE_PORT,
    load_clickhouse_settings,
)


class FakeQueryResult:
    def __init__(self, result_set):
        self.result_set = result_set


class FakeClient:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.inserts: list[tuple[str, list[list[object]], list[str]]] = []
        self.queries: list[str] = []
        self.next_result = FakeQueryResult([])

    def command(self, sql: str) -> None:
        self.commands.append(sql)

    def insert(self, table: str, data: list[list[object]], column_names: list[str]) -> None:
        self.inserts.append((table, data, column_names))

    def query(self, sql: str) -> FakeQueryResult:
        self.queries.append(sql)
        return self.next_result


def _write_vault(path: Path, payload: dict[str, str], password: str) -> None:
    vault = VaultLib([("default", VaultSecret(password.encode("utf-8")))])
    encrypted = vault.encrypt(yaml.safe_dump(payload, sort_keys=True).encode("utf-8"))
    path.write_bytes(encrypted)


def test_load_clickhouse_settings_requires_all_fields_from_vault(tmp_path: Path) -> None:
    vault_file = tmp_path / "clickhouse.vault.yml"
    password_file = tmp_path / ".vault_pass.txt"
    password_file.write_text("test-pass\n", encoding="utf-8")
    _write_vault(vault_file, {"CLICKHOUSE_USER": "app_user"}, password="test-pass")

    with pytest.raises(ValueError, match="CLICKHOUSE_PASSWORD"):
        load_clickhouse_settings(vault_file=vault_file, password_file=password_file)


def test_load_clickhouse_settings_reads_credentials_from_vault(tmp_path: Path) -> None:
    vault_file = tmp_path / "clickhouse.vault.yml"
    password_file = tmp_path / ".vault_pass.txt"
    password_file.write_text("test-pass\n", encoding="utf-8")
    _write_vault(
        vault_file,
        {
            "CLICKHOUSE_USER": "vault_user",
            "CLICKHOUSE_PASSWORD": "vault_password",
        },
        password="test-pass",
    )

    settings = load_clickhouse_settings(vault_file=vault_file, password_file=password_file)

    assert settings.host == DEFAULT_CLICKHOUSE_HOST
    assert settings.port == DEFAULT_CLICKHOUSE_PORT
    assert settings.database == DEFAULT_CLICKHOUSE_DATABASE
    assert settings.username == "vault_user"
    assert settings.password == "vault_password"


def test_clickhouse_store_creates_schema_and_inserts_predictions() -> None:
    fake_client = FakeClient()
    store = ClickHousePredictionStore(
        ClickHouseSettings(
            host="clickhouse",
            port=8123,
            username="app_user",
            password="secret",
            database="bbc_news",
        ),
        client=fake_client,
    )

    store.ensure_ready()
    store.save_predictions(["one", "two"], ["sport", "business"])

    assert len(fake_client.commands) == 3
    assert fake_client.inserts[0][0] == "bbc_news.prediction_logs"
    assert fake_client.inserts[0][2] == [
        "request_id",
        "created_at",
        "row_position",
        "input_text",
        "predicted_label",
    ]
    assert len(fake_client.inserts[0][1]) == 2


def test_clickhouse_store_reads_recent_predictions() -> None:
    fake_client = FakeClient()
    fake_client.next_result = FakeQueryResult(
        [
            (
                "request-1",
                0,
                "stock market gains",
                "business",
                datetime(2026, 4, 2, 10, 0, tzinfo=timezone.utc),
            )
        ]
    )
    store = ClickHousePredictionStore(
        ClickHouseSettings(
            host="clickhouse",
            port=8123,
            username="app_user",
            password="secret",
            database="bbc_news",
        ),
        client=fake_client,
    )

    records = store.fetch_recent_predictions(limit=1)

    assert records == [
        PredictionLogRecord(
            request_id="request-1",
            row_position=0,
            input_text="stock market gains",
            predicted_label="business",
            created_at="2026-04-02T10:00:00+00:00",
        )
    ]
