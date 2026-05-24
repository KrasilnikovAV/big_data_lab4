from __future__ import annotations

from pathlib import Path

import yaml
from ansible.parsing.vault import VaultLib, VaultSecret

from bbc_news.secrets import load_ansible_vault, load_secret_value


def _write_vault(path: Path, payload: dict[str, str], password: str) -> None:
    vault = VaultLib([("default", VaultSecret(password.encode("utf-8")))])
    encrypted = vault.encrypt(yaml.safe_dump(payload, sort_keys=True).encode("utf-8"))
    path.write_bytes(encrypted)


def test_load_ansible_vault_reads_encrypted_payload(tmp_path: Path) -> None:
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

    payload = load_ansible_vault(vault_file=vault_file, password_file=password_file)

    assert payload["CLICKHOUSE_USER"] == "vault_user"
    assert payload["CLICKHOUSE_PASSWORD"] == "vault_password"


def test_load_secret_value_reads_value_from_vault(tmp_path: Path) -> None:
    vault_file = tmp_path / "clickhouse.vault.yml"
    password_file = tmp_path / ".vault_pass.txt"
    password_file.write_text("test-pass\n", encoding="utf-8")
    _write_vault(
        vault_file,
        {
            "CLICKHOUSE_USER": "vault_user",
        },
        password="test-pass",
    )

    assert (
        load_secret_value(
            "CLICKHOUSE_USER",
            vault_file=vault_file,
            password_file=password_file,
        )
        == "vault_user"
    )


def test_load_secret_value_returns_default_for_missing_key(tmp_path: Path) -> None:
    vault_file = tmp_path / "clickhouse.vault.yml"
    password_file = tmp_path / ".vault_pass.txt"
    password_file.write_text("test-pass\n", encoding="utf-8")
    _write_vault(vault_file, {}, password="test-pass")

    assert (
        load_secret_value(
            "CLICKHOUSE_PASSWORD",
            default="fallback",
            vault_file=vault_file,
            password_file=password_file,
        )
        == "fallback"
    )
