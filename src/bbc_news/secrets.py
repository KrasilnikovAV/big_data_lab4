from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ansible.parsing.vault import VaultLib, VaultSecret


DEFAULT_VAULT_IDENTITY = "default"
DEFAULT_VAULT_FILE_CANDIDATES = (
    Path("/app/secrets/clickhouse.vault.yml"),
    Path("secrets/clickhouse.vault.yml"),
)
DEFAULT_VAULT_PASSWORD_FILE_CANDIDATES = (
    Path("/run/secrets/.vault_pass.txt"),
    Path("/run/secrets/ansible_vault_password"),
    Path("secrets/.vault_pass.txt"),
)


@dataclass(frozen=True)
class VaultFileConfig:
    vault_file: str | Path | None = None
    password_file: str | Path | None = None


class AnsibleVaultSecretStore:
    def __init__(
        self,
        vault_file: str | Path | None = None,
        password_file: str | Path | None = None,
        vault_identity: str = DEFAULT_VAULT_IDENTITY,
    ) -> None:
        self.file_config = VaultFileConfig(vault_file=vault_file, password_file=password_file)
        self.vault_identity = vault_identity

    def load_payload(self) -> dict[str, Any]:
        resolved_vault_file = self._resolve_vault_file(self.file_config.vault_file)
        resolved_password = self._resolve_password(self.file_config.password_file)
        cipher_text = resolved_vault_file.read_bytes()
        vault = VaultLib([(self.vault_identity, VaultSecret(resolved_password.encode("utf-8")))])
        plain_text = vault.decrypt(cipher_text).decode("utf-8")

        payload = yaml.safe_load(plain_text) or {}
        if not isinstance(payload, dict):
            raise ValueError("Ansible Vault payload must be a mapping.")
        return payload

    def get_value(self, name: str, default: str | None = None) -> str | None:
        value = self.load_payload().get(name, default)
        if value is None:
            return None
        return str(value).strip()

    @staticmethod
    def _resolve_vault_file(vault_file: str | Path | None) -> Path:
        if vault_file is not None:
            resolved = Path(vault_file).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Ansible Vault file was not found: {resolved}")
            return resolved

        for candidate in DEFAULT_VAULT_FILE_CANDIDATES:
            resolved = candidate.expanduser().resolve()
            if resolved.exists():
                return resolved

        raise FileNotFoundError(
            "Ansible Vault file was not found. Checked: "
            + ", ".join(str(path) for path in DEFAULT_VAULT_FILE_CANDIDATES)
        )

    @staticmethod
    def _resolve_password(password_file: str | Path | None) -> str:
        candidates = (
            [Path(password_file).expanduser()]
            if password_file is not None
            else list(DEFAULT_VAULT_PASSWORD_FILE_CANDIDATES)
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.exists():
                continue
            file_password = resolved.read_text(encoding="utf-8").strip()
            if file_password:
                return file_password

        raise FileNotFoundError(
            "Ansible Vault password file was not found or is empty. Checked: "
            + ", ".join(str(path) for path in candidates)
        )


def load_ansible_vault(
    vault_file: str | Path | None = None,
    password_file: str | Path | None = None,
) -> dict[str, Any]:
    return AnsibleVaultSecretStore(vault_file=vault_file, password_file=password_file).load_payload()


def load_secret_value(
    name: str,
    default: str | None = None,
    vault_file: str | Path | None = None,
    password_file: str | Path | None = None,
) -> str | None:
    return AnsibleVaultSecretStore(
        vault_file=vault_file,
        password_file=password_file,
    ).get_value(name, default=default)
