from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bbc_news.secrets import AnsibleVaultSecretStore

CLICKHOUSE_USER_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
DEFAULT_VAULT_FILE = Path("/app/secrets/clickhouse.vault.yml")
DEFAULT_PASSWORD_FILE = Path("/run/secrets/.vault_pass.txt")
DEFAULT_OUTPUT_FILE = Path("/vault/clickhouse-user.xml")


class ClickHouseBootstrapper:
    def __init__(
        self,
        vault_file: Path = DEFAULT_VAULT_FILE,
        password_file: Path = DEFAULT_PASSWORD_FILE,
        output_file: Path = DEFAULT_OUTPUT_FILE,
    ) -> None:
        self.secret_store = AnsibleVaultSecretStore(vault_file=vault_file, password_file=password_file)
        self.output_file = output_file

    def run(self) -> Path:
        username = self._require_secret("CLICKHOUSE_USER")
        password = self._require_secret("CLICKHOUSE_PASSWORD")
        self._validate_username(username)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(self._build_xml(username, password), encoding="utf-8")
        if not self.output_file.exists() or self.output_file.stat().st_size == 0:
            raise RuntimeError(f"Generated ClickHouse config is missing or empty: {self.output_file}")
        return self.output_file

    def _require_secret(self, name: str) -> str:
        value = self.secret_store.get_value(name, default="")
        if not value:
            raise ValueError(f"{name} was not found in Vault payload.")
        return value

    @staticmethod
    def _validate_username(username: str) -> None:
        if not CLICKHOUSE_USER_PATTERN.fullmatch(username):
            raise ValueError("Unsupported CLICKHOUSE_USER value.")

    @staticmethod
    def _build_xml(username: str, password: str) -> str:
        root = ElementTree.Element("clickhouse")
        users = ElementTree.SubElement(root, "users")
        user = ElementTree.SubElement(users, username)
        ElementTree.SubElement(user, "password").text = password
        ElementTree.SubElement(user, "profile").text = "default"
        ElementTree.SubElement(user, "quota").text = "default"

        networks = ElementTree.SubElement(user, "networks")
        ElementTree.SubElement(networks, "ip").text = "::/0"

        grants = ElementTree.SubElement(user, "grants")
        ElementTree.SubElement(grants, "query").text = "GRANT ALL ON *.*"

        ElementTree.indent(root, space="  ")
        return ElementTree.tostring(root, encoding="unicode")


def main() -> int:
    output_path = ClickHouseBootstrapper().run()
    print(f"Generated ClickHouse user config at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
