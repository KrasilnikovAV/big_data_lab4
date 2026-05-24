from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    train_path: Path
    test_path: Path
    text_column: str
    target_column: str
    id_column: str


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    random_state: int


@dataclass(frozen=True)
class VectorizerConfig:
    lowercase: bool
    ngram_min: int
    ngram_max: int
    min_df: int
    max_df: float
    sublinear_tf: bool


@dataclass(frozen=True)
class ModelConfig:
    algorithm: str
    c: float
    max_iter: int
    n_jobs: int


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    split: SplitConfig
    vectorizer: VectorizerConfig
    model: ModelConfig


class AppConfigLoader:
    def __init__(self, path: Path | str) -> None:
        self.config_path = Path(path).resolve()

    def load(self) -> AppConfig:
        parser = ConfigParser()
        read_files = parser.read(self.config_path, encoding="utf-8")
        if not read_files:
            raise FileNotFoundError(f"Configuration file was not found: {self.config_path}")

        base_dir = self.config_path.parent

        data = DataConfig(
            train_path=self._resolve_path(base_dir, parser.get("data", "train_path")),
            test_path=self._resolve_path(base_dir, parser.get("data", "test_path")),
            text_column=parser.get("data", "text_column", fallback="Text"),
            target_column=parser.get("data", "target_column", fallback="Category"),
            id_column=parser.get("data", "id_column", fallback="ArticleId"),
        )

        split = SplitConfig(
            test_size=parser.getfloat("split", "test_size", fallback=0.2),
            random_state=parser.getint("split", "random_state", fallback=42),
        )

        vectorizer = VectorizerConfig(
            lowercase=parser.getboolean("vectorizer", "lowercase", fallback=True),
            ngram_min=parser.getint("vectorizer", "ngram_min", fallback=1),
            ngram_max=parser.getint("vectorizer", "ngram_max", fallback=2),
            min_df=parser.getint("vectorizer", "min_df", fallback=2),
            max_df=parser.getfloat("vectorizer", "max_df", fallback=0.95),
            sublinear_tf=parser.getboolean("vectorizer", "sublinear_tf", fallback=True),
        )

        model = ModelConfig(
            algorithm=parser.get("model", "algorithm", fallback="linear_svc").strip().lower(),
            c=parser.getfloat("model", "c", fallback=1.0),
            max_iter=parser.getint("model", "max_iter", fallback=2000),
            n_jobs=parser.getint("model", "n_jobs", fallback=-1),
        )

        return AppConfig(data=data, split=split, vectorizer=vectorizer, model=model)

    @staticmethod
    def _resolve_path(base_dir: Path, value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (base_dir / candidate).resolve()


def load_config(path: Path | str) -> AppConfig:
    return AppConfigLoader(path).load()
