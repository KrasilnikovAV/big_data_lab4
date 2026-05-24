from __future__ import annotations

import argparse
from pathlib import Path

from bbc_news.config import load_config
from bbc_news.data import DEFAULT_DATASET_SERVICE, DatasetFrameService
from bbc_news.storage import ClickHousePredictionStore, DatasetRecord, load_clickhouse_settings


class ClickHouseDatasetLoader:
    def __init__(self, data_service: DatasetFrameService | None = None) -> None:
        self.data_service = data_service or DEFAULT_DATASET_SERVICE

    @staticmethod
    def _frame_to_records(
        frame,
        id_column: str,
        text_column: str,
        target_column: str | None,
    ) -> list[DatasetRecord]:
        records: list[DatasetRecord] = []
        for row in frame.to_dict(orient="records"):
            category = None if target_column is None else str(row[target_column])
            records.append(
                DatasetRecord(
                    article_id=str(row[id_column]),
                    text=str(row[text_column]),
                    category=category,
                )
            )
        return records

    def load_data(self, config_path: Path) -> tuple[int, int]:
        settings = load_clickhouse_settings()
        store = ClickHousePredictionStore(settings)
        store.ensure_ready()

        app_config = load_config(config_path)
        train_frame = self.data_service.load_training_frame(app_config.data)
        test_frame = self.data_service.load_inference_frame(app_config.data)

        train_records = self._frame_to_records(
            train_frame,
            app_config.data.id_column,
            app_config.data.text_column,
            app_config.data.target_column,
        )
        test_records = self._frame_to_records(
            test_frame,
            app_config.data.id_column,
            app_config.data.text_column,
            None,
        )

        inserted_train = store.insert_dataset_rows("train", train_records)
        inserted_test = store.insert_dataset_rows("test", test_records)
        return inserted_train, inserted_test


DEFAULT_CLICKHOUSE_DATASET_LOADER = ClickHouseDatasetLoader()


def load_data_to_clickhouse(config_path: Path) -> tuple[int, int]:
    return DEFAULT_CLICKHOUSE_DATASET_LOADER.load_data(config_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load train/test CSV data into ClickHouse.")
    parser.add_argument("--config", default="config.ini", help="Path to project configuration file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inserted_train, inserted_test = load_data_to_clickhouse(Path(args.config))
    print(f"Inserted {inserted_train} train rows and {inserted_test} test rows into ClickHouse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
