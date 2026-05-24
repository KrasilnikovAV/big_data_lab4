from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .config import load_config
from .data import (
    DEFAULT_DATASET_SERVICE,
    DatasetFrameService,
)
from .model import DEFAULT_PIPELINE_FACTORY, TextClassificationPipelineFactory


class TrainingArtifactsWriter:
    @staticmethod
    def _prepare_output_path(path: Path | str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def write_model(self, pipeline: object, output_model_path: Path | str) -> None:
        output_path = self._prepare_output_path(output_model_path)
        joblib.dump(pipeline, output_path)

    def write_metrics(self, metrics: dict[str, float | int], metrics_path: Path | str) -> None:
        output_path = self._prepare_output_path(metrics_path)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def write_submission(self, submission: pd.DataFrame, submission_path: Path | str) -> None:
        output_path = self._prepare_output_path(submission_path)
        submission.to_csv(output_path, index=False)


class BBCNewsTrainer:
    def __init__(
        self,
        data_service: DatasetFrameService | None = None,
        pipeline_factory: TextClassificationPipelineFactory | None = None,
        artifacts_writer: TrainingArtifactsWriter | None = None,
    ) -> None:
        self.data_service = data_service or DEFAULT_DATASET_SERVICE
        self.pipeline_factory = pipeline_factory or DEFAULT_PIPELINE_FACTORY
        self.artifacts_writer = artifacts_writer or TrainingArtifactsWriter()

    def train_and_evaluate(
        self,
        config_path: Path | str,
        output_model_path: Path | str,
        metrics_path: Path | str,
        submission_path: Path | str | None = None,
    ) -> dict[str, float | int]:
        app_config = load_config(config_path)
        train_frame = self.data_service.load_training_frame(app_config.data)
        texts, labels = self.data_service.extract_features_targets(
            train_frame, app_config.data.text_column, app_config.data.target_column
        )

        x_train, x_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=app_config.split.test_size,
            random_state=app_config.split.random_state,
            stratify=labels,
        )

        pipeline = self.pipeline_factory.build(app_config.model, app_config.vectorizer)
        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_val)
        metrics: dict[str, float | int] = {
            "accuracy": float(accuracy_score(y_val, predictions)),
            "f1_macro": float(f1_score(y_val, predictions, average="macro")),
            "f1_weighted": float(f1_score(y_val, predictions, average="weighted")),
            "n_train": len(x_train),
            "n_validation": len(x_val),
        }

        self.artifacts_writer.write_model(pipeline, output_model_path)
        self.artifacts_writer.write_metrics(metrics, metrics_path)

        if submission_path is not None:
            submission = self._build_submission_frame(pipeline, app_config)
            self.artifacts_writer.write_submission(submission, submission_path)

        return metrics

    def _build_submission_frame(self, pipeline: object, app_config) -> pd.DataFrame:
        test_frame = self.data_service.load_inference_frame(app_config.data)
        test_texts = self.data_service.extract_texts(test_frame, app_config.data.text_column)
        test_predictions = pipeline.predict(test_texts)
        return pd.DataFrame(
            {
                app_config.data.id_column: test_frame[app_config.data.id_column],
                app_config.data.target_column: test_predictions,
            }
        )


DEFAULT_TRAINER = BBCNewsTrainer()


def train_and_evaluate(
    config_path: Path | str,
    output_model_path: Path | str,
    metrics_path: Path | str,
    submission_path: Path | str | None = None,
) -> dict[str, float | int]:
    return DEFAULT_TRAINER.train_and_evaluate(
        config_path=config_path,
        output_model_path=output_model_path,
        metrics_path=metrics_path,
        submission_path=submission_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BBC News classifier.")
    parser.add_argument("--config", default="config.ini", help="Path to config.ini file.")
    parser.add_argument(
        "--output-model",
        default="artifacts/model.joblib",
        help="Path to save trained model.",
    )
    parser.add_argument(
        "--metrics",
        default="artifacts/metrics.json",
        help="Path to save validation metrics.",
    )
    parser.add_argument(
        "--submission",
        default="artifacts/submission.csv",
        help="Path to save predictions for test dataset.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = train_and_evaluate(
        config_path=args.config,
        output_model_path=args.output_model,
        metrics_path=args.metrics,
        submission_path=args.submission,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
