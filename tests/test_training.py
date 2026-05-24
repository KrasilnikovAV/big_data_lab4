from __future__ import annotations

import pandas as pd

from bbc_news.train import train_and_evaluate


def test_train_and_evaluate_creates_artifacts(tmp_path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    model_path = tmp_path / "artifacts" / "model.joblib"
    metrics_path = tmp_path / "artifacts" / "metrics.json"
    submission_path = tmp_path / "artifacts" / "submission.csv"

    train_rows = []
    for i in range(20):
        if i % 2 == 0:
            train_rows.append(
                {
                    "ArticleId": i,
                    "Text": f"team won championship game {i}",
                    "Category": "sport",
                }
            )
        else:
            train_rows.append(
                {
                    "ArticleId": i,
                    "Text": f"stock market and business growth {i}",
                    "Category": "business",
                }
            )
    pd.DataFrame(train_rows).to_csv(train_path, index=False)

    pd.DataFrame(
        [
            {"ArticleId": 101, "Text": "market update for investors"},
            {"ArticleId": 102, "Text": "football team won again"},
            {"ArticleId": 103, "Text": "global economy report"},
        ]
    ).to_csv(test_path, index=False)

    config_path = tmp_path / "config.ini"
    config_path.write_text(
        (
            "[data]\n"
            f"train_path = {train_path}\n"
            f"test_path = {test_path}\n"
            "text_column = Text\n"
            "target_column = Category\n"
            "id_column = ArticleId\n\n"
            "[split]\n"
            "test_size = 0.2\n"
            "random_state = 42\n\n"
            "[vectorizer]\n"
            "lowercase = true\n"
            "ngram_min = 1\n"
            "ngram_max = 2\n"
            "min_df = 1\n"
            "max_df = 1.0\n"
            "sublinear_tf = true\n\n"
            "[model]\n"
            "algorithm = logistic_regression\n"
            "c = 1.0\n"
            "max_iter = 500\n"
            "n_jobs = 1\n"
        ),
        encoding="utf-8",
    )

    metrics = train_and_evaluate(
        config_path=config_path,
        output_model_path=model_path,
        metrics_path=metrics_path,
        submission_path=submission_path,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert submission_path.exists()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0

    submission = pd.read_csv(submission_path)
    assert list(submission.columns) == ["ArticleId", "Category"]
    assert len(submission) == 3
