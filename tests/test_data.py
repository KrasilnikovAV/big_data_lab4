from __future__ import annotations

import pandas as pd

from bbc_news.config import DataConfig
from bbc_news.data import extract_features_targets, load_training_frame


def test_load_training_frame_filters_invalid_rows(tmp_path) -> None:
    train_path = tmp_path / "train.csv"
    frame = pd.DataFrame(
        [
            {"ArticleId": 1, "Text": " economy update ", "Category": "business"},
            {"ArticleId": 2, "Text": " ", "Category": "sport"},
            {"ArticleId": 3, "Text": None, "Category": "tech"},
            {"ArticleId": 4, "Text": "team wins", "Category": "sport"},
        ]
    )
    frame.to_csv(train_path, index=False)

    cfg = DataConfig(
        train_path=train_path,
        test_path=train_path,
        text_column="Text",
        target_column="Category",
        id_column="ArticleId",
    )

    cleaned = load_training_frame(cfg)
    texts, labels = extract_features_targets(cleaned, cfg.text_column, cfg.target_column)

    assert len(cleaned) == 2
    assert texts == ["economy update", "team wins"]
    assert labels == ["business", "sport"]
