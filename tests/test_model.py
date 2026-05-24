from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from bbc_news.config import ModelConfig, VectorizerConfig
from bbc_news.model import build_pipeline


def _vectorizer_config() -> VectorizerConfig:
    return VectorizerConfig(
        lowercase=True,
        ngram_min=1,
        ngram_max=2,
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
    )


def test_build_pipeline_with_linear_svc() -> None:
    pipeline = build_pipeline(
        ModelConfig(algorithm="linear_svc", c=1.0, max_iter=1000, n_jobs=-1),
        _vectorizer_config(),
    )
    assert isinstance(pipeline.named_steps["clf"], LinearSVC)


def test_build_pipeline_with_logistic_regression() -> None:
    pipeline = build_pipeline(
        ModelConfig(algorithm="logistic_regression", c=1.0, max_iter=1000, n_jobs=-1),
        _vectorizer_config(),
    )
    assert isinstance(pipeline.named_steps["clf"], LogisticRegression)


def test_build_pipeline_with_unknown_algorithm_raises() -> None:
    with pytest.raises(ValueError):
        build_pipeline(
            ModelConfig(algorithm="unknown", c=1.0, max_iter=1000, n_jobs=-1),
            _vectorizer_config(),
        )
