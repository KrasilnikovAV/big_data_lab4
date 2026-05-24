from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import ModelConfig, VectorizerConfig


class TextClassificationPipelineFactory:
    def build(self, model_config: ModelConfig, vectorizer_config: VectorizerConfig) -> Pipeline:
        vectorizer = TfidfVectorizer(
            lowercase=vectorizer_config.lowercase,
            ngram_range=(vectorizer_config.ngram_min, vectorizer_config.ngram_max),
            min_df=vectorizer_config.min_df,
            max_df=vectorizer_config.max_df,
            sublinear_tf=vectorizer_config.sublinear_tf,
        )
        classifier = self._build_classifier(model_config)
        return Pipeline([("tfidf", vectorizer), ("clf", classifier)])

    @staticmethod
    def _build_classifier(model_config: ModelConfig):
        if model_config.algorithm == "linear_svc":
            return LinearSVC(C=model_config.c)
        if model_config.algorithm == "logistic_regression":
            return LogisticRegression(
                C=model_config.c,
                max_iter=model_config.max_iter,
                n_jobs=model_config.n_jobs,
            )
        raise ValueError(
            "Unsupported algorithm. Use one of: linear_svc, logistic_regression."
        )


DEFAULT_PIPELINE_FACTORY = TextClassificationPipelineFactory()


def build_pipeline(model_config: ModelConfig, vectorizer_config: VectorizerConfig) -> Pipeline:
    return DEFAULT_PIPELINE_FACTORY.build(model_config, vectorizer_config)
