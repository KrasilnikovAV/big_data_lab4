from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Protocol

import joblib
from sklearn.pipeline import Pipeline

from .data import normalize_payload_texts


class PredictionModel(Protocol):
    def predict(self, texts: list[str]) -> object:
        ...


@dataclass(slots=True)
class RequestTextDecoder:
    def decode(self, texts: Iterable[str], encoding: str = "plain") -> list[str]:
        raw_texts = list(texts)
        if encoding == "base64":
            raw_texts = [self._decode_base64_text(text) for text in raw_texts]
        elif encoding != "plain":
            raise ValueError("Unsupported encoding. Use 'plain' or 'base64'.")

        normalized_texts = normalize_payload_texts(raw_texts)
        if any(text == "" for text in normalized_texts):
            raise ValueError("Texts must not be blank.")
        return normalized_texts

    @staticmethod
    def _decode_base64_text(text: str) -> str:
        encoded_text = str(text).strip()
        if encoded_text == "":
            raise ValueError("Base64 text payload must not be blank.")

        try:
            decoded_bytes = base64.b64decode(encoded_text, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 payload.") from exc

        try:
            return decoded_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Base64 payload must decode to UTF-8 text.") from exc


@dataclass(slots=True)
class NewsPredictionService:
    decoder: RequestTextDecoder = field(default_factory=RequestTextDecoder)

    def health_status(self, model: PredictionModel | None) -> dict[str, str]:
        if model is None:
            return {"status": "model_not_loaded"}
        return {"status": "ok"}

    def predict(
        self,
        model: PredictionModel | None,
        texts: Iterable[str],
        encoding: str = "plain",
    ) -> list[str]:
        if model is None:
            raise RuntimeError("Model is not loaded.")

        prepared_texts = self.decoder.decode(texts, encoding=encoding)
        predictions = model.predict(prepared_texts)
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()
        return [str(label) for label in predictions]


DEFAULT_PREDICTION_SERVICE = NewsPredictionService()


@lru_cache(maxsize=1)
def load_model(model_path: str | Path) -> Pipeline:
    return joblib.load(model_path)


def predict_texts(
    model: PredictionModel,
    texts: Iterable[str],
    encoding: str = "plain",
) -> list[str]:
    return DEFAULT_PREDICTION_SERVICE.predict(model, texts, encoding=encoding)
