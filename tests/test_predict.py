from __future__ import annotations

import base64

import pytest

from bbc_news.predict import NewsPredictionService, RequestTextDecoder


class DummyModel:
    def __init__(self) -> None:
        self.received_texts: list[str] = []

    def predict(self, texts: list[str]) -> list[str]:
        self.received_texts = texts
        return ["sport" for _ in texts]


def test_request_text_decoder_normalizes_plain_texts() -> None:
    decoder = RequestTextDecoder()

    assert decoder.decode(["  headline  ", "match report"]) == [
        "headline",
        "match report",
    ]


def test_request_text_decoder_decodes_base64_texts() -> None:
    decoder = RequestTextDecoder()
    encoded = base64.b64encode("market update".encode("utf-8")).decode("utf-8")

    assert decoder.decode([encoded], encoding="base64") == ["market update"]


def test_prediction_service_passes_decoded_texts_to_model() -> None:
    service = NewsPredictionService()
    model = DummyModel()
    encoded = base64.b64encode("team won again".encode("utf-8")).decode("utf-8")

    predictions = service.predict(model, [encoded], encoding="base64")

    assert predictions == ["sport"]
    assert model.received_texts == ["team won again"]


def test_request_text_decoder_rejects_invalid_base64() -> None:
    decoder = RequestTextDecoder()

    with pytest.raises(ValueError, match="Invalid base64 payload."):
        decoder.decode(["broken-base64"], encoding="base64")
