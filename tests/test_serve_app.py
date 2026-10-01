"""Tests for the Chronos-2 FastAPI app (fake pipeline, no torch)."""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from conftest import make_serve_payload
from fastapi.testclient import TestClient

from celine.forecasting.serve.app import MAX_BODY_BYTES, create_app
from celine.forecasting.serve.contract import ForecastRequest, SeriesOut
from celine.forecasting.serve.model import ChronosServingModel, ModelCard

TOKEN = "test-token-123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakePipeline:
    def predict_quantiles(self, inputs, **kwargs):
        h, nq = kwargs["prediction_length"], len(kwargs["quantile_levels"])
        return [np.ones((1, h, nq), dtype=np.float32) for _ in inputs], None


@pytest.fixture
def model(tmp_path: Path, serve_card_dict: dict) -> ChronosServingModel:
    (tmp_path / "model_card.json").write_text(json.dumps(serve_card_dict))
    return ChronosServingModel(tmp_path, pipeline=FakePipeline())


@pytest.fixture
def client(model: ChronosServingModel) -> TestClient:
    return TestClient(create_app(model, TOKEN))


def test_health_needs_no_auth(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "model_name": "chronos2-fleet",
        "model_version": "fleet-test",
        "device": "injected",
        "dtype": "float32",
        "golden": None,
    }


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": TOKEN},
    ],
)
def test_model_and_forecast_require_bearer_token(
    client: TestClient, serve_payload: dict, headers: dict
) -> None:
    assert client.get("/model", headers=headers).status_code == 401
    assert client.post("/forecast", json=serve_payload, headers=headers).status_code == 401


def test_model_returns_card(client: TestClient, serve_card_dict: dict) -> None:
    resp = client.get("/model", headers=AUTH)
    assert resp.status_code == 200
    runtime = {"device": "injected", "dtype": "float32", "golden": None}
    assert resp.json() == {**serve_card_dict, **runtime}


def test_forecast_ok(client: TestClient) -> None:
    body = make_serve_payload(n_series=2, context=48, horizon=24, quantiles=[0.1, 0.5, 0.9])
    resp = client.post("/forecast", json=body, headers=AUTH)
    assert resp.status_code == 200
    out = resp.json()
    assert out["model_name"] == "chronos2-fleet"
    assert out["model_version"] == "fleet-test"
    assert out["origin"] == "2026-09-01T00:00:00Z"
    assert out["horizon"] == 24
    assert out["quantiles"] == [0.1, 0.5, 0.9]
    assert isinstance(out["latency_ms"], int) and out["latency_ms"] >= 0
    assert [s["id"] for s in out["series"]] == ["s0", "s1"]
    assert set(out["series"][0]["quantiles"]) == {"0.1", "0.5", "0.9"}
    assert len(out["series"][0]["quantiles"]["0.5"]) == 24


def test_forecast_insufficient_history(client: TestClient) -> None:
    body = make_serve_payload(n_series=2)
    body["series"][1]["target"] = [None] * len(body["series"][1]["target"])
    out = client.post("/forecast", json=body, headers=AUTH).json()
    assert out["series"][1] == {"id": "s1", "status": "insufficient_history", "quantiles": None}
    assert out["series"][0]["status"] == "ok"


def test_forecast_rejects_bad_series_id(client: TestClient, serve_payload: dict) -> None:
    serve_payload["series"][0]["id"] = "c2g-000000000"
    assert client.post("/forecast", json=serve_payload, headers=AUTH).status_code == 422


def test_forecast_rejects_quantiles_outside_card(client: TestClient, serve_payload: dict) -> None:
    serve_payload["quantiles"] = [0.05, 0.5]
    resp = client.post("/forecast", json=serve_payload, headers=AUTH)
    assert resp.status_code == 422
    assert "quantiles" in json.dumps(resp.json()["detail"])


def test_forecast_rejects_length_mismatch(client: TestClient, serve_payload: dict) -> None:
    serve_payload["weather"]["future"]["temperature_2m"].append(1.0)
    assert client.post("/forecast", json=serve_payload, headers=AUTH).status_code == 422


def test_forecast_rejects_horizon_above_card(client: TestClient) -> None:
    body = make_serve_payload(horizon=49)
    assert client.post("/forecast", json=body, headers=AUTH).status_code == 422


def test_forecast_rejects_all_null_weather(client: TestClient, serve_payload: dict) -> None:
    for part in ("past", "future"):
        n = len(serve_payload["weather"][part]["cloud_cover"])
        serve_payload["weather"][part]["cloud_cover"] = [None] * n
    assert client.post("/forecast", json=serve_payload, headers=AUTH).status_code == 422


def test_forecast_rejects_malformed_json(client: TestClient) -> None:
    resp = client.post(
        "/forecast", content=b"{not json", headers={**AUTH, "Content-Type": "application/json"}
    )
    assert resp.status_code == 422


def test_default_body_limit_is_20_mb() -> None:
    assert MAX_BODY_BYTES == 20 * 1024 * 1024


def test_forecast_rejects_oversized_body(model: ChronosServingModel) -> None:
    client = TestClient(create_app(model, TOKEN, max_body_bytes=1000))
    body = make_serve_payload(n_series=5)
    assert len(json.dumps(body)) > 1000
    assert client.post("/forecast", json=body, headers=AUTH).status_code == 413


def test_forecast_rejects_oversized_streamed_body(model: ChronosServingModel) -> None:
    client = TestClient(create_app(model, TOKEN, max_body_bytes=1000))
    payload = json.dumps(make_serve_payload(n_series=5)).encode()

    def chunks():
        for i in range(0, len(payload), 256):
            yield payload[i : i + 256]

    resp = client.post(
        "/forecast", content=chunks(), headers={**AUTH, "Content-Type": "application/json"}
    )
    assert resp.status_code == 413


def test_empty_token_refused(model: ChronosServingModel) -> None:
    with pytest.raises(ValueError, match="token"):
        create_app(model, "")


def test_logs_never_contain_target_values(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    body = make_serve_payload(n_series=1)
    body["series"][0]["target"] = [987.654321] * len(body["series"][0]["target"])
    with caplog.at_level(logging.DEBUG):
        assert client.post("/forecast", json=body, headers=AUTH).status_code == 200
    assert caplog.records, "expected a request log line"
    assert "987.654" not in caplog.text


class _SlowPredictor:
    """Detects overlapping forecast calls."""

    def __init__(self, card: ModelCard) -> None:
        self.card = card
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def card_payload(self) -> dict:
        return self.card.model_dump()

    def runtime_info(self) -> dict:
        return {"device": "fake", "dtype": "float32"}

    def forecast(self, request: ForecastRequest) -> list[SeriesOut]:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._guard:
            self.active -= 1
        return [
            SeriesOut(id=s.id, status="insufficient_history", quantiles=None)
            for s in request.series
        ]


def test_model_calls_are_serialised(serve_card_dict: dict) -> None:
    predictor = _SlowPredictor(ModelCard.model_validate(serve_card_dict))
    client = TestClient(create_app(predictor, TOKEN))
    body = make_serve_payload(n_series=1)
    with ThreadPoolExecutor(4) as pool:
        codes = list(
            pool.map(
                lambda _: client.post("/forecast", json=body, headers=AUTH).status_code, range(4)
            )
        )
    assert codes == [200] * 4
    assert predictor.max_active == 1
