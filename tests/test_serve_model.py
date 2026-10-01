"""Tests for the Chronos-2 serving model (fake pipeline, no torch)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import SERVE_WEATHER, make_serve_payload
from pydantic import ValidationError

from celine.forecasting.serve import model as model_mod
from celine.forecasting.serve.contract import ForecastRequest
from celine.forecasting.serve.covariates import calendar_covariates
from celine.forecasting.serve.model import (
    ChronosServingModel,
    ModelCard,
    build_inputs,
)


class FakePipeline:
    """Records calls; returns deterministic, deliberately unsorted/negative quantiles."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict_quantiles(self, inputs, **kwargs):
        self.calls.append({"inputs": inputs, **kwargs})
        h = kwargs["prediction_length"]
        nq = len(kwargs["quantile_levels"])
        out = []
        for i, _ in enumerate(inputs):
            # descending across quantiles, and the first step negative
            base = np.arange(h, dtype=np.float32)[:, None] + i - 1.0
            out.append((base - np.arange(nq, dtype=np.float32)[None, :])[None, :, :])
        return out, [o[..., 0] for o in out]


@pytest.fixture
def model_dir(tmp_path: Path, serve_card_dict: dict) -> Path:
    (tmp_path / "model_card.json").write_text(json.dumps(serve_card_dict))
    return tmp_path


def _req(body: dict) -> ForecastRequest:
    return ForecastRequest.model_validate(body)


# ── ModelCard ─────────────────────────────────────────────────────────────────


def test_model_card_from_dir(model_dir: Path, serve_card_dict: dict) -> None:
    card = ModelCard.from_dir(model_dir)
    assert card.model_version == "fleet-test"
    assert card.context_length == 96
    assert card.weather_covariates == SERVE_WEATHER


def test_model_card_rejects_unknown_covariate(serve_card_dict: dict) -> None:
    serve_card_dict["covariates"].append("price")
    with pytest.raises(ValidationError, match="price"):
        ModelCard.model_validate(serve_card_dict)


def test_model_card_rejects_weather_not_in_covariates(serve_card_dict: dict) -> None:
    serve_card_dict["covariates"].remove("cloud_cover")
    with pytest.raises(ValidationError, match="cloud_cover"):
        ModelCard.model_validate(serve_card_dict)


def test_model_card_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ModelCard.from_dir(tmp_path)


# ── input building ────────────────────────────────────────────────────────────


def test_build_inputs_structure_and_order(serve_card_dict: dict) -> None:
    card = ModelCard.model_validate(serve_card_dict)
    body = make_serve_payload(n_series=2, context=48, horizon=24)
    body["series"][0]["target"][5] = None
    inputs, ids = build_inputs(_req(body), card)
    assert ids == ["s0", "s1"]
    first = inputs[0]
    assert first["target"].dtype == np.float32
    assert first["target"].shape == (48,)
    assert np.isnan(first["target"][5])
    assert list(first["past_covariates"]) == card.covariates
    assert list(first["future_covariates"]) == card.covariates
    for col in card.covariates:
        assert first["past_covariates"][col].shape == (48,)
        assert first["future_covariates"][col].shape == (24,)
        assert first["past_covariates"][col].dtype == np.float32


def test_build_inputs_covariate_values(serve_card_dict: dict) -> None:
    card = ModelCard.model_validate(serve_card_dict)
    body = make_serve_payload(n_series=1, context=48, horizon=24)
    body["weather"]["past"]["temperature_2m"][10:12] = [None, None]
    body["weather"]["future"]["cloud_cover"][0] = None
    req = _req(body)
    inputs, _ = build_inputs(req, card)
    grid = pd.date_range(req.origin - pd.Timedelta(hours=48), periods=72, freq="h")
    cal = calendar_covariates(grid, "Europe/Rome")
    past, future = inputs[0]["past_covariates"], inputs[0]["future_covariates"]
    np.testing.assert_allclose(past["hour_sin"], cal["hour_sin"].iloc[:48], atol=1e-6)
    np.testing.assert_allclose(future["dow_cos"], cal["dow_cos"].iloc[48:], atol=1e-6)
    # linear interpolation within the past column
    t = body["weather"]["past"]["temperature_2m"]
    np.testing.assert_allclose(
        past["temperature_2m"][10:12], np.linspace(t[9], t[12], 4)[1:3], rtol=1e-5
    )
    # a gap at the first future hour interpolates across the past/future boundary
    last_past = body["weather"]["past"]["cloud_cover"][-1]
    next_future = body["weather"]["future"]["cloud_cover"][1]
    assert future["cloud_cover"][0] == pytest.approx((last_past + next_future) / 2, rel=1e-5)


def test_build_inputs_skips_insufficient_history(serve_card_dict: dict) -> None:
    card = ModelCard.model_validate(serve_card_dict)
    body = make_serve_payload(n_series=3, context=48, horizon=24)
    body["series"][1]["target"] = [None] * 25 + [1.0] * 23  # 23 finite < 24
    body["series"][2]["target"] = [None] * 24 + [1.0] * 24  # exactly 24: ok
    _, ids = build_inputs(_req(body), card)
    assert ids == ["s0", "s2"]


# ── ChronosServingModel ───────────────────────────────────────────────────────


def test_forecast_calls_pipeline_and_postprocesses(model_dir: Path) -> None:
    fake = FakePipeline()
    m = ChronosServingModel(model_dir, batch_size=7, pipeline=fake)
    body = make_serve_payload(n_series=3, context=48, horizon=24, quantiles=[0.1, 0.5, 0.9])
    body["series"][1]["target"] = [None] * 48
    out = m.forecast(_req(body))

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["prediction_length"] == 24
    assert call["quantile_levels"] == [0.1, 0.5, 0.9]
    assert call["batch_size"] == 7
    assert call["context_length"] == 48
    assert len(call["inputs"]) == 2

    assert [s.id for s in out] == ["s0", "s1", "s2"]
    assert out[1].status == "insufficient_history" and out[1].quantiles is None
    s0 = out[0]
    assert s0.status == "ok"
    assert list(s0.quantiles) == ["0.1", "0.5", "0.9"]
    q = np.array([s0.quantiles[k] for k in ("0.1", "0.5", "0.9")])
    assert q.shape == (3, 24)
    assert (q >= 0).all()
    assert (np.diff(q, axis=0) >= 0).all()  # sorted across quantile levels
    # second forecastable input (s2) is mapped back by id, not position
    assert out[2].quantiles["0.9"][-1] == pytest.approx(23.0)


def test_forecast_defaults_to_card_quantiles(model_dir: Path) -> None:
    fake = FakePipeline()
    out = ChronosServingModel(model_dir, pipeline=fake).forecast(_req(make_serve_payload()))
    assert fake.calls[0]["quantile_levels"] == [0.1, 0.25, 0.5, 0.75, 0.9]
    assert fake.calls[0]["batch_size"] == 256
    assert list(out[0].quantiles) == ["0.1", "0.25", "0.5", "0.75", "0.9"]


def test_forecast_without_eligible_series_skips_pipeline(model_dir: Path) -> None:
    fake = FakePipeline()
    body = make_serve_payload(n_series=2)
    for s in body["series"]:
        s["target"] = [None] * len(s["target"])
    out = ChronosServingModel(model_dir, pipeline=fake).forecast(_req(body))
    assert fake.calls == []
    assert {s.status for s in out} == {"insufficient_history"}


def test_pipeline_is_loaded_lazily_once(model_dir: Path, monkeypatch) -> None:
    loads: list[tuple] = []

    def fake_loader(path, dtype, device):
        loads.append((path, dtype, device))
        return FakePipeline(), "cuda:0 Fake GPU"

    monkeypatch.setattr(model_mod, "_load_pipeline", fake_loader)
    m = ChronosServingModel(model_dir, dtype="bfloat16")
    assert loads == []
    m.forecast(_req(make_serve_payload()))
    m.forecast(_req(make_serve_payload()))
    assert loads == [(model_dir, "bfloat16", "auto")]


def test_invalid_dtype_rejected(model_dir: Path) -> None:
    with pytest.raises(ValueError, match="dtype"):
        ChronosServingModel(model_dir, dtype="int8")


def test_card_payload_is_raw_file_content(model_dir: Path, serve_card_dict: dict) -> None:
    m = ChronosServingModel(model_dir, pipeline=FakePipeline())
    assert m.card_payload() == serve_card_dict


# ── device selection ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("requested", "cuda_available", "expected"),
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cuda", True, "cuda"),
        ("cpu", True, "cpu"),
        ("cpu", False, "cpu"),
    ],
)
def test_resolve_device(requested: str, cuda_available: bool, expected: str) -> None:
    assert model_mod.resolve_device(requested, cuda_available) == expected


def test_resolve_device_cuda_without_gpu_fails_fast() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        model_mod.resolve_device("cuda", cuda_available=False)


def test_invalid_device_rejected(model_dir: Path) -> None:
    with pytest.raises(ValueError, match="device"):
        ChronosServingModel(model_dir, device="tpu")


def test_runtime_info_reports_loaded_device(model_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        model_mod, "_load_pipeline", lambda path, dtype, device: (FakePipeline(), "cuda:0 GB10")
    )
    m = ChronosServingModel(model_dir, dtype="bfloat16", device="cuda")
    assert m.runtime_info() == {"device": None, "dtype": "bfloat16", "golden": None}
    m.load()
    assert m.runtime_info() == {"device": "cuda:0 GB10", "dtype": "bfloat16", "golden": "absent"}


def test_load_propagates_missing_cuda(model_dir: Path, monkeypatch) -> None:
    def no_gpu(path, dtype, device):
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    monkeypatch.setattr(model_mod, "_load_pipeline", no_gpu)
    with pytest.raises(RuntimeError, match="CUDA"):
        ChronosServingModel(model_dir, device="cuda").load()
