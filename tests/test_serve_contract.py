"""Tests for the Chronos-2 HTTP contract (request/response models, card checks)."""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import pytest
from conftest import make_serve_payload
from pydantic import ValidationError

from celine.forecasting.serve.contract import (
    CardViolation,
    ForecastRequest,
    ForecastResponse,
    SeriesOut,
    quantile_key,
    resolve_quantiles,
    validate_against_card,
)
from celine.forecasting.serve.model import ModelCard


def _req(body: dict) -> ForecastRequest:
    return ForecastRequest.model_validate(body)


def test_valid_request_parses(serve_payload: dict) -> None:
    req = _req(serve_payload)
    assert req.origin == datetime(2026, 9, 1, tzinfo=UTC)
    assert req.context_length == 48
    assert req.quantiles is None
    assert [s.id for s in req.series] == ["s0", "s1", "s2"]


def test_origin_with_offset_is_normalised_to_utc(serve_payload: dict) -> None:
    serve_payload["origin"] = "2026-09-01T02:00:00+02:00"
    assert _req(serve_payload).origin == datetime(2026, 9, 1, tzinfo=UTC)
    assert _req(serve_payload).origin.utcoffset().total_seconds() == 0


@pytest.mark.parametrize("origin", ["2026-09-01T00:00:00", "2026-09-01T00:30:00Z"])
def test_origin_must_be_aware_and_hour_aligned(serve_payload: dict, origin: str) -> None:
    serve_payload["origin"] = origin
    with pytest.raises(ValidationError):
        _req(serve_payload)


@pytest.mark.parametrize("bad_id", ["c2g-000000000", "s", "s123456", "S1", "s1 ", "x1"])
def test_series_id_regex_rejects(serve_payload: dict, bad_id: str) -> None:
    serve_payload["series"][0]["id"] = bad_id
    with pytest.raises(ValidationError):
        _req(serve_payload)


def test_series_id_regex_accepts_five_digits(serve_payload: dict) -> None:
    serve_payload["series"][0]["id"] = "s99999"
    assert _req(serve_payload).series[0].id == "s99999"


def test_duplicate_ids_rejected(serve_payload: dict) -> None:
    serve_payload["series"][1]["id"] = "s0"
    with pytest.raises(ValidationError, match="unique"):
        _req(serve_payload)


def test_series_count_limits() -> None:
    body = make_serve_payload(n_series=1)
    body["series"] = []
    with pytest.raises(ValidationError):
        _req(body)
    body = make_serve_payload(n_series=1, context=2, horizon=1)
    body["series"] = [{"id": f"s{i}", "target": [1.0, 2.0]} for i in range(2001)]
    with pytest.raises(ValidationError):
        _req(body)


def test_target_lengths_must_match(serve_payload: dict) -> None:
    serve_payload["series"][2]["target"] = serve_payload["series"][2]["target"][:-1]
    with pytest.raises(ValidationError, match="same length"):
        _req(serve_payload)


def test_empty_target_rejected() -> None:
    body = make_serve_payload(n_series=1)
    body["series"][0]["target"] = []
    with pytest.raises(ValidationError):
        _req(body)


def test_past_weather_length_must_equal_context(serve_payload: dict) -> None:
    serve_payload["weather"]["past"]["cloud_cover"].append(1.0)
    with pytest.raises(ValidationError, match="weather.past"):
        _req(serve_payload)


def test_future_weather_length_must_equal_horizon(serve_payload: dict) -> None:
    serve_payload["weather"]["future"]["cloud_cover"].pop()
    with pytest.raises(ValidationError, match="weather.future"):
        _req(serve_payload)


def test_past_and_future_weather_keys_must_match(serve_payload: dict) -> None:
    del serve_payload["weather"]["future"]["cloud_cover"]
    with pytest.raises(ValidationError, match="keys"):
        _req(serve_payload)


def test_horizon_must_be_positive(serve_payload: dict) -> None:
    serve_payload["horizon"] = 0
    with pytest.raises(ValidationError):
        _req(serve_payload)


@pytest.mark.parametrize("qs", [[0.0, 0.5], [0.5, 1.0], [0.5, 0.5], []])
def test_quantiles_shape_rules(serve_payload: dict, qs: list) -> None:
    serve_payload["quantiles"] = qs
    with pytest.raises(ValidationError):
        _req(serve_payload)


def test_quantiles_are_sorted(serve_payload: dict) -> None:
    serve_payload["quantiles"] = [0.9, 0.1, 0.5]
    assert _req(serve_payload).quantiles == [0.1, 0.5, 0.9]


def test_nulls_allowed_in_target_and_weather(serve_payload: dict) -> None:
    serve_payload["series"][0]["target"][3] = None
    serve_payload["weather"]["past"]["temperature_2m"][0] = None
    req = _req(serve_payload)
    assert req.series[0].target[3] is None


def test_unknown_top_level_field_rejected(serve_payload: dict) -> None:
    serve_payload["extra"] = 1
    with pytest.raises(ValidationError):
        _req(serve_payload)


# ── card-dependent checks ──────────────────────────────────────────────────────


def test_card_checks_pass_for_valid_request(serve_payload: dict, serve_card_dict: dict) -> None:
    validate_against_card(_req(serve_payload), ModelCard.model_validate(serve_card_dict))


def test_horizon_above_card_max_rejected(serve_card_dict: dict) -> None:
    body = make_serve_payload(horizon=49)
    with pytest.raises(CardViolation, match="max_horizon"):
        validate_against_card(_req(body), ModelCard.model_validate(serve_card_dict))


def test_context_above_card_context_length_rejected(serve_card_dict: dict) -> None:
    body = make_serve_payload(context=97)
    with pytest.raises(CardViolation, match="context_length"):
        validate_against_card(_req(body), ModelCard.model_validate(serve_card_dict))


def test_quantiles_must_be_subset_of_card(serve_payload: dict, serve_card_dict: dict) -> None:
    serve_payload["quantiles"] = [0.1, 0.5, 0.95]
    with pytest.raises(CardViolation, match="quantiles"):
        validate_against_card(_req(serve_payload), ModelCard.model_validate(serve_card_dict))


def test_weather_keys_must_equal_card(serve_payload: dict, serve_card_dict: dict) -> None:
    for part in ("past", "future"):
        serve_payload["weather"][part]["wind_speed"] = copy.copy(
            serve_payload["weather"][part]["cloud_cover"]
        )
    with pytest.raises(CardViolation, match="weather"):
        validate_against_card(_req(serve_payload), ModelCard.model_validate(serve_card_dict))


def test_entirely_null_weather_column_rejected(serve_payload: dict, serve_card_dict: dict) -> None:
    for part in ("past", "future"):
        col = serve_payload["weather"][part]["temperature_2m"]
        serve_payload["weather"][part]["temperature_2m"] = [None] * len(col)
    with pytest.raises(CardViolation, match="temperature_2m"):
        validate_against_card(_req(serve_payload), ModelCard.model_validate(serve_card_dict))


def test_resolve_quantiles_defaults_to_card(serve_payload: dict, serve_card_dict: dict) -> None:
    card = ModelCard.model_validate(serve_card_dict)
    assert resolve_quantiles(_req(serve_payload), card) == [0.1, 0.25, 0.5, 0.75, 0.9]
    serve_payload["quantiles"] = [0.5, 0.1]
    assert resolve_quantiles(_req(serve_payload), card) == [0.1, 0.5]


def test_response_serialisation() -> None:
    resp = ForecastResponse(
        model_name="m",
        model_version="v",
        origin=datetime(2026, 9, 1, tzinfo=UTC),
        horizon=2,
        quantiles=[0.1, 0.5],
        latency_ms=12,
        series=[
            SeriesOut(id="s0", status="ok", quantiles={"0.1": [0.0, 1.0], "0.5": [1.0, 2.0]}),
            SeriesOut(id="s7", status="insufficient_history", quantiles=None),
        ],
    )
    out = resp.model_dump(mode="json")
    assert out["origin"] == "2026-09-01T00:00:00Z"
    assert out["series"][1] == {"id": "s7", "status": "insufficient_history", "quantiles": None}


def test_quantile_key_format() -> None:
    assert [quantile_key(q) for q in (0.1, 0.25, 0.5, 0.9)] == ["0.1", "0.25", "0.5", "0.9"]
