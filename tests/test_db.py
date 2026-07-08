"""Tests for the SQL data loader (celine.forecasting.core.db).

All tests mock pd.read_sql so no real database is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.meter.ingest import normalize_meters
from celine.forecasting.meter.validation import validate_raw_schema


@pytest.fixture
def meter_df():
    """DataFrame matching the meter contract (device_id, ts, consumption_kw, production_kw)."""
    ts = pd.date_range("2025-03-01", periods=96, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "device_id": "dev-A",
            "ts": ts,
            "consumption_kw": np.full(96, 0.2),
            "production_kw": np.full(96, 0.1),
        }
    )


@pytest.fixture
def mapped_df():
    """DataFrame with non-standard column names requiring mapping."""
    ts = pd.date_range("2025-03-01", periods=96, freq="15min")  # naive
    return pd.DataFrame(
        {
            "sensor_ref": "dev-B",
            "ts": ts,
            "kwh_in": np.full(96, 0.3),
            "kwh_out": np.full(96, 0.0),
        }
    )


@pytest.fixture
def weather_df():
    """DataFrame matching the weather contract."""
    ts = pd.date_range("2025-03-01", periods=24, freq="h")
    return pd.DataFrame(
        {
            "datetime": ts,
            "global_tilted_irradiance": np.zeros(24),
            "shortwave_radiation": np.zeros(24),
            "cloud_cover": np.full(24, 30.0),
            "temperature_2m": np.full(24, 10.0),
        }
    )


@pytest.fixture
def mock_engine():
    return MagicMock()


class TestLoadMetersFromDb:
    def test_single_source(self, meter_df, mock_engine):
        from celine.forecasting.core.db import load_meters_from_db

        config = [{"table": "silver.meters"}]
        with patch("pandas.read_sql", return_value=meter_df):
            df = load_meters_from_db(
                config, engine=mock_engine,
                normalizer=normalize_meters, validator=validate_raw_schema,
            )

        assert "device_id" in df.columns
        assert "ts" in df.columns
        assert len(df) == 96

    def test_multi_source_dedup(self, meter_df, mock_engine):
        from celine.forecasting.core.db import load_meters_from_db

        df2 = meter_df.copy()
        df2["consumption_kw"] = 0.999

        config = [{"table": "silver.table_a"}, {"table": "silver.table_b"}]
        with patch("pandas.read_sql", side_effect=[meter_df, df2]):
            df = load_meters_from_db(
                config, engine=mock_engine,
                normalizer=normalize_meters, validator=validate_raw_schema,
            )

        assert len(df) == 96
        assert (df["consumption_kw"] == 0.2).all()

    def test_column_mapping(self, mapped_df, mock_engine):
        from celine.forecasting.core.db import load_meters_from_db

        config = [
            {
                "table": "silver.other",
                "columns": {
                    "sensor_ref": "device_id",
                    "kwh_in": "consumption_kw",
                    "kwh_out": "production_kw",
                },
                "assume_tz": "UTC",
            }
        ]
        with patch("pandas.read_sql", return_value=mapped_df):
            df = load_meters_from_db(
                config, engine=mock_engine,
                normalizer=normalize_meters, validator=validate_raw_schema,
            )

        assert "device_id" in df.columns
        assert "consumption_kw" in df.columns
        assert (df["device_id"] == "dev-B").all()

    def test_empty_sources_raises(self, mock_engine):
        from celine.forecasting.core.db import load_meters_from_db
        from celine.forecasting.core.schema import SchemaError

        config = [{"table": "silver.empty"}]
        with (
            patch("pandas.read_sql", return_value=pd.DataFrame()),
            pytest.raises(SchemaError, match="empty"),
        ):
            load_meters_from_db(config, engine=mock_engine)

    def test_invalid_table_name_raises(self, mock_engine):
        from celine.forecasting.core.db import load_meters_from_db

        config = [{"table": "silver.meters; DROP TABLE"}]
        with pytest.raises(ValueError, match="Invalid table name"):
            load_meters_from_db(config, engine=mock_engine)


class TestLoadWeatherFromDb:
    def test_weather_load(self, weather_df, mock_engine):
        from celine.forecasting.core.db import load_weather_from_db

        config = {"table": "gold.weather"}
        with patch("pandas.read_sql", return_value=weather_df):
            df = load_weather_from_db(
                config, engine=mock_engine,
                validator=validate_raw_schema,
            )

        assert "datetime" in df.columns
        assert len(df) == 24

    def test_weather_column_mapping(self, mock_engine):
        from celine.forecasting.core.db import load_weather_from_db

        ts = pd.date_range("2025-03-01", periods=24, freq="h")
        df_src = pd.DataFrame(
            {
                "ts_utc": ts,
                "temperature_2m": np.full(24, 10.0),
            }
        )
        config = {"table": "gold.weather", "columns": {"ts_utc": "datetime"}}
        with patch("pandas.read_sql", return_value=df_src):
            df = load_weather_from_db(
                config, engine=mock_engine,
                validator=validate_raw_schema,
            )

        assert "datetime" in df.columns


class TestBuildEngine:
    def test_explicit_uri(self):
        from celine.forecasting.core.db import build_engine

        engine = build_engine("sqlite:///:memory:")
        assert engine is not None

    def test_env_var_fallback(self):
        from celine.forecasting.core.db import build_engine

        with patch.dict("os.environ", {"DATABASE_URL": "sqlite:///:memory:"}):
            engine = build_engine()
            assert engine is not None

    def test_falls_back_to_settings(self):
        from celine.forecasting.core.db import build_engine

        engine = build_engine()
        assert engine is not None
