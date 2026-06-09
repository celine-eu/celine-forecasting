"""Configuration loading for the meter-forecast pipeline.

The YAML config (``config/default_config.yaml``) is the single place every
tunable lives. It is loaded into a lightweight, typed :class:`ForecastConfig`
dataclass so the rest of the package never reaches into raw dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default_config.yaml"


@dataclass
class ForecastConfig:
    """Typed view over the YAML configuration.

    Attributes:
        raw: The original parsed dictionary (escape hatch for rarely-used keys).
        random_seed: Global RNG seed for reproducibility.
        local_tz: IANA timezone for calendar features (e.g. ``Europe/Rome``).
        targets: Target columns to forecast.
    """

    raw: dict[str, Any]
    random_seed: int
    local_tz: str
    targets: list[str]

    # --- sub-sections kept as dicts for flexibility ---
    cleaning: dict[str, Any] = field(default_factory=dict)
    sufficiency: dict[str, Any] = field(default_factory=dict)
    cv: dict[str, Any] = field(default_factory=dict)
    lgb_params: dict[str, Any] = field(default_factory=dict)
    cqr: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    backtest: dict[str, Any] = field(default_factory=dict)
    tracking: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- helpers
    @property
    def horizon_bands(self) -> dict[str, list[int]]:
        """Inclusive horizon ranges expanded to explicit hour lists."""
        bands = self.raw.get("horizon_bands", {})
        return {name: list(range(lo, hi + 1)) for name, (lo, hi) in bands.items()}

    @property
    def forecast_horizon(self) -> int:
        return int(self.raw.get("forecast_horizon", 48))

    @property
    def min_span_days(self) -> int:
        """Minimum energy span; falls back to ``cv.folds * cv.test_days + 14``."""
        explicit = self.sufficiency.get("min_span_days")
        if explicit is not None:
            return int(explicit)
        return int(self.cv["folds"] * self.cv["test_days"] + 14)

    def band_for_horizon(self, horizon: int) -> str:
        """Return the band name a given forecast horizon belongs to."""
        for name, hours in self.horizon_bands.items():
            if horizon in hours:
                return name
        raise ValueError(f"Horizon {horizon} is outside all configured bands")


def load_config(path: str | Path | None = None) -> ForecastConfig:
    """Load and validate the pipeline configuration.

    Args:
        path: Path to a YAML config file. Defaults to the packaged
            ``config/default_config.yaml``.

    Returns:
        A populated :class:`ForecastConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML is empty or missing required top-level keys.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not raw:
        raise ValueError(f"Config file is empty: {config_path}")

    required = {"random_seed", "local_tz", "targets"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    return ForecastConfig(
        raw=raw,
        random_seed=int(raw["random_seed"]),
        local_tz=str(raw["local_tz"]),
        targets=list(raw["targets"]),
        cleaning=raw.get("cleaning", {}),
        sufficiency=raw.get("sufficiency", {}),
        cv=raw.get("cv", {}),
        lgb_params=raw.get("lgb_params", {}),
        cqr=raw.get("cqr", {}),
        features=raw.get("features", {}),
        backtest=raw.get("backtest", {}),
        tracking=raw.get("tracking", {}),
    )
