"""celine-forecasting — open-source energy forecasting for the CELINE platform.

This is the top-level namespace package. Pipeline-specific functionality lives
in subpackages:

* :mod:`celine.forecasting.core` — shared infrastructure (config, DB, tracking,
  weather, evaluation metrics).
* :mod:`celine.forecasting.meter` — per-device smart-meter forecasting pipeline
  (LightGBM + CQR prediction intervals).
* :mod:`celine.forecasting.rec` — REC-aggregate energy forecasting pipeline
  (LightGBM quantile models + conformal calibration).
"""

from __future__ import annotations

__version__ = "0.1.0"
