# celine-forecasting

Open-source energy forecasting for smart meters and renewable energy communities.

## Packages

### celine.meter_forecasting

Per-device 48h energy forecasting using LightGBM with Conformalized Quantile Regression (CQR) prediction intervals.

- Full training pipeline: data cleaning, validation, feature engineering, model training, calibration, evaluation
- CLI: `meter-forecast validate|run|train`
- MLflow integration for experiment tracking and model versioning
- MLflow model serving via custom `pyfunc` wrapper

```bash
uv sync --extra mlflow

# Quick start: train on your meter data
meter-forecast run --meters data.csv --lat 46.07 --lon 11.12 --output out/
```

### Model backends

Forecasting strategies are **pluggable** behind a shared `Forecaster` interface.
The model-agnostic engine (data contracts, weather, CQR intervals, naive
baselines, per-horizon bias correction, evaluation, MLflow tracking + serving)
lives in `core/`; each strategy is a self-contained folder under `models/`.
LightGBM is the default backend; neural backends (IBM Granite TTM, Chronos, …)
plug in the same way behind optional dependency extras. Choose one per run:

```bash
meter-forecast train --meters data.csv --model lightgbm --scope per_device
```

`--scope pooled` trains one model per device-type group instead of one per device.

## Local MLflow Development

```bash
docker compose up -d
# MLflow UI at http://localhost:5000
# MinIO Console at http://localhost:9001 (minioadmin/minioadmin)

export MLFLOW_TRACKING_URI=http://localhost:5000
meter-forecast train --meters data.csv --output out/
```

## License

Apache 2.0
