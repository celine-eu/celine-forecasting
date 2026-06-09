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
