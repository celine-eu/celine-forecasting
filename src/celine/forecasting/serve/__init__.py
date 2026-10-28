"""Chronos-2 fleet inference server (HTTP contract in :mod:`.contract`).

Modules:
    covariates: torch-free calendar/weather covariates, identical to training.
    contract:   pydantic request/response models and model-card validation.
    model:      model card and the lazily loaded Chronos-2 predictor.
    app:        FastAPI application factory (``create_app``).

Start it with ``meter-forecast serve`` (see ``cli.py``).
"""
