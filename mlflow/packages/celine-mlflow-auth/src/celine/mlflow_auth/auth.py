from flask import Flask, Response, make_response, request
from werkzeug.datastructures import Authorization

from celine.mlflow_auth.jwt import extract_jwt_claims
from celine.mlflow_auth.user import resolve_mlflow_user


def _get_store():
    from mlflow.server.auth import store

    return store


def _make_401_response() -> Response:
    res = make_response("Authentication required. Provide a valid JWT token.", 401)
    return res


def authorize_request() -> Authorization | Response:
    """
    MLflow authorization_function entry point.

    Called by MLflow's authenticate_request() on every Flask request.
    Extracts and verifies the KC JWT from oauth2-proxy headers,
    auto-provisions the user, and returns Authorization or 401.
    """
    claims = extract_jwt_claims(request.headers)
    if not claims:
        return _make_401_response()

    store = _get_store()
    username = resolve_mlflow_user(store, claims)
    if not username:
        return _make_401_response()

    return Authorization("bearer", {"username": username})


def create_app(app: Flask | None = None):
    """
    MLflow app entry point registered as 'celine-auth'.

    Delegates to MLflow's built-in auth create_app which wires up
    the store, before_request hooks, and validators. Our custom
    authorization_function (set in the INI config) replaces the
    default basic-auth handler with JWT verification.

    Supports env var overrides before MLflow reads the config:
      CELINE_MLFLOW_AUTH_DATABASE_URI  — auth DB connection string
      CELINE_MLFLOW_AUTH_ADMIN_PASSWORD — bootstrap admin password
    """
    import os

    import mlflow.server.auth as _auth_module
    from mlflow.server.auth import create_app as _mlflow_create_app

    if app is None:
        from mlflow.server import app as _default_app

        app = _default_app

    overrides = {}
    if db_uri := os.getenv("CELINE_MLFLOW_AUTH_DATABASE_URI"):
        overrides["database_uri"] = db_uri
    if admin_pw := os.getenv("CELINE_MLFLOW_AUTH_ADMIN_PASSWORD"):
        overrides["admin_password"] = admin_pw
    if overrides:
        _auth_module.auth_config = _auth_module.auth_config._replace(**overrides)

    # Uvicorn spawns multiple workers that all call create_app concurrently.
    # The first worker runs migrations; others may hit IntegrityError on the
    # alembic_version_auth table. Retry once after a short delay to let the
    # winning worker finish.
    import time

    try:
        return _mlflow_create_app(app)
    except Exception:
        time.sleep(2)
        _auth_module._auth_initialized = False
        return _mlflow_create_app(app)
