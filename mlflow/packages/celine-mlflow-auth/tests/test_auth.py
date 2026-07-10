from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from werkzeug.datastructures import Authorization


@pytest.fixture()
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    return app


class TestAuthorizeRequest:
    def test_valid_jwt_returns_authorization(self, app):
        with app.test_request_context(headers={"X-Auth-Request-Access-Token": "valid-token"}):
            with (
                patch(
                    "celine.mlflow_auth.auth.extract_jwt_claims",
                    return_value={"sub": "alice", "preferred_username": "alice", "groups": ["viewer"]},
                ),
                patch(
                    "celine.mlflow_auth.auth.resolve_mlflow_user",
                    return_value="alice",
                ),
                patch("celine.mlflow_auth.auth._get_store", return_value=MagicMock()),
            ):
                from celine.mlflow_auth.auth import authorize_request

                result = authorize_request()

            assert isinstance(result, Authorization)
            assert result.username == "alice"

    def test_no_jwt_returns_401(self, app):
        with app.test_request_context():
            with patch("celine.mlflow_auth.auth.extract_jwt_claims", return_value=None):
                from celine.mlflow_auth.auth import authorize_request

                result = authorize_request()

            assert not isinstance(result, Authorization)
            assert result.status_code == 401

    def test_invalid_jwt_returns_401(self, app):
        with app.test_request_context(headers={"X-Auth-Request-Access-Token": "bad-token"}):
            with patch("celine.mlflow_auth.auth.extract_jwt_claims", return_value=None):
                from celine.mlflow_auth.auth import authorize_request

                result = authorize_request()

            assert result.status_code == 401

    def test_denied_user_returns_401(self, app):
        with app.test_request_context(headers={"X-Auth-Request-Access-Token": "valid-token"}):
            with (
                patch(
                    "celine.mlflow_auth.auth.extract_jwt_claims",
                    return_value={"sub": "alice", "groups": []},
                ),
                patch(
                    "celine.mlflow_auth.auth.resolve_mlflow_user",
                    return_value=None,
                ),
                patch("celine.mlflow_auth.auth._get_store", return_value=MagicMock()),
            ):
                from celine.mlflow_auth.auth import authorize_request

                result = authorize_request()

            assert result.status_code == 401

    def test_authorization_type_is_bearer(self, app):
        with app.test_request_context(headers={"X-Auth-Request-Access-Token": "valid-token"}):
            with (
                patch(
                    "celine.mlflow_auth.auth.extract_jwt_claims",
                    return_value={"sub": "bob", "preferred_username": "bob", "groups": ["admin"]},
                ),
                patch(
                    "celine.mlflow_auth.auth.resolve_mlflow_user",
                    return_value="bob",
                ),
                patch("celine.mlflow_auth.auth._get_store", return_value=MagicMock()),
            ):
                from celine.mlflow_auth.auth import authorize_request

                result = authorize_request()

            assert result.type == "bearer"
