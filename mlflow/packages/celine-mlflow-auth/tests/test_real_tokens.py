"""Integration: real tokens from a local Keycloak through the whole request path.

Each token is minted by the realm, its signature is verified against the realm's JWKS by
``extract_jwt_claims``, and ``authorize_request`` decides on it, provisioning into an
in-memory store. Nothing is mocked between the token and the decision.

Opt-in, local only (never a shared environment):

    CELINE_MLFLOW_AUTH_TEST_KEYCLOAK=http://keycloak.celine.localhost \
        uv run --no-sync pytest tests/test_real_tokens.py

Expects the local dev realm ``celine`` converged to the two-level model (celine-policies
``keycloak bootstrap`` + ``seed-dev-users``): ``admin`` holds the ``platform-admin`` realm
role, ``org-admin`` is in an organisation's ``admins`` group only, ``org-viewer`` in its
``viewers`` group. Passwords equal usernames; service-client secrets equal client ids.

``CELINE_MLFLOW_AUTH_TEST_LEGACY_TOKEN`` may carry an access token of a user still in the
retired realm group ``/admins`` (top-level ``groups`` claim); that case is skipped without it.
"""

import base64
import json
import os
from types import SimpleNamespace

import pytest
import requests
from flask import Flask
from werkzeug.datastructures import Authorization

KEYCLOAK = os.getenv("CELINE_MLFLOW_AUTH_TEST_KEYCLOAK", "").rstrip("/")
REALM = os.getenv("CELINE_MLFLOW_AUTH_TEST_REALM", "celine")
USER_CLIENT = os.getenv("CELINE_MLFLOW_AUTH_TEST_USER_CLIENT", "oauth2_proxy")
USER_CLIENT_SECRET = os.getenv("CELINE_MLFLOW_AUTH_TEST_USER_CLIENT_SECRET", USER_CLIENT)
LEGACY_TOKEN = os.getenv("CELINE_MLFLOW_AUTH_TEST_LEGACY_TOKEN", "")

pytestmark = pytest.mark.skipif(
    not KEYCLOAK, reason="set CELINE_MLFLOW_AUTH_TEST_KEYCLOAK to a local Keycloak"
)


class _Store:
    """The slice of MLflow's auth store that resolve_mlflow_user uses."""

    def __init__(self):
        self.users: dict[str, bool] = {}

    def has_user(self, username):
        return username in self.users

    def create_user(self, username, password, is_admin=False):
        self.users[username] = is_admin

    def get_user(self, username):
        return SimpleNamespace(is_admin=self.users[username])

    def update_user(self, username, is_admin=None):
        self.users[username] = is_admin


def _token_url() -> str:
    return f"{KEYCLOAK}/realms/{REALM}/protocol/openid-connect/token"


def _user_token(username: str) -> str:
    resp = requests.post(
        _token_url(),
        data={
            "grant_type": "password",
            "client_id": USER_CLIENT,
            "client_secret": USER_CLIENT_SECRET,
            "username": username,
            "password": username,
            "scope": "openid email profile organization:*",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _service_token(client_id: str) -> str:
    resp = requests.post(
        _token_url(),
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_id,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _unverified_claims(token: str) -> dict:
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


@pytest.fixture()
def store(monkeypatch):
    from celine.mlflow_auth import jwt, user

    monkeypatch.setattr(jwt, "JWKS_URL", f"{KEYCLOAK}/realms/{REALM}/protocol/openid-connect/certs")
    jwt.get_jwks_uri.cache_clear()
    jwt.get_public_key.cache_clear()
    user._user_cache.clear()
    s = _Store()
    monkeypatch.setattr("celine.mlflow_auth.auth._get_store", lambda: s)
    yield s
    user._user_cache.clear()


def _authorize(token: str):
    from celine.mlflow_auth.auth import authorize_request

    app = Flask(__name__)
    with app.test_request_context(headers={"X-Auth-Request-Access-Token": token}):
        return authorize_request()


def _assert_denied(result):
    assert not isinstance(result, Authorization)
    assert result.status_code == 401


class TestPeople:
    def test_platform_admin_is_admitted_as_admin(self, store):
        token = _user_token("admin")
        assert "platform-admin" in _unverified_claims(token)["realm_access"]["roles"]

        result = _authorize(token)

        assert isinstance(result, Authorization)
        assert result.username == "admin"
        assert store.users == {"admin": True}

    def test_organisation_admin_is_not_a_platform_admin(self, store):
        token = _user_token("org-admin")
        claims = _unverified_claims(token)
        assert "platform-admin" not in claims.get("realm_access", {}).get("roles", [])
        org_groups = [g for o in claims.get("organization", {}).values() for g in o["groups"]]
        assert "/admins" in org_groups, "fixture: org-admin must hold an organisation admins group"

        _assert_denied(_authorize(token))
        assert store.users == {}

    def test_organisation_viewer_is_denied(self, store):
        _assert_denied(_authorize(_user_token("org-viewer")))
        assert store.users == {}

    @pytest.mark.skipif(not LEGACY_TOKEN, reason="set CELINE_MLFLOW_AUTH_TEST_LEGACY_TOKEN")
    def test_legacy_realm_admin_group_grants_nothing(self, store):
        claims = _unverified_claims(LEGACY_TOKEN)
        assert "/admins" in claims.get("groups", []) or "admins" in claims.get("groups", [])

        _assert_denied(_authorize(LEGACY_TOKEN))
        assert store.users == {}

    def test_forged_token_is_rejected(self, store):
        header, payload, signature = _user_token("org-admin").split(".")
        forged = _unverified_claims(".".join([header, payload, signature]))
        forged["realm_access"] = {"roles": ["platform-admin"]}
        body = base64.urlsafe_b64encode(json.dumps(forged).encode()).rstrip(b"=").decode()

        _assert_denied(_authorize(".".join([header, body, signature])))
        assert store.users == {}


class TestServices:
    def test_service_with_mlflow_admin_scope(self, store):
        token = _service_token("svc-forecast")
        assert "mlflow.admin" in _unverified_claims(token)["scope"].split()

        result = _authorize(token)

        assert isinstance(result, Authorization)
        assert store.users == {result.username: True}

    def test_admin_cli_client(self, store):
        result = _authorize(_service_token("celine-cli"))

        assert isinstance(result, Authorization)
        assert result.username == "celine-cli"
        assert store.users == {"celine-cli": True}

    def test_service_without_mlflow_scope_is_denied(self, store):
        token = _service_token("svc-digital-twin")
        assert not {"mlflow.admin", "mlflow.read"} & set(
            _unverified_claims(token).get("scope", "").split()
        )

        _assert_denied(_authorize(token))
        assert store.users == {}
