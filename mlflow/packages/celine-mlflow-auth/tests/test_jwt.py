import json
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from werkzeug.datastructures import Headers


@pytest.fixture(autouse=True)
def _clear_lru_caches():
    """Clear module-level lru_cache between tests."""
    from celine.mlflow_auth.jwt import _audiences, get_jwks_uri, get_public_key

    _audiences.cache_clear()
    get_jwks_uri.cache_clear()
    get_public_key.cache_clear()
    yield
    _audiences.cache_clear()
    get_jwks_uri.cache_clear()
    get_public_key.cache_clear()


@pytest.fixture()
def rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture()
def jwks_json(rsa_keypair):
    _, pub = rsa_keypair
    jwk = json.loads(RSAAlgorithm.to_jwk(pub))
    jwk["kid"] = "test-kid-1"
    jwk["use"] = "sig"
    return {"keys": [jwk]}


@pytest.fixture()
def signed_token(rsa_keypair):
    priv, _ = rsa_keypair

    def _make(claims=None, headers=None):
        c = {"sub": "user1", "iss": "https://kc.example.com/realms/test", **(claims or {})}
        h = {"kid": "test-kid-1", **(headers or {})}
        return pyjwt.encode(c, priv, algorithm="RS256", headers=h)

    return _make


class TestExtractJwtClaims:
    @patch("celine.mlflow_auth.jwt.requests.get")
    def test_extract_from_x_auth_header(self, mock_get, signed_token, jwks_json):
        mock_resp = MagicMock()
        mock_resp.json.return_value = jwks_json
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from celine.mlflow_auth.jwt import extract_jwt_claims

        token = signed_token()
        headers = Headers([("X-Auth-Request-Access-Token", token)])

        with patch("celine.mlflow_auth.jwt.JWKS_URL", "https://kc.example.com/jwks"):
            result = extract_jwt_claims(headers)

        assert result is not None
        assert result["sub"] == "user1"

    @patch("celine.mlflow_auth.jwt.requests.get")
    def test_extract_from_x_forwarded(self, mock_get, signed_token, jwks_json):
        mock_resp = MagicMock()
        mock_resp.json.return_value = jwks_json
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from celine.mlflow_auth.jwt import extract_jwt_claims

        token = signed_token()
        headers = Headers([("X-Forwarded-Access-Token", token)])

        with patch("celine.mlflow_auth.jwt.JWKS_URL", "https://kc.example.com/jwks"):
            result = extract_jwt_claims(headers)

        assert result is not None
        assert result["sub"] == "user1"

    @patch("celine.mlflow_auth.jwt.requests.get")
    def test_extract_from_bearer(self, mock_get, signed_token, jwks_json):
        mock_resp = MagicMock()
        mock_resp.json.return_value = jwks_json
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from celine.mlflow_auth.jwt import extract_jwt_claims

        token = signed_token()
        headers = Headers([("Authorization", f"Bearer {token}")])

        with patch("celine.mlflow_auth.jwt.JWKS_URL", "https://kc.example.com/jwks"):
            result = extract_jwt_claims(headers)

        assert result is not None
        assert result["sub"] == "user1"

    def test_extract_no_token(self):
        from celine.mlflow_auth.jwt import extract_jwt_claims

        headers = Headers()
        assert extract_jwt_claims(headers) is None

    @patch("celine.mlflow_auth.jwt.requests.get")
    def test_extract_expired(self, mock_get, rsa_keypair, jwks_json):
        import time

        priv, _ = rsa_keypair
        mock_resp = MagicMock()
        mock_resp.json.return_value = jwks_json
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from celine.mlflow_auth.jwt import extract_jwt_claims

        token = pyjwt.encode(
            {"sub": "user1", "iss": "https://kc.example.com", "exp": int(time.time()) - 3600},
            priv,
            algorithm="RS256",
            headers={"kid": "test-kid-1"},
        )
        headers = Headers([("X-Auth-Request-Access-Token", token)])

        with patch("celine.mlflow_auth.jwt.JWKS_URL", "https://kc.example.com/jwks"):
            assert extract_jwt_claims(headers) is None

    @patch("celine.mlflow_auth.jwt.requests.get")
    def test_extract_invalid_token(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"keys": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from celine.mlflow_auth.jwt import extract_jwt_claims

        headers = Headers([("X-Auth-Request-Access-Token", "not-a-jwt")])

        with patch("celine.mlflow_auth.jwt.JWKS_URL", "https://kc.example.com/jwks"):
            assert extract_jwt_claims(headers) is None

    @patch("celine.mlflow_auth.jwt.requests.get")
    def test_jwks_uri_from_env(self, mock_get, signed_token, jwks_json):
        mock_resp = MagicMock()
        mock_resp.json.return_value = jwks_json
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from celine.mlflow_auth.jwt import get_jwks_uri

        token = signed_token()
        with patch("celine.mlflow_auth.jwt.JWKS_URL", "https://custom.example.com/jwks"):
            uri = get_jwks_uri(token)

        assert uri == "https://custom.example.com/jwks"
        mock_get.assert_not_called()

    @patch("celine.mlflow_auth.jwt.requests.get")
    def test_jwks_uri_from_issuer(self, mock_get, signed_token):
        oidc_resp = MagicMock()
        oidc_resp.json.return_value = {"jwks_uri": "https://kc.example.com/jwks-discovered"}
        oidc_resp.raise_for_status = MagicMock()
        mock_get.return_value = oidc_resp

        from celine.mlflow_auth.jwt import get_jwks_uri

        token = signed_token(claims={"iss": "https://kc.example.com/realms/test"})
        with patch("celine.mlflow_auth.jwt.JWKS_URL", None):
            uri = get_jwks_uri(token)

        assert uri == "https://kc.example.com/jwks-discovered"
        mock_get.assert_called_once_with(
            "https://kc.example.com/realms/test/.well-known/openid-configuration",
            verify=True,
        )


class TestHeaderPriority:
    @patch("celine.mlflow_auth.jwt.requests.get")
    def test_x_auth_takes_priority_over_bearer(self, mock_get, rsa_keypair, jwks_json):
        priv, _ = rsa_keypair
        mock_resp = MagicMock()
        mock_resp.json.return_value = jwks_json
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from celine.mlflow_auth.jwt import extract_jwt_claims

        token_a = pyjwt.encode(
            {"sub": "from-x-auth", "iss": "https://kc.example.com"},
            priv, algorithm="RS256", headers={"kid": "test-kid-1"},
        )
        token_b = pyjwt.encode(
            {"sub": "from-bearer", "iss": "https://kc.example.com"},
            priv, algorithm="RS256", headers={"kid": "test-kid-1"},
        )
        headers = Headers([
            ("X-Auth-Request-Access-Token", token_a),
            ("Authorization", f"Bearer {token_b}"),
        ])

        with patch("celine.mlflow_auth.jwt.JWKS_URL", "https://kc.example.com/jwks"):
            result = extract_jwt_claims(headers)

        assert result["sub"] == "from-x-auth"
