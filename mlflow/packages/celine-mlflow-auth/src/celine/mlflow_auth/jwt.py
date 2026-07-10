import json
import logging
import os
from functools import lru_cache

import jwt
import requests
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from jwt.algorithms import RSAAlgorithm
from werkzeug.datastructures import Headers

logger = logging.getLogger(__name__)

JWKS_URL = os.getenv(
    "CELINE_MLFLOW_AUTH_JWKS_URL",
    "http://keycloak.celine.localhost/realms/celine/protocol/openid-connect/certs",
)
JWT_AUDIENCE = os.getenv("CELINE_MLFLOW_AUTH_JWT_AUDIENCE", "")
VERIFY_SSL = os.getenv("CELINE_MLFLOW_AUTH_SKIP_SSL_VERIFY", "false").lower() != "true"


@lru_cache(maxsize=1)
def _audiences() -> list[str]:
    return [a.strip() for a in JWT_AUDIENCE.split(",") if a.strip()]


@lru_cache(maxsize=1)
def get_jwks_uri(token: str) -> str:
    if JWKS_URL:
        return JWKS_URL
    claims = jwt.decode(token, options={"verify_signature": False})
    issuer = claims["iss"]
    resp = requests.get(issuer.rstrip("/") + "/.well-known/openid-configuration", verify=VERIFY_SSL)
    resp.raise_for_status()
    return resp.json()["jwks_uri"]


@lru_cache(maxsize=1)
def get_public_key(jwks_url: str, token: str) -> RSAPublicKey:
    resp = requests.get(jwks_url, verify=VERIFY_SSL)
    resp.raise_for_status()
    jwks = resp.json()

    kid = jwt.get_unverified_header(token).get("kid")
    if not kid:
        raise ValueError("JWT header missing kid")

    for key_data in jwks.get("keys", []):
        if key_data.get("kid") == kid:
            public_key = RSAAlgorithm.from_jwk(json.dumps(key_data))
            if not isinstance(public_key, RSAPublicKey):
                raise TypeError("Expected RSAPublicKey from JWK")
            return public_key

    raise ValueError(f"No matching JWK for kid={kid}")


def extract_jwt_claims(headers: Headers) -> dict | None:
    """Extract and verify KC JWT from incoming request headers."""
    auth = headers.get("Authorization", "")
    token = (
        headers.get("X-Auth-Request-Access-Token")
        or headers.get("X-Forwarded-Access-Token")
        or (auth.split(" ", 1)[1].strip() if auth.lower().startswith("bearer ") else None)
    )

    if not token:
        return None

    audiences = _audiences()
    decode_kwargs: dict = {"algorithms": ["RS256"]}
    if audiences:
        decode_kwargs["audience"] = audiences
    else:
        decode_kwargs["options"] = {"verify_aud": False}

    try:
        jwks_url = get_jwks_uri(token)
        public_key = get_public_key(jwks_url, token)
        return jwt.decode(token, public_key, **decode_kwargs)

    except jwt.ExpiredSignatureError:
        logger.info("JWT expired — user must re-authenticate")
        return None

    except jwt.InvalidTokenError as e:
        logger.warning("Invalid JWT: %s", e)
        return None
