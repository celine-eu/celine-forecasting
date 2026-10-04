"""
Map verified Keycloak claims to an MLflow access decision.

MLflow is for platform administrators only. Two paths, chosen by celine-sdk's
``is_service_account``:

**People** (browser login through oauth2-proxy, or a user bearer token):
  ``realm_access.roles`` holds ``platform-admin``  → is_admin=True
  anything else                                     → denied (None)

  Nothing else in the token grants access:

  - an organisation's own groups (``organization.<alias>.groups``, even ``admins``)
    are valid only inside that organisation, and MLflow has no organisation scope;
  - a realm group in the top-level ``groups`` claim (``/admins``, ``admins``) is not
    a platform mechanism and grants nothing, if one still reaches a token;
  - realm roles other than ``platform-admin`` grant nothing.

**Service accounts** (client_credentials grant): checked against the ``scope`` claim
(space-separated string):
    mlflow.admin → is_admin=True
    mlflow.read  → is_admin=False
    no mlflow.* scope → denied (None)
"""

from celine.sdk.auth import is_platform_admin, is_service_account

_MLFLOW_ADMIN_SCOPES = frozenset({"mlflow.admin"})
_MLFLOW_ACCESS_SCOPES = frozenset({"mlflow.admin", "mlflow.read"})


def _resolve_service_account(claims: dict) -> bool | None:
    raw_scope = claims.get("scope", "")
    scopes = set(raw_scope.split()) if isinstance(raw_scope, str) else set()

    if not scopes & _MLFLOW_ACCESS_SCOPES:
        return None

    return bool(scopes & _MLFLOW_ADMIN_SCOPES)


def _resolve_user(claims: dict) -> bool | None:
    return True if is_platform_admin(claims) else None


def resolve_is_admin(claims: dict) -> bool | None:
    """
    Parse verified Keycloak JWT claims into an MLflow is_admin decision.

    Service accounts (client_credentials) are checked against their scopes.
    People are checked for the ``platform-admin`` realm role, and nothing else.

    Returns:
        True  — admin (a platform administrator, or a service holding mlflow.admin)
        False — regular user (a service holding mlflow.read only)
        None  — deny access
    """
    if is_service_account(claims):
        return _resolve_service_account(claims)
    return _resolve_user(claims)
