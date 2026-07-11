"""
Map Keycloak claims to MLflow is_admin flag.

Two auth paths based on token type:

**Plain users** (browser login via oauth2-proxy):
  claims.groups                     — realm-level groups (list of path strings)
  claims.organization.<slug>.groups — org-level groups per org (ignored here)

  Role mapping (realm-level groups only):
    admin | admins | realm_admin | realm_manager | manager | managers → is_admin=True
    Any other realm group (viewer, editor, member, etc.)              → is_admin=False
    No realm groups (even if org-level membership exists)             → denied (None)

  Users with only org-level membership are explicitly denied access.
  MLflow does not have per-org scoping — only realm-level users are allowed.

**Service accounts** (client_credentials grant):
  Detected by the presence of ``client_id`` in claims. Checked against
  the ``scope`` claim (space-separated string):
    mlflow.admin → is_admin=True
    mlflow.read  → is_admin=False
    No mlflow.* scope → denied (None)
"""

_ADMIN_GROUPS = frozenset({
    "admin", "admins", "realm_admin", "realm_manager", "manager", "managers",
})

_MLFLOW_ADMIN_SCOPES = frozenset({"mlflow.admin"})
_MLFLOW_ACCESS_SCOPES = frozenset({"mlflow.admin", "mlflow.read"})


def _group_name(group: str) -> str:
    """Return terminal path segment: '/admins' → 'admins', 'viewers' → 'viewers'."""
    return group.strip("/").rsplit("/", 1)[-1] if group else ""


def _is_service_account(claims: dict) -> bool:
    if "client_id" in claims:
        return True
    if claims.get("preferred_username") or claims.get("email") or claims.get("groups") is not None:
        return False
    return True


def _resolve_service_account(claims: dict) -> bool | None:
    raw_scope = claims.get("scope", "")
    scopes = set(raw_scope.split()) if raw_scope else set()

    if not scopes & _MLFLOW_ACCESS_SCOPES:
        return None

    return bool(scopes & _MLFLOW_ADMIN_SCOPES)


def _resolve_user_groups(claims: dict) -> bool | None:
    raw_groups = claims.get("groups") or []
    if isinstance(raw_groups, str):
        raw_groups = [raw_groups]

    names = [_group_name(g) for g in raw_groups]
    names = [n for n in names if n]

    if not names:
        return None

    return any(n in _ADMIN_GROUPS for n in names)


def resolve_is_admin(claims: dict) -> bool | None:
    """
    Parse KC JWT claims into an MLflow is_admin decision.

    Service accounts (client_credentials) are checked against scopes.
    Plain users are checked against realm-level groups.

    Returns:
        True  — user should be admin
        False — user should be regular user
        None  — no matching scopes/groups, deny access
    """
    if _is_service_account(claims):
        return _resolve_service_account(claims)
    return _resolve_user_groups(claims)
