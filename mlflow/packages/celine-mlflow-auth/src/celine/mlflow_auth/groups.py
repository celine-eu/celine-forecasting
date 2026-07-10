"""
Map Keycloak claims to MLflow is_admin flag.

KC claim structure:
  claims.groups                     — realm-level groups (list of path strings)
  claims.organization.<slug>.groups — org-level groups per org (ignored here)

Role mapping (realm-level groups only):
  admin | admins | realm_admin | realm_manager | manager | managers → is_admin=True
  Any other realm group (viewer, editor, member, etc.)              → is_admin=False
  No realm groups (even if org-level membership exists)             → denied (None)

Users with only org-level membership are explicitly denied access.
MLflow does not have per-org scoping — only realm-level users are allowed.
"""

_ADMIN_GROUPS = frozenset({
    "admin", "admins", "realm_admin", "realm_manager", "manager", "managers",
})


def _group_name(group: str) -> str:
    """Return terminal path segment: '/admins' → 'admins', 'viewers' → 'viewers'."""
    return group.strip("/").rsplit("/", 1)[-1] if group else ""


def resolve_is_admin(claims: dict) -> bool | None:
    """
    Parse KC JWT claims into an MLflow is_admin decision.

    Only realm-level groups (claims.groups) grant access.
    Org-level membership (claims.organization) is ignored — users with
    only org groups are denied.

    Returns:
        True  — user should be admin
        False — user should be regular user
        None  — no realm-level groups, deny access
    """
    raw_groups = claims.get("groups") or []
    if isinstance(raw_groups, str):
        raw_groups = [raw_groups]

    names = [_group_name(g) for g in raw_groups]
    names = [n for n in names if n]

    if not names:
        return None

    return any(n in _ADMIN_GROUPS for n in names)
