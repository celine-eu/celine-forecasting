import logging
import os
import secrets
import threading

from cachetools import TTLCache

from celine.mlflow_auth.groups import resolve_is_admin

logger = logging.getLogger(__name__)

_CLI_ADMIN_AZP: frozenset[str] = frozenset(
    c.strip()
    for c in os.getenv("CELINE_MLFLOW_AUTH_CLI_ADMIN_AZP", "celine-cli").split(",")
    if c.strip()
)

_SYNC_CACHE_TTL = int(os.getenv("CELINE_MLFLOW_AUTH_SYNC_CACHE_TTL", "60"))

# username → is_admin; TTL avoids DB lookups on every request
_user_cache: TTLCache = TTLCache(maxsize=10_000, ttl=_SYNC_CACHE_TTL)
_cache_lock = threading.Lock()


def _extract_username(claims: dict) -> str | None:
    return (
        claims.get("preferred_username")
        or claims.get("email")
        or claims.get("azp")
        or claims.get("sub")
    )


def resolve_mlflow_user(store, claims: dict) -> str | None:
    """
    Auto-provision or sync an MLflow user from verified JWT claims.

    Returns the username on success, or None to deny access.
    """
    username = _extract_username(claims)
    if not username:
        logger.warning("No username in JWT claims (sub=%s)", claims.get("sub"))
        return None

    azp = claims.get("azp")
    if azp in _CLI_ADMIN_AZP:
        is_admin = True
    else:
        result = resolve_is_admin(claims)
        if result is None:
            logger.warning("User %s has no KC groups — access denied", username)
            return None
        is_admin = result

    with _cache_lock:
        cached = _user_cache.get(username)
    if cached is not None and cached == is_admin:
        return username

    try:
        if not store.has_user(username):
            placeholder = secrets.token_urlsafe(48)
            try:
                store.create_user(username, placeholder, is_admin=is_admin)
                logger.info("Auto-provisioned MLflow user=%s is_admin=%s", username, is_admin)
            except Exception:
                # Race condition: another worker created the user between
                # has_user() and create_user(). Fall through to sync path.
                if not store.has_user(username):
                    raise
                logger.debug("User %s created by another worker, proceeding to sync", username)

        user = store.get_user(username)
        if user.is_admin != is_admin:
            store.update_user(username, is_admin=is_admin)
            logger.info("Synced MLflow user=%s is_admin=%s→%s", username, user.is_admin, is_admin)

    except Exception:
        logger.exception("Error resolving MLflow user=%s", username)
        return None

    with _cache_lock:
        _user_cache[username] = is_admin

    return username
