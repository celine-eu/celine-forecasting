from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from celine.mlflow_auth.user import _user_cache, resolve_mlflow_user


@pytest.fixture(autouse=True)
def _clear_cache():
    _user_cache.clear()
    yield
    _user_cache.clear()


def _make_store(*, has_user=False, user_is_admin=False):
    store = MagicMock()
    store.has_user.return_value = has_user
    store.get_user.return_value = SimpleNamespace(is_admin=user_is_admin)
    store.create_user.return_value = SimpleNamespace(username="test", is_admin=False)
    return store


def _platform_admin(username: str = "alice", **extra) -> dict:
    return {"preferred_username": username, "realm_access": {"roles": ["platform-admin"]}, **extra}


def _org_admin(username: str = "alice", **extra) -> dict:
    return {
        "preferred_username": username,
        "realm_access": {"roles": ["default-roles-celine"]},
        "organization": {"example-rec": {"groups": ["/admins"]}},
        **extra,
    }


# Keycloak 26 client-credentials token: no preferred_username, grant in `jti`.
def _service(azp: str, scope: str) -> dict:
    return {"azp": azp, "sub": "00000000-svc", "jti": "trrtcc:0000", "scope": scope}


class TestResolveNewUser:
    def test_creates_platform_admin(self):
        store = _make_store(has_user=False)

        result = resolve_mlflow_user(store, _platform_admin("bob"))

        assert result == "bob"
        store.create_user.assert_called_once()
        args = store.create_user.call_args
        assert args[0][0] == "bob"
        assert len(args[0][1]) >= 12  # placeholder password
        assert args[1] == {"is_admin": True}

    def test_creates_read_only_service(self):
        store = _make_store(has_user=False)

        result = resolve_mlflow_user(store, _service("svc-reader", "mlflow.read"))

        assert result == "svc-reader"
        assert store.create_user.call_args[1] == {"is_admin": False}


class TestResolveExistingUser:
    def test_no_change_needed(self):
        store = _make_store(has_user=True, user_is_admin=True)

        result = resolve_mlflow_user(store, _platform_admin())

        assert result == "alice"
        store.create_user.assert_not_called()
        store.update_user.assert_not_called()

    def test_role_synced_on_promotion(self):
        store = _make_store(has_user=True, user_is_admin=False)

        result = resolve_mlflow_user(store, _platform_admin())

        assert result == "alice"
        store.update_user.assert_called_once_with("alice", is_admin=True)

    def test_service_synced_on_demotion(self):
        store = _make_store(has_user=True, user_is_admin=True)

        result = resolve_mlflow_user(store, _service("svc-reader", "mlflow.read"))

        assert result == "svc-reader"
        store.update_user.assert_called_once_with("svc-reader", is_admin=False)

    def test_former_admin_without_the_role_is_denied_not_demoted(self):
        store = _make_store(has_user=True, user_is_admin=True)

        assert resolve_mlflow_user(store, _org_admin()) is None
        store.update_user.assert_not_called()


class TestServiceAccount:
    def test_cli_admin_azp(self):
        store = _make_store(has_user=False)
        claims = _service("celine-cli", "openid")

        with patch("celine.mlflow_auth.user._CLI_ADMIN_AZP", frozenset({"celine-cli"})):
            result = resolve_mlflow_user(store, claims)

        # azp comes before sub in username priority
        assert result == "celine-cli"
        assert store.create_user.call_args[1] == {"is_admin": True}

    def test_person_through_cli_client_is_not_trusted_by_azp(self):
        store = _make_store(has_user=False)
        claims = _org_admin(azp="celine-cli")

        with patch("celine.mlflow_auth.user._CLI_ADMIN_AZP", frozenset({"celine-cli"})):
            assert resolve_mlflow_user(store, claims) is None
        store.create_user.assert_not_called()

    def test_platform_admin_through_cli_client_is_admin_by_role(self):
        store = _make_store(has_user=False)
        claims = _platform_admin(azp="celine-cli")

        with patch("celine.mlflow_auth.user._CLI_ADMIN_AZP", frozenset({"celine-cli"})):
            assert resolve_mlflow_user(store, claims) == "alice"
        assert store.create_user.call_args[1] == {"is_admin": True}

    def test_service_without_mlflow_scope_denied(self):
        store = _make_store()
        assert resolve_mlflow_user(store, _service("svc-other", "dataset.query")) is None
        store.create_user.assert_not_called()


class TestDenied:
    def test_no_username(self):
        store = _make_store()
        assert resolve_mlflow_user(store, {}) is None
        store.create_user.assert_not_called()

    def test_organisation_admin(self):
        store = _make_store()
        assert resolve_mlflow_user(store, _org_admin()) is None
        store.create_user.assert_not_called()

    def test_legacy_realm_admin_group(self):
        store = _make_store()
        claims = {
            "preferred_username": "alice",
            "groups": ["/admins", "admins"],
            "realm_access": {"roles": ["admin"]},
        }
        assert resolve_mlflow_user(store, claims) is None
        store.create_user.assert_not_called()

    def test_identity_only(self):
        store = _make_store()
        assert resolve_mlflow_user(store, {"preferred_username": "alice"}) is None


class TestCache:
    def test_cache_avoids_db_hit(self):
        store = _make_store(has_user=True, user_is_admin=True)

        resolve_mlflow_user(store, _platform_admin())
        store.reset_mock()

        resolve_mlflow_user(store, _platform_admin())

        store.has_user.assert_not_called()
        store.get_user.assert_not_called()

    def test_cache_invalidated_on_role_change(self):
        store = _make_store(has_user=True, user_is_admin=False)
        reader = _service("svc-x", "mlflow.read")
        admin = _service("svc-x", "mlflow.admin")

        resolve_mlflow_user(store, reader)
        store.reset_mock()

        store.has_user.return_value = True
        store.get_user.return_value = SimpleNamespace(is_admin=False)
        resolve_mlflow_user(store, admin)

        store.has_user.assert_called()
        store.update_user.assert_called_once_with("svc-x", is_admin=True)

    def test_cached_admin_losing_the_role_is_denied(self):
        store = _make_store(has_user=True, user_is_admin=True)

        assert resolve_mlflow_user(store, _platform_admin()) == "alice"
        assert resolve_mlflow_user(store, _org_admin()) is None


class TestRaceCondition:
    def test_concurrent_create_handled(self):
        store = _make_store(has_user=False)
        store.create_user.side_effect = Exception("RESOURCE_ALREADY_EXISTS")
        # After the race, has_user returns True on the second call
        store.has_user.side_effect = [False, True]
        store.get_user.return_value = SimpleNamespace(is_admin=True)

        result = resolve_mlflow_user(store, _platform_admin())

        assert result == "alice"


class TestUsernamePriority:
    def test_preferred_username_first(self):
        store = _make_store(has_user=True, user_is_admin=True)
        claims = _platform_admin(email="alice@example.com", azp="some-client", sub="user-uuid")
        assert resolve_mlflow_user(store, claims) == "alice"

    def test_email_fallback(self):
        store = _make_store(has_user=True, user_is_admin=True)
        claims = {
            "email": "alice@example.com",
            "sub": "user-uuid",
            "realm_access": {"roles": ["platform-admin"]},
        }
        assert resolve_mlflow_user(store, claims) == "alice@example.com"

    def test_azp_fallback(self):
        store = _make_store(has_user=True, user_is_admin=False)

        with patch("celine.mlflow_auth.user._CLI_ADMIN_AZP", frozenset()):
            assert resolve_mlflow_user(store, _service("my-client", "mlflow.read")) == "my-client"

    def test_sub_fallback(self):
        store = _make_store(has_user=True, user_is_admin=False)
        claims = {"sub": "user-uuid", "client_id": "x", "scope": "mlflow.read"}
        assert resolve_mlflow_user(store, claims) == "user-uuid"
