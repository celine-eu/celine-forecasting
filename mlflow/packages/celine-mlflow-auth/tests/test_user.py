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


class TestResolveNewUser:
    def test_creates_user(self):
        store = _make_store(has_user=False)
        claims = {"preferred_username": "alice", "groups": ["viewer"]}

        result = resolve_mlflow_user(store, claims)

        assert result == "alice"
        store.create_user.assert_called_once()
        args = store.create_user.call_args
        assert args[0][0] == "alice"
        assert len(args[0][1]) >= 12  # placeholder password
        assert args[1] == {"is_admin": False}

    def test_creates_admin_user(self):
        store = _make_store(has_user=False)
        claims = {"preferred_username": "bob", "groups": ["admins"]}

        result = resolve_mlflow_user(store, claims)

        assert result == "bob"
        assert store.create_user.call_args[1] == {"is_admin": True}


class TestResolveExistingUser:
    def test_no_change_needed(self):
        store = _make_store(has_user=True, user_is_admin=False)
        claims = {"preferred_username": "alice", "groups": ["viewer"]}

        result = resolve_mlflow_user(store, claims)

        assert result == "alice"
        store.create_user.assert_not_called()
        store.update_user.assert_not_called()

    def test_role_synced_on_promotion(self):
        store = _make_store(has_user=True, user_is_admin=False)
        claims = {"preferred_username": "alice", "groups": ["admins"]}

        result = resolve_mlflow_user(store, claims)

        assert result == "alice"
        store.update_user.assert_called_once_with("alice", is_admin=True)

    def test_role_synced_on_demotion(self):
        store = _make_store(has_user=True, user_is_admin=True)
        claims = {"preferred_username": "alice", "groups": ["viewer"]}

        result = resolve_mlflow_user(store, claims)

        assert result == "alice"
        store.update_user.assert_called_once_with("alice", is_admin=False)


class TestServiceAccount:
    def test_cli_admin_azp(self):
        store = _make_store(has_user=False)
        claims = {"azp": "celine-cli", "sub": "service-account-celine-cli"}

        with patch("celine.mlflow_auth.user._CLI_ADMIN_AZP", frozenset({"celine-cli"})):
            result = resolve_mlflow_user(store, claims)

        # azp comes before sub in username priority
        assert result == "celine-cli"
        assert store.create_user.call_args[1] == {"is_admin": True}


class TestDenied:
    def test_no_username(self):
        store = _make_store()
        assert resolve_mlflow_user(store, {}) is None
        store.create_user.assert_not_called()

    def test_no_groups(self):
        store = _make_store()
        claims = {"preferred_username": "alice", "groups": []}
        assert resolve_mlflow_user(store, claims) is None

    def test_null_groups(self):
        store = _make_store()
        claims = {"preferred_username": "alice"}
        assert resolve_mlflow_user(store, claims) is None


class TestCache:
    def test_cache_avoids_db_hit(self):
        store = _make_store(has_user=True, user_is_admin=False)
        claims = {"preferred_username": "alice", "groups": ["viewer"]}

        resolve_mlflow_user(store, claims)
        store.reset_mock()

        resolve_mlflow_user(store, claims)

        store.has_user.assert_not_called()
        store.get_user.assert_not_called()

    def test_cache_invalidated_on_role_change(self):
        store = _make_store(has_user=True, user_is_admin=False)
        claims_viewer = {"preferred_username": "alice", "groups": ["viewer"]}
        claims_admin = {"preferred_username": "alice", "groups": ["admins"]}

        resolve_mlflow_user(store, claims_viewer)
        store.reset_mock()

        store.has_user.return_value = True
        store.get_user.return_value = SimpleNamespace(is_admin=False)
        resolve_mlflow_user(store, claims_admin)

        store.has_user.assert_called()
        store.update_user.assert_called_once_with("alice", is_admin=True)


class TestRaceCondition:
    def test_concurrent_create_handled(self):
        store = _make_store(has_user=False)
        store.create_user.side_effect = Exception("RESOURCE_ALREADY_EXISTS")
        # After the race, has_user returns True on the second call
        store.has_user.side_effect = [False, True]
        store.get_user.return_value = SimpleNamespace(is_admin=False)

        claims = {"preferred_username": "alice", "groups": ["viewer"]}
        result = resolve_mlflow_user(store, claims)

        assert result == "alice"


class TestUsernamePriority:
    def test_preferred_username_first(self):
        store = _make_store(has_user=True, user_is_admin=False)
        claims = {
            "preferred_username": "alice",
            "email": "alice@example.com",
            "azp": "some-client",
            "sub": "user-uuid",
            "groups": ["viewer"],
        }
        assert resolve_mlflow_user(store, claims) == "alice"

    def test_email_fallback(self):
        store = _make_store(has_user=True, user_is_admin=False)
        claims = {"email": "alice@example.com", "sub": "user-uuid", "groups": ["viewer"]}
        assert resolve_mlflow_user(store, claims) == "alice@example.com"

    def test_azp_fallback(self):
        store = _make_store(has_user=True, user_is_admin=False)
        claims = {"azp": "my-client", "sub": "user-uuid", "groups": ["viewer"]}

        with patch("celine.mlflow_auth.user._CLI_ADMIN_AZP", frozenset()):
            assert resolve_mlflow_user(store, claims) == "my-client"

    def test_sub_fallback(self):
        store = _make_store(has_user=True, user_is_admin=False)
        claims = {"sub": "user-uuid", "groups": ["viewer"]}
        assert resolve_mlflow_user(store, claims) == "user-uuid"
