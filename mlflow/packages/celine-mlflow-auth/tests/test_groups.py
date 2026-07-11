import pytest

from celine.mlflow_auth.groups import _group_name, _is_service_account, resolve_is_admin


class TestGroupName:
    def test_simple(self):
        assert _group_name("admins") == "admins"

    def test_path_prefix(self):
        assert _group_name("/admins") == "admins"

    def test_nested_path(self):
        assert _group_name("/realm/roles/admins") == "admins"

    def test_trailing_slash(self):
        assert _group_name("admins/") == "admins"

    def test_empty(self):
        assert _group_name("") == ""


class TestResolveIsAdmin:
    @pytest.mark.parametrize("group", [
        "admin", "admins", "realm_admin", "realm_manager", "manager", "managers",
    ])
    def test_admin_groups(self, group):
        assert resolve_is_admin({"groups": [group]}) is True

    @pytest.mark.parametrize("group", [
        "admin", "admins", "realm_admin", "realm_manager", "manager", "managers",
    ])
    def test_admin_groups_with_path_prefix(self, group):
        assert resolve_is_admin({"groups": [f"/some/path/{group}"]}) is True

    @pytest.mark.parametrize("group", [
        "viewer", "viewers", "editor", "editors", "member", "participant", "user",
    ])
    def test_non_admin_groups(self, group):
        assert resolve_is_admin({"groups": [group]}) is False

    def test_mixed_groups_admin_wins(self):
        assert resolve_is_admin({"groups": ["viewer", "admin"]}) is True

    def test_empty_groups_denied(self):
        assert resolve_is_admin({"groups": []}) is None

    def test_null_groups_denied(self):
        assert resolve_is_admin({}) is None

    def test_groups_key_none_denied(self):
        assert resolve_is_admin({"groups": None}) is None

    def test_string_group_coerced_to_list(self):
        assert resolve_is_admin({"groups": "admin"}) is True

    def test_string_non_admin_group(self):
        assert resolve_is_admin({"groups": "viewer"}) is False


class TestOrgOnlyDenied:
    """Users with only org-level membership (no realm groups) are denied."""

    def test_org_only_no_realm_groups(self):
        claims = {
            "organization": {"example_dso": {"groups": ["admins"]}},
        }
        assert resolve_is_admin(claims) is None

    def test_org_only_empty_realm_groups(self):
        claims = {
            "groups": [],
            "organization": {"example_rec": {"groups": ["viewers"]}},
        }
        assert resolve_is_admin(claims) is None

    def test_org_only_null_realm_groups(self):
        claims = {
            "groups": None,
            "organization": {"example_dso": {"groups": ["managers"]}},
        }
        assert resolve_is_admin(claims) is None

    def test_realm_plus_org_allowed(self):
        claims = {
            "groups": ["/viewers"],
            "organization": {"example_dso": {"groups": ["admins"]}},
        }
        assert resolve_is_admin(claims) is False

    def test_realm_admin_plus_org_allowed(self):
        claims = {
            "groups": ["/admins"],
            "organization": {"example_dso": {"groups": ["viewers"]}},
        }
        assert resolve_is_admin(claims) is True


class TestServiceAccountDetection:
    def test_client_id_present(self):
        assert _is_service_account({"client_id": "svc-forecast"}) is True

    def test_no_username_no_email(self):
        assert _is_service_account({"azp": "svc-forecast", "scope": "mlflow.admin"}) is True

    def test_user_with_username(self):
        assert _is_service_account({"preferred_username": "alice"}) is False

    def test_user_with_email(self):
        assert _is_service_account({"email": "alice@example.com"}) is False


class TestServiceAccountScopes:
    def _svc_claims(self, scope: str) -> dict:
        return {"azp": "svc-forecast", "scope": scope}

    def test_admin_scope(self):
        assert resolve_is_admin(self._svc_claims("mlflow.admin")) is True

    def test_read_scope(self):
        assert resolve_is_admin(self._svc_claims("mlflow.read")) is False

    def test_admin_and_read(self):
        assert resolve_is_admin(self._svc_claims("mlflow.admin mlflow.read")) is True

    def test_no_mlflow_scope_denied(self):
        assert resolve_is_admin(self._svc_claims("dataset.query")) is None

    def test_empty_scope_denied(self):
        assert resolve_is_admin(self._svc_claims("")) is None

    def test_no_scope_claim_denied(self):
        assert resolve_is_admin({"azp": "svc-forecast"}) is None

    def test_token_with_groups_treated_as_user(self):
        claims = {"azp": "svc-forecast", "scope": "mlflow.read", "groups": ["/admins"]}
        assert resolve_is_admin(claims) is True  # groups present → user path → admin group wins

    def test_client_id_claim_uses_scope_path(self):
        claims = {"client_id": "svc-forecast", "scope": "mlflow.admin"}
        assert resolve_is_admin(claims) is True
