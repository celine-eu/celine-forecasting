import pytest

from celine.mlflow_auth.groups import _group_name, resolve_is_admin


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
