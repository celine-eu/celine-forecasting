"""MLflow access: platform-admin realm role for people, mlflow.* scopes for services.

Claim shapes are the ones the local Keycloak issues through ``oauth2_proxy`` (organisation
groups keep their leading slash) plus the old shape of a realm ``/admins`` member, which
must now grant nothing.
"""

import pytest

from celine.mlflow_auth.groups import resolve_is_admin

PLATFORM_ADMIN = {
    "azp": "oauth2_proxy",
    "preferred_username": "admin",
    "email": "admin@example.org",
    "realm_access": {"roles": ["platform-admin"]},
    "organization": {"example-rec": {"type": ["rec"], "groups": ["/admins"]}},
}
ORG_ADMIN = {
    "azp": "oauth2_proxy",
    "preferred_username": "org-admin",
    "realm_access": {"roles": ["default-roles-celine", "offline_access", "uma_authorization"]},
    "organization": {"example-rec": {"type": ["rec"], "groups": ["/admins"]}},
}
LEGACY_REALM_ADMIN = {
    "azp": "oauth2_proxy",
    "preferred_username": "legacy-admin",
    "email": "legacy-admin@example.org",
    "groups": ["/admins", "admins"],
    "realm_access": {"roles": ["admin"]},
    "organization": {"example-rec": {"type": ["rec"], "groups": ["/admins"]}},
}


class TestPeople:
    def test_platform_admin_is_admin(self):
        assert resolve_is_admin(PLATFORM_ADMIN) is True

    def test_platform_admin_without_any_organisation(self):
        claims = {"preferred_username": "ops", "realm_access": {"roles": ["platform-admin"]}}
        assert resolve_is_admin(claims) is True

    def test_platform_admin_among_other_roles(self):
        claims = {
            "preferred_username": "ops",
            "realm_access": {"roles": ["default-roles-celine", "platform-admin"]},
        }
        assert resolve_is_admin(claims) is True

    def test_organisation_admin_is_denied(self):
        assert resolve_is_admin(ORG_ADMIN) is None

    @pytest.mark.parametrize("group", ["/admins", "/managers", "/editors", "/viewers"])
    def test_any_organisation_group_is_denied(self, group):
        claims = {
            "preferred_username": "alice",
            "organization": {"example-dso": {"groups": [group]}},
        }
        assert resolve_is_admin(claims) is None

    def test_organisation_group_named_like_the_role_is_denied(self):
        claims = {
            "preferred_username": "alice",
            "organization": {"example-rec": {"groups": ["/platform-admin"]}},
        }
        assert resolve_is_admin(claims) is None

    def test_legacy_realm_admin_group_grants_nothing(self):
        assert resolve_is_admin(LEGACY_REALM_ADMIN) is None

    @pytest.mark.parametrize(
        "groups",
        [
            ["/admins"], ["admins"], ["admin"], ["/managers"], ["managers"], ["/viewers"],
            ["realm_admin"], ["realm_manager"], ["platform-admin"], ["/platform-admin"],
        ],
    )
    def test_realm_group_claim_grants_nothing(self, groups):
        assert resolve_is_admin({"preferred_username": "alice", "groups": groups}) is None

    @pytest.mark.parametrize("role", ["admin", "manager", "editor", "viewer", "admins"])
    def test_other_realm_roles_grant_nothing(self, role):
        claims = {"preferred_username": "alice", "realm_access": {"roles": [role]}}
        assert resolve_is_admin(claims) is None

    def test_role_outside_realm_access_grants_nothing(self):
        claims = {
            "preferred_username": "alice",
            "roles": ["platform-admin"],
            "resource_access": {"oauth2_proxy": {"roles": ["platform-admin"]}},
        }
        assert resolve_is_admin(claims) is None

    @pytest.mark.parametrize(
        "realm_access",
        [None, {}, {"roles": None}, {"roles": "platform-admin"}, "platform-admin"],
    )
    def test_malformed_realm_access_denied(self, realm_access):
        claims = {"preferred_username": "alice", "realm_access": realm_access}
        assert resolve_is_admin(claims) is None

    def test_no_claims_beyond_identity_denied(self):
        assert resolve_is_admin({"preferred_username": "alice"}) is None

    def test_user_with_mlflow_scope_is_still_judged_on_the_role(self):
        claims = {"preferred_username": "alice", "scope": "openid mlflow.admin"}
        assert resolve_is_admin(claims) is None


class TestServiceAccountScopes:
    # Keycloak 26 encodes the grant in `jti`; `trrtcc:` is client credentials.
    def _svc_claims(self, scope: str) -> dict:
        return {"azp": "svc-forecast", "jti": "trrtcc:0000", "scope": scope}

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
        assert resolve_is_admin({"azp": "svc-forecast", "jti": "trrtcc:0000"}) is None

    def test_non_string_scope_denied(self):
        assert resolve_is_admin({**self._svc_claims(""), "scope": ["mlflow.admin"]}) is None

    def test_client_id_claim_uses_scope_path(self):
        assert resolve_is_admin({"client_id": "svc-forecast", "scope": "mlflow.admin"}) is True

    def test_service_account_username_uses_scope_path(self):
        claims = {"preferred_username": "service-account-svc-forecast", "scope": "mlflow.read"}
        assert resolve_is_admin(claims) is False

    def test_platform_admin_role_on_a_service_does_not_replace_scopes(self):
        claims = {
            **self._svc_claims("dataset.query"),
            "realm_access": {"roles": ["platform-admin"]},
        }
        assert resolve_is_admin(claims) is None

    def test_token_with_realm_group_is_a_person_and_the_group_grants_nothing(self):
        claims = {"azp": "svc-forecast", "scope": "mlflow.read", "groups": ["/admins"]}
        assert resolve_is_admin(claims) is None
