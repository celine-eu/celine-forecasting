# ADR-0001 — MLflow admits only platform administrators and scoped services

**Date:** 2026-10-03
**Status:** accepted

## Context

The MLflow auth plugin (`mlflow/packages/celine-mlflow-auth`) granted on Keycloak **realm
groups**: realm `admin(s)`/`manager(s)` became MLflow admins, any other realm group made a
regular user, and a person with no realm group was denied.

The platform now runs many organisations in one realm, and each organisation has groups with
the same names as the realm groups (`admins`, `managers`, `editors`, `viewers`). A reader of
the top-level `groups` claim, or of a merge of both levels, could read one organisation's
`admins` as a platform admin. The platform decided (`celine-policies`) on exactly two levels:
the realm **role** `platform-admin` is the only platform-wide grant, and an organisation's
groups are valid only inside that organisation. Realm groups and the realm roles
`admin`/`manager`/`editor`/`viewer` are removed. MLflow holds every organisation's models and
runs and has no organisation scope, so an organisation's groups cannot apply to it.

## Decision

- A person is admitted only with the realm role `platform-admin`
  (`celine.sdk.auth.is_platform_admin`) and becomes an MLflow admin. Every other person is
  denied.
- A service account is admitted by scope: `mlflow.admin` as admin, `mlflow.read` as a
  regular user, no `mlflow.*` scope denied. The `CELINE_MLFLOW_AUTH_CLI_ADMIN_AZP` clients
  (default `celine-cli`) stay admins, for service-account tokens only.
- Person versus service is decided by `celine.sdk.auth.is_service_account`, never by
  whether the token carries a group.
- The top-level `groups` claim is not read. A realm group still present in a token grants
  nothing; neither does any organisation group or any other realm role.
- There is no compatibility path for tokens minted before the change.

## Consequences

- Organisation members, including an organisation's `admins`, cannot use MLflow at all,
  even read-only. Opening it to them needs an organisation scope on experiments first, and
  a new ADR that supersedes this one.
- The plugin now depends on `celine-sdk >= 2.0.0`, and the MLflow image must ship with the
  realm change: the previous plugin denies the platform admin, who no longer carries a realm
  group, and admits any token still carrying `/admins`.
- Re-reading `groups` "for convenience" reopens the ambiguity this record closes.
