"""Environment-driven settings with dev defaults.

All infrastructure configuration (database, MLflow, MinIO) lives here.
Values are read from environment variables, with a ``.env`` file loaded
automatically if present. Dev defaults match the local docker-compose setup.

Usage::

    from celine.forecasting.core.settings import settings

    engine = create_engine(settings.database_url)
"""

from __future__ import annotations

import glob
import logging
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql://postgres:securepassword123@172.17.0.1:15432/datasets"

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow.celine.localhost"

    # OIDC auth (service-to-service via Keycloak)
    oidc_issuer_url: str = "http://keycloak.celine.localhost/realms/celine"
    oidc_client_id: str = "svc-forecast"
    oidc_client_secret: str = "svc-forecast"

    # MinIO / S3 artifact store
    mlflow_s3_endpoint_url: str = "http://172.17.0.1:9000"
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"

    # Training
    training_n_jobs: int = 4

    # Governance
    governance_file: str = "governance.yaml"
    governance_search_paths: str = ""

    def governance_tags(self, consumed_tables: list[str] | None = None) -> dict[str, str]:
        """Build MLflow tags from governance metadata.

        Loads the local governance.yaml for project-level metadata, then
        optionally resolves consumed_tables against governance files found
        in governance_search_paths to attach input lineage.
        """
        tags: dict[str, str] = {}

        local = _find_file(self.governance_file)
        if local:
            gov = _load_yaml(local)
            defaults = gov.get("defaults", {})
            if defaults.get("source_url"):
                tags["mlflow.source.name"] = defaults["source_url"]
                tags["mlflow.source.type"] = "PROJECT"
            for key in ("license", "source_system", "classification"):
                if defaults.get(key):
                    tags[key] = defaults[key]
            for owner in defaults.get("ownership", []):
                tags[f"owner.{owner['type']}"] = owner["name"]

        if consumed_tables and self.governance_search_paths:
            registry = _build_governance_registry(self.governance_search_paths)
            for table in consumed_tables:
                table_key = f"datasets.{table}"
                entry = registry.get(table_key)
                if entry:
                    tags[f"input.{table}.license"] = entry.get("license", "unknown")
                    tags[f"input.{table}.classification"] = entry.get("classification", "green")
                    owners = entry.get("ownership", [])
                    if owners:
                        tags[f"input.{table}.owner"] = owners[0].get("name", "unknown")
                    tags[f"input.{table}.source_system"] = entry.get("source_system", "unknown")

        return tags


def _find_file(name: str) -> Path | None:
    """Walk up from cwd looking for a file by name."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / name
        if candidate.is_file():
            return candidate
    return None


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def _build_governance_registry(search_paths: str) -> dict[str, dict]:
    """Scan governance files and build a {dataset_key: entry} lookup."""
    registry: dict[str, dict] = {}
    for pattern in search_paths.split(","):
        pattern = pattern.strip()
        if not pattern:
            continue
        for path in glob.glob(pattern, recursive=True):
            gov = _load_yaml(Path(path))
            defaults = gov.get("defaults", {})
            for key, entry in gov.get("sources", {}).items():
                merged = {**defaults, **{k: v for k, v in entry.items() if v is not None}}
                registry[key] = merged
    return registry


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
