import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    project_id: str
    region: str
    gcs_bucket: str
    bq_default_dataset: str
    credentials_path: str | None
    restricts_fields: list[str]
    numeric_restricts_fields: list[str]
    embedding_text_fields: list[str]
    embedding_metadata_fields: list[str]
    embedding_model_name: str
    embedding_output_dimensionality: int
    index_id: str
    endpoint_id: str
    deployed_index_id: str
    batch_root: str
    batch_delete_root: str
    index_defaults: dict[str, Any]


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "parameters" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    return AppConfig(
        project_id=raw.get("project_id", ""),
        region=raw.get("region", "us-central1"),
        gcs_bucket=raw.get("gcs_bucket", ""),
        bq_default_dataset=raw.get("bq_default_dataset", ""),
        credentials_path=credentials_path,
        restricts_fields=raw.get("filters", {}).get("restricts_fields", []),
        numeric_restricts_fields=raw.get("filters", {}).get("numeric_restricts_fields", []),
        embedding_text_fields=raw.get("embedding", {}).get("text_fields", []),
        embedding_metadata_fields=raw.get("embedding", {}).get("metadata_fields", []),
        embedding_model_name=raw.get("embedding", {}).get("model_name", "gemini-embedding-001"),
        embedding_output_dimensionality=int(raw.get("embedding", {}).get("output_dimensionality", 768)),
        index_id=raw.get("resource_names", {}).get("index_id", ""),
        endpoint_id=raw.get("resource_names", {}).get("endpoint_id", ""),
        deployed_index_id=raw.get("resource_names", {}).get("deployed_index_id", ""),
        batch_root=raw.get("batch_paths", {}).get("batch_root", ""),
        batch_delete_root=raw.get("batch_paths", {}).get("delete_root", ""),
        index_defaults=raw.get("index_defaults", {}) or {},
    )


def update_resource_names(
    index_id: str | None = None,
    endpoint_id: str | None = None,
    deployed_index_id: str | None = None,
    path: str | os.PathLike | None = None,
) -> None:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    resource_names = raw.get("resource_names", {}) or {}
    if index_id:
        resource_names["index_id"] = index_id
    if endpoint_id:
        resource_names["endpoint_id"] = endpoint_id
    if deployed_index_id:
        resource_names["deployed_index_id"] = deployed_index_id

    raw["resource_names"] = resource_names

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
