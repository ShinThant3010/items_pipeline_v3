import os
from pathlib import Path

import yaml

from functions.utils.config import load_config


def test_load_config_reads_defaults_and_env(tmp_path, monkeypatch):
    data = {
        "project_id": "proj-1",
        "region": "us-central1",
        "gcs_bucket": "gs://bucket/",
        "bq_default_dataset": "dataset",
        "filters": {
            "restricts_fields": ["level"],
            "numeric_restricts_fields": ["created_at"],
        },
        "embedding": {
            "model_name": "model-1",
            "output_dimensionality": 512,
            "text_fields": ["title"],
            "metadata_fields": ["id", "title"],
        },
        "resource_names": {
            "index_id": "idx",
            "endpoint_id": "ep",
            "deployed_index_id": "dep",
        },
        "batch_paths": {
            "batch_root": "gs://bucket/batch/",
            "delete_root": "gs://bucket/delete/",
        },
        "index_defaults": {
            "shard_size": "SHARD_SIZE_SMALL",
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/creds.json")

    config = load_config(path)

    assert config.project_id == "proj-1"
    assert config.credentials_path == "/tmp/creds.json"
    assert config.embedding_output_dimensionality == 512
    assert config.embedding_text_fields == ["title"]
    assert config.embedding_metadata_fields == ["id", "title"]
    assert config.batch_root == "gs://bucket/batch/"
    assert config.batch_delete_root == "gs://bucket/delete/"

