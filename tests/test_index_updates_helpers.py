from datetime import datetime

from functions.core import index_updates
from functions.utils.config import AppConfig


def _config() -> AppConfig:
    return AppConfig(
        project_id="proj",
        region="us-central1",
        gcs_bucket="gs://bucket/",
        bq_default_dataset="dataset",
        credentials_path=None,
        restricts_fields=["level", "category_id"],
        numeric_restricts_fields=["created_at", "score"],
        embedding_text_fields=["title", "desc"],
        embedding_metadata_fields=["id", "title"],
        embedding_model_name="model",
        embedding_output_dimensionality=768,
        index_id="idx",
        endpoint_id="ep",
        deployed_index_id="dep",
        batch_root="gs://bucket/batch/",
        batch_delete_root="gs://bucket/delete/",
        index_defaults={},
    )


def test_parse_timestamp_handles_datetime_and_string():
    ts = datetime(2024, 1, 1, 12, 0, 0)
    assert index_updates._parse_timestamp(ts) == int(ts.timestamp())
    assert index_updates._parse_timestamp("2024-01-01 12:00:00") == int(ts.timestamp())


def test_build_text_and_metadata():
    config = _config()
    row = {"title": "Hello", "desc": "World", "id": "1"}
    assert index_updates._build_text(config, row) == "Hello\nWorld"
    assert index_updates._build_metadata(config, row) == {"id": "1", "title": "Hello"}


def test_build_restricts_and_numeric_restricts():
    config = _config()
    row = {
        "level": "Beginner",
        "category_id": ["cat1", "cat2"],
        "created_at": 1700000000,
        "score": 1.5,
    }
    restricts = index_updates._build_restricts(config, row)
    assert {"namespace": "level", "allow": ["Beginner"]} in restricts
    assert {"namespace": "category_id", "allow": ["cat1", "cat2"]} in restricts

    numeric = index_updates._build_numeric_restricts(config, row)
    assert {"namespace": "created_at", "value_int": 1700000000} in numeric
    assert {"namespace": "score", "value_float": 1.5} in numeric


def test_build_index_datapoints_minimal():
    items = [
        {
            "id": "1",
            "embedding": [0.1, 0.2],
            "restricts": [{"namespace": "level", "allow": ["Beginner"]}],
            "numeric_restricts": [{"namespace": "score", "value_float": 1.5}],
            "embedding_metadata": {"id": "1", "title": "Hello"},
        }
    ]
    datapoints = index_updates._build_index_datapoints(items)
    assert len(datapoints) == 1
    assert datapoints[0].datapoint_id == "1"
