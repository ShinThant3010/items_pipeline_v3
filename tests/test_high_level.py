from types import SimpleNamespace

from functions.core import index_updates, search as search_module, vertex_index
from functions.utils.config import AppConfig


def _config() -> AppConfig:
    return AppConfig(
        project_id="proj",
        region="us-central1",
        gcs_bucket="gs://bucket/",
        bq_default_dataset="dataset",
        credentials_path=None,
        restricts_fields=["level"],
        numeric_restricts_fields=["created_at"],
        embedding_text_fields=["title"],
        embedding_metadata_fields=["id", "title"],
        embedding_model_name="model",
        embedding_output_dimensionality=3,
        index_id="idx",
        endpoint_id="ep",
        deployed_index_id="dep",
        batch_root="gs://bucket/batch/",
        batch_delete_root="gs://bucket/delete/",
        index_defaults={},
    )


def test_search_index_dense_flow(monkeypatch):
    config = _config()
    calls = {}

    class _FakeModel:
        def get_embeddings(self, inputs, output_dimensionality):
            calls["embedding_inputs"] = inputs
            calls["embedding_output_dimensionality"] = output_dimensionality
            return [SimpleNamespace(values=[0.1, 0.2, 0.3])]

    class _FakeEndpoint:
        def find_neighbors(self, deployed_index_id, queries, num_neighbors, return_full_datapoint, filter=None):
            calls["deployed_index_id"] = deployed_index_id
            calls["queries"] = queries
            calls["num_neighbors"] = num_neighbors
            calls["return_full_datapoint"] = return_full_datapoint
            calls["filter"] = filter
            neighbor = SimpleNamespace(id="item-1", distance=0.5)
            return [[neighbor]]

    monkeypatch.setattr(search_module.vertexai, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        search_module.TextEmbeddingModel,
        "from_pretrained",
        lambda name: _FakeModel(),
    )
    monkeypatch.setattr(search_module.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        search_module.aiplatform,
        "MatchingEngineIndexEndpoint",
        lambda index_endpoint_name: _FakeEndpoint(),
    )
    monkeypatch.setattr(
        search_module,
        "_load_metadata_from_gcs",
        lambda gcs_prefix, ids: calls.setdefault("metadata_lookup", (gcs_prefix, ids)) or {},
    )

    payload = {"query_type": "text", "query": "hello", "top_k": 5}
    result = search_module.search_index(config, payload)

    assert result["query"] == "hello"
    assert result["results"][0]["id"] == "item-1"
    assert result["results"][0]["score"] == 0.5
    assert calls["embedding_output_dimensionality"] == config.embedding_output_dimensionality
    assert calls["deployed_index_id"] == config.deployed_index_id
    assert calls["queries"] == [[0.1, 0.2, 0.3]]
    assert calls["num_neighbors"] == 5
    assert calls["return_full_datapoint"] is True
    assert calls["filter"] is None
    assert calls["metadata_lookup"][0] == config.batch_root
    assert calls["metadata_lookup"][1] == {"item-1"}


def test_search_index_vector_flow(monkeypatch):
    config = _config()
    calls = {}

    class _FakeEndpoint:
        def find_neighbors(self, deployed_index_id, queries, num_neighbors, return_full_datapoint, filter=None):
            calls["deployed_index_id"] = deployed_index_id
            calls["queries"] = queries
            calls["num_neighbors"] = num_neighbors
            calls["return_full_datapoint"] = return_full_datapoint
            calls["filter"] = filter
            neighbor = SimpleNamespace(id="item-2", distance=0.25)
            return [[neighbor]]

    monkeypatch.setattr(search_module.vertexai, "init", lambda **kwargs: None)
    monkeypatch.setattr(search_module.TextEmbeddingModel, "from_pretrained", lambda name: None)
    monkeypatch.setattr(search_module.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        search_module.aiplatform,
        "MatchingEngineIndexEndpoint",
        lambda index_endpoint_name: _FakeEndpoint(),
    )
    monkeypatch.setattr(
        search_module,
        "_load_metadata_from_gcs",
        lambda gcs_prefix, ids: calls.setdefault("metadata_lookup", (gcs_prefix, ids)) or {},
    )

    payload = {"query_type": "vector", "query": [0.4, 0.5, 0.6], "top_k": 3}
    result = search_module.search_index(config, payload)

    assert result["query_type"] == "vector"
    assert result["results"][0]["id"] == "item-2"
    assert calls["deployed_index_id"] == config.deployed_index_id
    assert calls["queries"] == [[0.4, 0.5, 0.6]]
    assert calls["num_neighbors"] == 3
    assert calls["return_full_datapoint"] is True
    assert calls["filter"] is None


def test_embed_data_writes_jsonl(monkeypatch):
    config = _config()
    rows = [{"id": "1", "title": "Hello", "created_at": 1700000000}]
    calls = {}

    class _FakeModel:
        def get_embeddings(self, inputs, output_dimensionality):
            calls["embedding_inputs"] = inputs
            calls["embedding_output_dimensionality"] = output_dimensionality
            return [SimpleNamespace(values=[0.1, 0.2, 0.3])]

    monkeypatch.setattr(index_updates, "_rows_from_bq", lambda table, where: rows)
    monkeypatch.setattr(index_updates.vertexai, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        index_updates.TextEmbeddingModel,
        "from_pretrained",
        lambda name: _FakeModel(),
    )
    def _fake_parse_gcs_prefix(prefix):
        calls["parse_gcs_prefix"] = prefix
        return ("bucket", "path")

    monkeypatch.setattr(index_updates, "_parse_gcs_prefix", _fake_parse_gcs_prefix)
    def _fake_write_jsonl(bucket, path, items):
        calls["write_jsonl_args"] = (bucket, path, items)
        return "gs://bucket/path/part-00000.json"

    monkeypatch.setattr(index_updates, "_write_jsonl_to_gcs", _fake_write_jsonl)

    payload = {
        "bigquery_table": "proj.dataset.table",
        "where": "TRUE",
        "gcs_output_prefix": "gs://bucket/path/",
        "dimension": 3,
    }
    result = index_updates.embed_data(config, payload)

    assert result["status"] == "EMBEDDED"
    assert result["row_count"] == 1
    assert result["gcs_output_file"].endswith("part-00000.json")
    assert calls["embedding_output_dimensionality"] == config.embedding_output_dimensionality
    assert calls["parse_gcs_prefix"] == "gs://bucket/path/"
    bucket, path, items = calls["write_jsonl_args"]
    assert bucket == "bucket"
    assert path == "path"
    assert items[0]["id"] == "1"
    assert "embedding_metadata" in items[0]


def test_deploy_index_calls_endpoint(monkeypatch):
    config = _config()
    called = {}

    class _FakeEndpoint:
        def __init__(self, index_endpoint_name):
            called["endpoint_id"] = index_endpoint_name

        def deploy_index(self, index, deployed_index_id, machine_type, min_replica_count, max_replica_count):
            called["index"] = index
            called["deployed_index_id"] = deployed_index_id
            called["min_replica_count"] = min_replica_count
            called["max_replica_count"] = max_replica_count

    class _FakeIndex:
        def __init__(self, index_name):
            called["index_id"] = index_name

    monkeypatch.setattr(vertex_index.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        vertex_index.aiplatform,
        "MatchingEngineIndexEndpoint",
        lambda index_endpoint_name: _FakeEndpoint(index_endpoint_name),
    )
    monkeypatch.setattr(
        vertex_index.aiplatform,
        "MatchingEngineIndex",
        lambda index_name: _FakeIndex(index_name),
    )
    monkeypatch.setattr(vertex_index, "update_resource_names", lambda **kwargs: called.update(kwargs))

    payload = {
        "endpoint_id": "projects/p/locations/r/indexEndpoints/ep",
        "index_id": "projects/p/locations/r/indexes/idx",
        "deployed_index_id": "dep",
        "min_replica_count": 1,
        "max_replica_count": 2,
    }
    result = vertex_index.deploy_index(config, payload)

    assert result["status"] == "DEPLOYED"
    assert called["endpoint_id"] == payload["endpoint_id"]
    assert called["index_id"] == payload["index_id"]
    assert called["deployed_index_id"] == "dep"
    assert called["min_replica_count"] == 1
    assert called["max_replica_count"] == 2


def test_batch_update_uses_contents_delta_uri(monkeypatch):
    config = _config()
    calls = {}

    monkeypatch.setattr(index_updates, "_parse_gcs_prefix", lambda prefix: ("bucket", "path"))
    monkeypatch.setattr(index_updates, "_list_gcs_files", lambda bucket, path: ["gs://bucket/path/part-00000.json"])
    monkeypatch.setattr(index_updates.aiplatform, "init", lambda **kwargs: None)

    class _FakeIndex:
        def __init__(self, index_name):
            calls["index_id"] = index_name

        def update_embeddings(self, contents_delta_uri, is_complete_overwrite):
            calls["contents_delta_uri"] = contents_delta_uri
            calls["is_complete_overwrite"] = is_complete_overwrite

    monkeypatch.setattr(
        index_updates.aiplatform,
        "MatchingEngineIndex",
        lambda index_name: _FakeIndex(index_name),
    )

    payload = {
        "index_id": "projects/p/locations/r/indexes/idx",
        "contents_delta_uri": "gs://bucket/path/",
        "is_complete_overwrite": True,
    }
    result = index_updates.batch_update(config, payload)

    assert result["status"] == "STARTED"
    assert result["contents_delta_uri"] == "gs://bucket/path/"
    assert calls["index_id"] == payload["index_id"]
    assert calls["contents_delta_uri"] == "gs://bucket/path/"
    assert calls["is_complete_overwrite"] is True
