from __future__ import annotations

import json
from typing import Any

from google.cloud import aiplatform
from google.cloud import storage
from google.cloud.aiplatform_v1.types import index as gca_index
from google.protobuf.struct_pb2 import Struct

from functions.utils.config import AppConfig
from functions.utils.embed_data import embed_data, _parse_gcs_prefix

"""
    Core Functions in API call: 
        1) embed_data
        2) streaming_update
        3) streaming_delete
        4) batch_update
"""

# used in batch_update
def _list_gcs_files(bucket_name: str, prefix: str) -> list[str]:
    """List files under a GCS prefix and return their URIs."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix.rstrip("/") + "/")
    uris = [f"gs://{bucket_name}/{blob.name}" for blob in blobs if not blob.name.endswith("/")]
    return uris

# used in streaming_update/_build_index_datapoints
def _struct_from_dict(data: dict[str, Any] | None) -> Struct | None:
    """Convert a dict into a protobuf Struct."""
    if not data:
        return None
    struct = Struct()
    struct.update(data)
    return struct

# used in streaming_update 
def _load_datapoints_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Load datapoints from API payload or a GCS folder."""
    source = payload.get("datapoints_source", "api")
    if source == "gcs":
        gcs_prefix = payload.get("datapoints_gcs_prefix")
        if not gcs_prefix:
            raise ValueError("datapoints_gcs_prefix is required when datapoints_source is gcs")
        bucket_name, prefix = _parse_gcs_prefix(gcs_prefix)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        items: list[dict[str, Any]] = []
        for blob in bucket.list_blobs(prefix=prefix.rstrip("/") + "/"):
            if blob.name.endswith("/"):
                continue
            content = blob.download_as_text()
            line_count = 0
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                items.append(json.loads(line))
            print(f"Loaded {line_count} lines from gs://{bucket_name}/{blob.name}")
        return items
    return payload.get("datapoints", [])

# used in streaming_update 
def _build_index_datapoints(items: list[dict[str, Any]]) -> list[gca_index.IndexDatapoint]:
    """Convert dict payloads into IndexDatapoint objects."""
    datapoints: list[gca_index.IndexDatapoint] = []
    for item in items:
        embedding_metadata = _struct_from_dict(item.get("embedding_metadata"))

        restricts: list[gca_index.IndexDatapoint.Restriction] = []
        for restrict in item.get("restricts", []) or []:
            restricts.append(
                gca_index.IndexDatapoint.Restriction(
                    namespace=restrict.get("namespace", ""),
                    allow_list=restrict.get("allow") or restrict.get("allow_list") or [],
                    deny_list=restrict.get("deny") or restrict.get("deny_list") or [],
                )
            )
        numeric_restricts = [
            gca_index.IndexDatapoint.NumericRestriction(**restrict)
            for restrict in item.get("numeric_restricts", []) or []
        ]

        datapoints.append(
            gca_index.IndexDatapoint(
                datapoint_id=str(item.get("id")),
                feature_vector=item.get("embedding", []),
                restricts=restricts,
                numeric_restricts=numeric_restricts,
                embedding_metadata=embedding_metadata,
            )
        )
    return datapoints


def streaming_update(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Streamingly upsert datapoints into a Vector Search index."""
    index_id = payload.get("index_id") or config.index_id
    if not index_id:
        raise ValueError("index_id is required for streaming update")

    items = _load_datapoints_payload(payload)
    datapoints = _build_index_datapoints(items)

    aiplatform.init(project=config.project_id, location=config.region)
    index = aiplatform.MatchingEngineIndex(index_name=index_id)
    index.upsert_datapoints(datapoints=datapoints)

    return {
        "index_id": index_id,
        "upserted": len(datapoints),
    }


def streaming_delete(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Streamingly delete datapoints from a Vector Search index."""
    index_id = payload.get("index_id") or config.index_id
    if not index_id:
        raise ValueError("index_id is required for streaming delete")

    ids = [str(item) for item in payload.get("datapoint_ids", [])]
    if not ids:
        raise ValueError("datapoint_ids must not be empty")

    aiplatform.init(project=config.project_id, location=config.region)
    index = aiplatform.MatchingEngineIndex(index_name=index_id)
    index.remove_datapoints(datapoint_ids=ids)

    return {
        "index_id": index_id,
        "deleted": len(ids),
    }


def batch_update(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Run batch update using pre-embedded JSONL in GCS."""
    index_id = payload.get("index_id") or config.index_id
    if not index_id:
        raise ValueError("index_id is required for batch update")

    gcs_prefix = payload.get("contents_delta_uri") or config.batch_root
    if not gcs_prefix:
        raise ValueError("contents_delta_uri is required for batch update")
    bucket, path = _parse_gcs_prefix(gcs_prefix)
    files = _list_gcs_files(bucket, path)
    if not files:
        raise FileNotFoundError(f"No files found in {gcs_prefix}")
    print("Batch update using files:")
    for uri in files:
        print(f" - {uri}")

    aiplatform.init(project=config.project_id, location=config.region)
    index = aiplatform.MatchingEngineIndex(index_name=index_id)
    index.update_embeddings(
        contents_delta_uri=gcs_prefix,
        is_complete_overwrite=bool(payload.get("is_complete_overwrite", False)),
    )
    return {
        "status": "STARTED",
        "index_id": index_id,
        "files": files,
        "contents_delta_uri": gcs_prefix,
    }
