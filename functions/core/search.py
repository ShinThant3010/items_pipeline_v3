from __future__ import annotations

import json
import math
import time
from typing import Any

from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
from google.cloud import storage
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from functions.utils.config import AppConfig


def _extract_neighbor(neighbor: Any) -> dict[str, Any]:
    datapoint = getattr(neighbor, "datapoint", None)
    if datapoint is not None:
        neighbor_id = getattr(datapoint, "datapoint_id", None) or getattr(datapoint, "id", None)
        metadata = getattr(datapoint, "embedding_metadata", None) or getattr(datapoint, "metadata", None)
    else:
        neighbor_id = getattr(neighbor, "id", None)
        metadata = None

    score = getattr(neighbor, "distance", None)
    if score is None:
        score = getattr(neighbor, "score", None)

    return {
        "id": neighbor_id,
        "score": score,
        "metadata": metadata,
    }


def _build_namespace_filters(restricts: list[dict[str, Any]] | None) -> list[Namespace]:
    filters: list[Namespace] = []
    for item in restricts or []:
        namespace = item.get("namespace") or item.get("name")
        if not namespace:
            continue
        allow = item.get("allow") or item.get("allow_list") or item.get("allow_tokens") or []
        deny = item.get("deny") or item.get("deny_list") or item.get("deny_tokens") or []
        filters.append(Namespace(namespace, list(allow), list(deny)))
    return filters

def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return values
    return [v / norm for v in values]

def _parse_gcs_prefix(prefix: str) -> tuple[str, str]:
    """Split a gs:// URI into bucket and object prefix."""
    if not prefix.startswith("gs://"):
        raise ValueError("metadata_gcs_prefix must start with gs://")
    remainder = prefix[5:]
    bucket, _, path = remainder.partition("/")
    return bucket, path


def _load_metadata_from_gcs(
    gcs_prefix: str,
    ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Lookup embedding_metadata from JSONL files in GCS for given ids."""
    if not ids:
        return {}

    bucket_name, prefix = _parse_gcs_prefix(gcs_prefix)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    found: dict[str, dict[str, Any]] = {}

    for blob in bucket.list_blobs(prefix=prefix.rstrip("/") + "/"):
        if blob.name.endswith("/"):
            continue
        content = blob.download_as_text()
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            record_id = str(record.get("id", ""))
            if record_id in ids and record.get("embedding_metadata"):
                found[record_id] = record["embedding_metadata"]
                if len(found) == len(ids):
                    return found
    return found


def search_index(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Search the deployed index with dense embeddings only."""
    endpoint_id = payload.get("endpoint_id") or config.endpoint_id
    deployed_index_id = payload.get("deployed_index_id") or config.deployed_index_id
    if not endpoint_id or not deployed_index_id:
        raise ValueError("endpoint_id and deployed_index_id are required")

    query_type = (payload.get("query_type") or "vector").lower()
    query = payload.get("query", "")
    top_k = int(payload.get("top_k", 10))
    metadata_gcs_prefix = payload.get("metadata_gcs_prefix") or config.batch_root
    restricts = payload.get("restricts")

    if query_type == "text":
        if not isinstance(query, str):
            raise ValueError("query must be a string when query_type is 'text'")
        embed_start = time.monotonic()
        vertexai.init(project=config.project_id, location=config.region)
        model = TextEmbeddingModel.from_pretrained(config.embedding_model_name)
        embedding = model.get_embeddings(
            [TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")],# semantic similarity task
            output_dimensionality=config.embedding_output_dimensionality,
        )[0]
        print(f"[search] embed runtime_sec={time.monotonic() - embed_start:.6f}")
        embedding_values = _l2_normalize(list(embedding.values))
    elif query_type == "vector":
        if not isinstance(query, list) or not all(isinstance(v, (float, int)) for v in query):
            raise ValueError("query must be a list of numbers when query_type is 'vector'")
        embedding_values = [float(v) for v in query]
    else:
        raise ValueError("query_type must be 'text' or 'vector'")

    aiplatform.init(project=config.project_id, location=config.region)
    endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id)
    filters = _build_namespace_filters(restricts)
    neighbors_start = time.monotonic()
    neighbors = endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[embedding_values],
        num_neighbors=top_k,
        return_full_datapoint=True,
        filter=filters or None,
    )
    print(f"[search] find_neighbors runtime_sec={time.monotonic() - neighbors_start:.6f}")

    results = []
    if neighbors:
        results = [_extract_neighbor(n) for n in neighbors[0]]

    missing_ids = {item["id"] for item in results if not item.get("metadata") and item.get("id")}
    if missing_ids and metadata_gcs_prefix:
        metadata_start = time.monotonic()
        metadata = _load_metadata_from_gcs(metadata_gcs_prefix, missing_ids)
        if metadata:
            for item in results:
                item_id = item.get("id")
                if item_id in metadata and not item.get("metadata"):
                    item["metadata"] = metadata[item_id]
    # print(f"[search] metadata retrieval runtime_sec={time.monotonic() - metadata_start:.6f}")

    return {
        "query": query,
        "query_type": query_type,
        "num_recommendations": len(results),
        "results": results,
    }
