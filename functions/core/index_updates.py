from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from google.cloud import bigquery
from google.cloud import aiplatform
from google.cloud import storage
from google.cloud.aiplatform_v1.types import index as gca_index
from google.protobuf.struct_pb2 import Struct
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from functions.utils.config import AppConfig

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
"""
    Core Functions in API call: 
        1) embed_data
        2) streaming_update
        3) streaming_delete
        4) batch_update
"""

# used in embed_data, streaming_update/_load_datapoints_payload, batch_update
def _parse_gcs_prefix(prefix: str) -> tuple[str, str]:
    """Split a gs:// URI into bucket and object prefix."""
    if not prefix.startswith("gs://"):
        raise ValueError("gcs_output_prefix must start with gs://")
    remainder = prefix[5:]
    bucket, _, path = remainder.partition("/")
    return bucket, path

# used in embed_data/_build_numeric_restricts
def _parse_timestamp(value: Any) -> int | None:
    """Parse timestamps into epoch seconds, if possible."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return int(value.timestamp())
    if isinstance(value, (int, float)):
        return int(value)
    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return int(datetime.strptime(str(value), fmt).timestamp())
        except ValueError:
            continue
    return None

# used in embed_data/_build_bm25
def _tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text into alphanumeric terms."""
    return _TOKEN_RE.findall(text.lower())

# used in embed_data
def _build_bm25(
    texts: list[str],
    sparse_dimensions: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[dict[str, list[float | int]]]:
    """Build sparse BM25-like vectors using hashed term buckets."""
    doc_tokens = [_tokenize(text) for text in texts]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)

    df: dict[str, int] = {}
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    total_docs = len(doc_tokens)
    sparse_vectors: list[dict[str, list[float | int]]] = []
    for tokens, dl in zip(doc_tokens, doc_lengths):
        term_freq: dict[str, int] = {}
        for term in tokens:
            term_freq[term] = term_freq.get(term, 0) + 1

        bucket_scores: dict[int, float] = {}
        for term, tf in term_freq.items():
            term_df = df.get(term, 1)
            idf = math.log((total_docs - term_df + 0.5) / (term_df + 0.5) + 1)
            denom = tf + k1 * (1 - b + b * (dl / max(avgdl, 1)))
            score = idf * ((tf * (k1 + 1)) / denom)
            idx = hash(term) % sparse_dimensions
            bucket_scores[idx] = bucket_scores.get(idx, 0.0) + score

        dimensions = list(bucket_scores.keys())
        values = [bucket_scores[idx] for idx in dimensions]
        sparse_vectors.append({"dimensions": dimensions, "values": values})

    return sparse_vectors

# used in embed_data
def _build_restricts(config: AppConfig, row: dict[str, Any]) -> list[dict[str, Any]]:
    """Create categorical restricts from configured fields."""
    restricts: list[dict[str, Any]] = []
    for field in config.restricts_fields:
        value = row.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, (list, tuple)):
            allow = [str(item) for item in value if item not in (None, "")]
        else:
            allow = [str(value)]
        if allow:
            restricts.append({"namespace": field, "allow": allow})
    return restricts

# used in embed_data
def _build_numeric_restricts(config: AppConfig, row: dict[str, Any]) -> list[dict[str, Any]]:
    """Create numeric restricts (including timestamp fields)."""
    numeric_restricts: list[dict[str, Any]] = []
    for field in config.numeric_restricts_fields:
        raw = row.get(field)
        value = _parse_timestamp(raw) if field in {"created_at", "updated_at"} else raw
        if value is None or value == "":
            continue
        if isinstance(value, float):
            numeric_restricts.append({"namespace": field, "value_float": value})
        else:
            numeric_restricts.append({"namespace": field, "value_int": int(value)})
    return numeric_restricts

# used in embed_data
def _build_metadata(config: AppConfig, row: dict[str, Any]) -> dict[str, Any]:
    """Select metadata fields to return with search results."""
    return {field: row.get(field) for field in config.embedding_metadata_fields if field in row}

# used in embed_data
def _build_text(config: AppConfig, row: dict[str, Any]) -> str:
    """Concatenate embedding text fields into a single document string."""
    parts: list[str] = []
    for field in config.embedding_text_fields:
        value = row.get(field)
        if value is None or value == "":
            continue
        parts.append(str(value))
    return "\n".join(parts)

# used in embed_data
def _rows_from_bq(table: str, where_clause: str) -> Iterable[dict[str, Any]]:
    """Yield BigQuery rows as dictionaries for the given WHERE clause."""
    client = bigquery.Client()
    query = f"SELECT * FROM `{table}` WHERE {where_clause}"
    for row in client.query(query):
        yield dict(row.items())

# used in embed_data
def _write_jsonl_to_gcs(bucket_name: str, prefix: str, items: list[dict[str, Any]]) -> str:
    """Write JSONL embeddings to GCS and return the object URI."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_name = f"{prefix.rstrip('/')}/part-00000.json"
    temp_path = Path("/tmp") / "vector-search-part-00000.json"
    lines = [json.dumps(item, ensure_ascii=True) for item in items]
    temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bucket.blob(blob_name).upload_from_filename(str(temp_path))
    return f"gs://{bucket_name}/{blob_name}"

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
        sparse = item.get("sparse_embedding") or {}
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
                sparse_embedding=gca_index.IndexDatapoint.SparseEmbedding(
                    dimensions=sparse.get("dimensions", []),
                    values=sparse.get("values", []),
                )
                if sparse
                else None,
                restricts=restricts,
                numeric_restricts=numeric_restricts,
                embedding_metadata=embedding_metadata,
            )
        )
    return datapoints


def embed_data(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Query BigQuery, embed records, and write JSONL to GCS."""
    use_bm25 = bool(payload.get("use_bm25", config.embedding_enable_bm25))
    rows = list(_rows_from_bq(payload["bigquery_table"], payload["where"]))
    texts = [_build_text(config, row) for row in rows]

    vertexai.init(project=config.project_id, location=config.region)
    model = TextEmbeddingModel.from_pretrained(config.embedding_model_name)
    inputs = [TextEmbeddingInput(text=text, task_type="RETRIEVAL_DOCUMENT") for text in texts]
    embeddings = model.get_embeddings(
        inputs,
        output_dimensionality=config.embedding_output_dimensionality,
    )

    sparse_vectors: list[dict[str, list[float | int]]] = []
    if use_bm25 and config.embedding_sparse_dimensions > 0:
        sparse_vectors = _build_bm25(texts, config.embedding_sparse_dimensions)
    else:
        sparse_vectors = [{} for _ in texts]

    json_items: list[dict[str, Any]] = []
    for row, embedding, sparse in zip(rows, embeddings, sparse_vectors):
        entry: dict[str, Any] = {
            "id": str(row.get("id")),
            "embedding": embedding.values,
            "embedding_metadata": _build_metadata(config, row),
        }

        restricts = _build_restricts(config, row)
        if restricts:
            entry["restricts"] = restricts

        numeric_restricts = _build_numeric_restricts(config, row)
        if numeric_restricts:
            entry["numeric_restricts"] = numeric_restricts

        if sparse and sparse.get("dimensions") and sparse.get("values"):
            entry["sparse_embedding"] = sparse

        json_items.append(entry)

    gcs_prefix = payload.get("gcs_output_prefix") or config.batch_root
    if not gcs_prefix:
        raise ValueError("gcs_output_prefix is required for embedding output")
    bucket, path = _parse_gcs_prefix(gcs_prefix)
    gcs_uri = _write_jsonl_to_gcs(bucket, path, json_items)

    return {
        "run_id": payload.get("run_id"),
        "status": "EMBEDDED",
        "gcs_output_prefix": gcs_prefix,
        "gcs_output_file": gcs_uri,
        "row_count": len(rows),
    }


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
    """Run batch CRUD update using GCS staging + Vertex AI update."""
    use_preembedded = bool(payload.get("use_preembedded", False))
    embed_result = None
    if not use_preembedded:
        embed_result = embed_data(config, payload)
    index_id = payload.get("index_id") or config.index_id
    if not index_id:
        raise ValueError("index_id is required for batch update")

    gcs_prefix = payload.get("gcs_output_prefix") or config.batch_root
    if not gcs_prefix:
        raise ValueError("gcs_output_prefix is required for batch update")
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
        is_complete_overwrite=False,
    )
    return {
        "run_id": payload.get("run_id"),
        "status": "STARTED",
        "update_type": payload.get("update_type"),
        "embedding": embed_result,
        "used_preembedded": use_preembedded,
        "index_id": index_id,
        "files": files,
        "gcs_output_prefix": gcs_prefix,
    }
