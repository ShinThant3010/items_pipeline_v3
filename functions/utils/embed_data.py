from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from google.cloud import bigquery
from google.cloud import storage
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from functions.utils.config import AppConfig


# used in embed_data
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
def _as_nonempty_text(text: str) -> str:
    """Ensure we never send empty content to the embedding API."""
    value = (text or "").strip()
    return value if value else " "


# used in embed_data
def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """
    L2-normalize rows of a 2D matrix.

    - Keeps dtype as float32 where possible
    - Avoids division by zero by treating zero-norm rows as norm=1
    """
    if mat.ndim != 2:
        raise ValueError("_l2_normalize expects a 2D array")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


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


def embed_data(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Query BigQuery, embed records, and write JSONL to GCS."""
    rows = list(_rows_from_bq(payload["bigquery_table"], payload["where"]))
    texts = [_as_nonempty_text(_build_text(config, row)) for row in rows]

    vertexai.init(project=config.project_id, location=config.region)
    model = TextEmbeddingModel.from_pretrained(config.embedding_model_name)
    inputs = [TextEmbeddingInput(text=text, task_type="RETRIEVAL_DOCUMENT") for text in texts]
    output_dimensionality = int(payload.get("dimension", config.embedding_output_dimensionality))
    embeddings = model.get_embeddings(
        inputs,
        output_dimensionality=output_dimensionality,
    )

    vectors = np.asarray([embedding.values for embedding in embeddings], dtype=np.float32)
    vectors = _l2_normalize(vectors)

    json_items: list[dict[str, Any]] = []
    for row, vector in zip(rows, vectors):
        entry: dict[str, Any] = {
            "id": str(row.get("id")),
            "embedding": vector.tolist(),
            "embedding_metadata": _build_metadata(config, row),
        }

        restricts = _build_restricts(config, row)
        if restricts:
            entry["restricts"] = restricts

        numeric_restricts = _build_numeric_restricts(config, row)
        if numeric_restricts:
            entry["numeric_restricts"] = numeric_restricts

        json_items.append(entry)

    gcs_prefix = payload.get("gcs_output_prefix") or config.batch_root
    if not gcs_prefix:
        raise ValueError("gcs_output_prefix is required for embedding output")
    bucket, path = _parse_gcs_prefix(gcs_prefix)
    gcs_uri = _write_jsonl_to_gcs(bucket, path, json_items)

    return {
        "status": "EMBEDDED",
        "gcs_output_prefix": gcs_prefix,
        "gcs_output_file": gcs_uri,
        "row_count": len(rows),
    }


__all__ = ["embed_data", "_parse_gcs_prefix"]
