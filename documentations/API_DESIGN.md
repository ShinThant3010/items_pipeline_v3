# API Design

Base path: `/v1`

## Authentication
- Service‑to‑service auth (IAM / OIDC)

## Lifecycle Endpoints

### POST /indexes
Create a Vertex AI Vector Search index.

Request:
```json
{
  "display_name": "items-index",
  "description": "Items index",
  "dimensions": 768,
  "distance_measure_type": "DOT_PRODUCT",
  "index_update_method": "BATCH_UPDATE",
  "shard_size": "SHARD_SIZE_SMALL"
}
```

Response:
```json
{
  "index_id": "projects/.../locations/.../indexes/...",
  "status": "CREATED"
}
```

### POST /endpoints
Create a Vector Search endpoint.

Request:
```json
{
  "display_name": "items-endpoint",
  "description": "Endpoint for items index"
}
```

Response:
```json
{
  "endpoint_id": "projects/.../locations/.../indexEndpoints/...",
  "status": "CREATED"
}
```

### POST /endpoints/{endpoint_id}/deploy
Deploy an index to an endpoint.

Request:
```json
{
  "index_id": "projects/.../locations/.../indexes/...",
  "deployed_index_id": "items-deployed",
  "min_replica_count": 1,
  "max_replica_count": 1
}
```

Response:
```json
{
  "deployed_index_id": "items-deployed",
  "status": "DEPLOYED"
}
```

## Data Preparation Endpoints

### POST /embed_data
Query BigQuery, embed, and write JSON (one object per line) to GCS.

Request:
```json
{
  "bigquery_table": "project.dataset.table",
  "where": "updated_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)",
  "gcs_output_prefix": "gs://bucket/path/",
  "dimension": 768
}
```

Response:
```json
{
  "status": "EMBEDDED",
  "gcs_output_prefix": "gs://bucket/path/",
  "gcs_output_file": "gs://bucket/path/part-00000.json",
  "row_count": 11890
}
```

## Batch CRUD Endpoints

### POST /batch/updates
Start a batch update using pre-embedded JSONL in GCS.

Request:
```json
{
  "index_id": "projects/.../locations/.../indexes/...",
  "contents_delta_uri": "gs://bucket/path/",
  "is_complete_overwrite": false
}
```

Response:
```json
{
  "status": "STARTED",
  "index_id": "projects/.../locations/.../indexes/...",
  "files": ["gs://bucket/path/part-00000.json"],
  "contents_delta_uri": "gs://bucket/path/"
}
```

Notes:
- `update_type` supports `SCHEDULED` (weekly) or `MANUAL` (ad‑hoc)

### GET /batch/updates/{id}
Get batch run status.

Response:
```json
{
  "id": "batch-1",
  "status": "SUCCEEDED",
  "counts": {
    "read": 12000,
    "embedded": 11890,
    "written": 11890,
    "ingested": 11890
  },
  "started_at": "2026-01-26T01:00:00Z",
  "completed_at": "2026-01-26T01:45:00Z"
}
```

Counts description:
- `read`: rows pulled from BigQuery for this run
- `embedded`: rows successfully embedded
- `written`: rows written to GCS JSON shards
- `ingested`: rows accepted by Vertex AI batch update
- Any of the above can be lower than `read` due to validation failures, embedding errors, or ingestion rejects

## Search Endpoint

### POST /search
Search the deployed index with dense vector similarity.

Request:
```json
{
  "endpoint_id": "projects/.../locations/.../indexEndpoints/...",
  "query": "intro to python",
  "top_k": 10
}
```

Response:
```json
{
  "query": "intro to python",
  "results": [
    {
      "id": "01KAWP7409ZH98683BM8SRW2C1",
      "score": 0.83,
      "metadata": {
        "lesson_title": "Programming for Everybody (Getting Started with Python)",
        "link": "https://www.edx.org/course/programming-for-everybody-getting-started-with-pyt"
      }
    }
  ]
}
```

### POST /batch/updates/{id}/cancel
Cancel a running batch update.

Response:
```json
{
  "id": "batch-1",
  "status": "CANCELLED"
}
```

## Error Format
```json
{
  "error": {
    "code": "INVALID_ARGUMENT",
    "message": "Missing index_id",
    "details": {}
  }
}
```

## Notes
- Batch updates write JSON to GCS and then trigger Vertex AI index update
- DOT_PRODUCT and L2_NORM supported for dense vector scoring
