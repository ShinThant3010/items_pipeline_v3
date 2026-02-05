# API Spec

## Base URLs
- Production (Cloud Run): `https://hyde-item-pipeline-api-<project>.run.app`
- Staging (Cloud Run): `https://hyde-item-pipeline-api-<env>.run.app`
- Local (Uvicorn/FastAPI): `http://127.0.0.1:8080`
- Swagger/OpenAPI: `https://hyde-item-pipeline-api-<project>.<region>.run.app/docs`

Base path: `/v1`

## POST /index/create
Create a Vertex AI Vector Search index.

Required fields:
- `display_name`
- `dimensions`

Defaults if not specified:
- `shard_size`
- `distance_measure_type`
- `feature_norm_type`
- `index_update_method`
- `approximate_neighbors_count`
- `leaf_node_embedding_count`
- `leaf_nodes_to_search_percent`

Request:
```json
{
  "display_name": "items-index",
  "description": "Items index",
  "dimensions": 768,
  "shard_size": "SHARD_SIZE_SMALL",
  "distance_measure_type": "DOT_PRODUCT",
  "feature_norm_type": "UNIT_L2_NORM",
  "index_update_method": "STREAM_UPDATE",
  "approximate_neighbors_count": 150,
  "leaf_node_embedding_count": 1000,
  "leaf_nodes_to_search_percent": 5
}
```

Response:
```json
{
  "index_id": "projects/.../locations/.../indexes/...",
  "status": "CREATED"
}
```

## POST /embed_data
Query BigQuery, embed, and write JSONL to GCS.

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
  "row_count": 1000
}
```

Notes:
- Embeddings are written as returned by the model.

## POST /streaming/update
Upsert datapoints into an index.

Request (datapoints from GCS):
```json
{
  "index_id": "projects/.../locations/.../indexes/...",
  "datapoints_source": "gcs",
  "datapoints_gcs_prefix": "gs://bucket/path/"
}
```

Request (datapoints in payload):
```json
{
  "index_id": "projects/.../locations/.../indexes/...",
  "datapoints_source": "api",
  "datapoints": [
    {
      "id": "...",
      "embedding": [0.0157, -0.0113, 0.0268],
      "restricts": [
        {"namespace": "level", "allow": ["Intermediate"]}
      ],
      "numeric_restricts": [
        {"namespace": "created_at", "value_int": 1700000000}
      ],
      "embedding_metadata": {
        "id": "...",
        "description": "description"
      }
    }
  ]
}
```

Response:
```json
{
  "index_id": "projects/.../locations/.../indexes/...",
  "upserted": 100
}
```

## POST /streaming/delete
Delete datapoints from an index.

Request:
```json
{
  "index_id": "projects/.../locations/.../indexes/...",
  "datapoint_ids": [
    "01KAWP7412YBV991H15H14CKJV"
  ]
}
```

Response:
```json
{
  "index_id": "projects/.../locations/.../indexes/...",
  "deleted": 1
}
```

## POST /batch/updates
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

## POST /endpoint/create
Create a Vector Search endpoint.

Request:
```json
{
  "display_name": "items-endpoint",
  "description": "Endpoint for items index",
  "public_endpoint_enabled": true
}
```

Response:
```json
{
  "endpoint_id": "projects/.../locations/.../indexEndpoints/...",
  "status": "CREATED"
}
```

## POST /endpoint/deploy
Deploy an index to an endpoint.

Request:
```json
{
  "endpoint_id": "projects/.../locations/.../indexEndpoints/...",
  "index_id": "projects/.../locations/.../indexes/...",
  "deployed_index_id": "items-deployed",
  "machine_type": "e2-standard-2",
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

## POST /search
Search the deployed index with dense vector similarity.

Request (query_type: vector):
```json
{
  "endpoint_id": "projects/.../locations/.../indexEndpoints/...",
  "deployed_index_id": "items_deployed",
  "query_type": "vector",
  "query": [0.1, 0.2, 0.3],
  "top_k": 10,
  "metadata_gcs_prefix": "gs://bucket/path/",
  "restricts": [
    {"namespace": "level", "allow": ["Beginner"]}
  ]
}
```

Request (query_type: text):
```json
{
  "endpoint_id": "projects/.../locations/.../indexEndpoints/...",
  "deployed_index_id": "items_deployed",
  "query_type": "text",
  "query": "intro to python",
  "top_k": 10,
  "metadata_gcs_prefix": "gs://bucket/path/",
  "restricts": [
    {"namespace": "level", "allow": ["Beginner"]}
  ]
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
        "lesson_title": "title",
        "link": "https://www.edx.org/xxx"
      }
    }
  ]
}
```

Notes:
- For `query_type: "text"`, the generated query embedding is L2-normalized before search.
- For `query_type: "vector"`, the vector is used as-is.

## GET /health
Health check.

Response:
```json
{
  "status": "ok"
}
```
