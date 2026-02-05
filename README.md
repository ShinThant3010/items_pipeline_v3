# Vertex AI Vector Search Pipeline

Backend pipeline for managing Vertex AI Vector Search indexes and endpoints, plus batch and streaming updates. It queries data from BigQuery, generates dense embeddings with Gemini, writes them to GCS, and updates the index for search.

## Status
This project is **work in progress** and in the **exploration stage**. Expect breaking changes, incomplete features, and unstable APIs.

## What This Service Does
- Creates Vector Search indexes and endpoints.
- Embeds BigQuery data into dense vectors and writes JSONL to GCS.
- Streams upsert/delete datapoints into an index.
- Runs batch updates from pre-embedded JSONL in GCS.
- Searches a deployed index with dense vector similarity.

## Requirements
- Python `3.11`.
- Google Cloud project with Vertex AI, BigQuery, and GCS enabled.
- Service account with access to Vertex AI, BigQuery, and GCS.

## Quickstart
1. Create a virtualenv and install dependencies.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set Google credentials.
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

3. Update config.
- See `functions/parameters/config.yaml`.

4. Run the API.
```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

5. Confirm health.
```bash
curl http://127.0.0.1:8080/health
```

## Configuration
The service reads `functions/parameters/config.yaml` and the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

Key config fields:
- `project_id`, `region`, `gcs_bucket`, `bq_default_dataset`
- `embedding.model_name`, `embedding.output_dimensionality`, `embedding.text_fields`, `embedding.metadata_fields`
- `resource_names.index_id`, `resource_names.endpoint_id`, `resource_names.deployed_index_id`
- `index_defaults.shard_size`, `index_defaults.distance_measure_type`, `index_defaults.feature_norm_type`

## API Summary
Base path: `/v1`

Endpoints:
- `POST /index/create`
- `POST /embed_data`
- `POST /streaming/update`
- `POST /streaming/delete`
- `POST /batch/updates`
- `POST /endpoint/create`
- `POST /endpoint/deploy`
- `POST /search`

Full request/response examples live in `api_docs/API_spec.md`.

### Search Notes
- For `query_type: "text"`, the generated query embedding is L2-normalized before search.
- For `query_type: "vector"`, the service uses the vector as-is. Provide a normalized vector if your index expects unit L2 norm.

## Example Requests
Text search:
```bash
curl -X POST http://127.0.0.1:8080/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "text",
    "query": "intro to python",
    "top_k": 5
  }'
```

Vector search:
```bash
curl -X POST http://127.0.0.1:8080/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "vector",
    "query": [0.1, 0.2, 0.3],
    "top_k": 5
  }'
```

## Project Layout
- `app.py`: FastAPI entrypoint and routes.
- `functions/`: core pipeline logic for index lifecycle, embedding, batch/streaming updates, and search.
- `api_docs/`: API design and spec docs.
- `tests/`: unit and integration tests.
- `__example_streaming_datapoints/`: sample streaming payloads.

## Tests
```bash
pytest
```

## Related Docs
- `api_docs/API_spec.md`
- `api_docs/API_DESIGN.md`
