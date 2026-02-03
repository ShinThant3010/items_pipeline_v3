# Data Flow Diagram (Text)

```
[BigQuery]
   |
   | 1) Weekly or manual query (custom where clause)
   v
[Transform & Validation]
   |
   | 2) Build canonical item JSON
   v
[Embedding Service]
   |\
   | \ 3a) Dense embeddings (dot product / L2)
   |  \
   |   3b) Sparse embeddings (BM25)
   v
[Merge + Package]
   |
   | 4) Write batch JSON to GCS
   v
[GCS Staging]
   |
   | 5) Vertex AI index batch update
   v
[Vertex AI Vector Search Index]
   |
   | 6) Deploy / Update on Endpoint
   v
[Vector Search Endpoint]

[Batch Orchestrator]
   |
   | 7) Logs status to Cloud SQL
   v
[Cloud SQL Log Table]
```

## Step Details
1) **BigQuery extract**: source data filtered by weekly update window
2) **Transform**: schema validation, metadata enrichment
3) **Embedding**: generate dense and sparse vectors
4) **Package**: write item JSON files to GCS (sharded)
5) **Index update**: Vertex AI batch CRUD ingestion
6) **Deploy**: index or update deployed index on endpoint
7) **Audit**: log run metadata and metrics in Cloud SQL

## Data Artifacts
- `gs://bucket/path/part-*.json`
- Cloud SQL table: `vector_index_update_log`

## Error Handling
- Invalid records are collected into a reject file in GCS
- Embedding failures are retried and logged
