# Vertex AI Vector Search Pipeline

Backend pipeline for managing Vertex AI Vector Search indexes and endpoints, plus batch and streaming updates. It queries data from BigQuery, generates embeddings with Gemini, writes them to GCS, and updates the index for search. Supports hybrid search (dense + sparse/BM25).

## Status
This project is **work in progress** and in the **exploration stage**. Expect breaking changes, incomplete features, and unstable APIs.
