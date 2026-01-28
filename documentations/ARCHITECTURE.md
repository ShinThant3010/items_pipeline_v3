# Vertex AI Vector Search Pipeline Architecture

## Overview
This pipeline is a backend API that orchestrates Vertex AI Vector Search (index + endpoint lifecycle) and weekly batch CRUD updates. Source data lives in BigQuery, is transformed and embedded, and written to GCS as JSON for ingestion. Update logs are stored in Cloud SQL.

## Goals
- Provide API endpoints to create index, create endpoint, deploy index to endpoint
- Support weekly batch CRUD updates and manual/ad‑hoc updates
- Enable hybrid search: dense + sparse vectors, dot product + L2 norm, BM25
- Maintain update audit log in Cloud SQL

## Non‑Goals (initial)
- Real‑time streaming updates
- Multi‑tenant authorization and billing
- UI or manual operator console

## High‑Level Components
- **API Service**
  - Exposes HTTP endpoints for index/endpoint lifecycle and batch CRUD
  - Orchestrates pipeline runs and validates inputs
- **Batch Orchestrator**
  - Schedules weekly update job
  - Handles retries, status updates, and idempotency
- **Data Extraction & Transform**
  - Queries BigQuery for source data
  - Builds canonical item JSON
- **Embedding Service**
  - Generates dense and sparse embeddings
  - Supports hybrid search setup (dense+BM25)
- **GCS Staging**
  - Stores JSON files for batch ingestion
- **Vertex AI Vector Search**
  - Index, Endpoint, Deployed Index resources
  - Batch CRUD operations via index update
- **Cloud SQL Log Store**
  - Small update log table for audit and debugging

## Deployment Model
- The API service and batch orchestrator can run as Cloud Run services
- Scheduled weekly update via Cloud Scheduler triggering the batch orchestrator
- Service accounts with least‑privilege access to BigQuery, GCS, Vertex AI, and Cloud SQL

## Data Contracts
- **Item JSON**
  - Required identifiers: `item_id`, `namespace` (or index namespace)
  - Payload: text fields for BM25, metadata fields, and embedding vectors
  - Sparse vector format compatible with Vertex AI hybrid search

## Security & Access
- IAM‑based access control
- Use VPC‑connector or private IP for Cloud SQL
- GCS buckets scoped to pipeline

## Reliability & Idempotency
- Each batch update has a unique `run_id`
- Updates are idempotent by `item_id`
- Logs contain status, counts, timestamps, and error summaries

## Observability
- Structured logs from API and batch jobs
- Cloud SQL log table stores:
  - run_id, start_ts, end_ts, status, counts, errors

## Risks & Mitigations
- **Large batch size** → chunked GCS JSON, per‑chunk ingestion
- **Embedding failures** → retry + dead‑letter record list
- **Schema drift** → validation in transform step, reject invalid records

## Versioning
- Index schema changes require a new index + endpoint or rolling rebuild strategy
