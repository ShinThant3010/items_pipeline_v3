import time

from fastapi import FastAPI, Request, HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from functions.utils.config import load_config
from functions.core.vertex_index import (
    create_index,
    create_endpoint,
    deploy_index,
    search_index,
)
from functions.core.embedding import embed_data, batch_update, streaming_update, streaming_delete

load_dotenv()
app = FastAPI(title="Vertex AI Vector Search Pipeline API", version="1.0.0")


@app.middleware("http")
async def add_response_time_header(request: Request, call_next):
    start = time.monotonic()
    request.state.start_time = start
    response = await call_next(request)
    elapsed = time.monotonic() - start
    response.headers["x-response-time-seconds"] = f"{elapsed:.6f}"
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler_with_time(request: Request, exc: HTTPException):
    response = await http_exception_handler(request, exc)
    start = getattr(request.state, "start_time", time.monotonic())
    elapsed = time.monotonic() - start
    response.headers["x-response-time-seconds"] = f"{elapsed:.6f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    start = getattr(request.state, "start_time", time.monotonic())
    elapsed = time.monotonic() - start
    response = PlainTextResponse("Internal Server Error", status_code=500)
    response.headers["x-response-time-seconds"] = f"{elapsed:.6f}"
    return response
config = load_config()


class CreateIndexRequest(BaseModel):
    display_name: str
    description: str | None = None
    dimensions: int
    distance_measure_type: str = Field(default="DOT_PRODUCT")
    index_update_method: str = Field(default="BATCH_UPDATE")
    shard_size: str = Field(default="SHARD_SIZE_SMALL")
    sparse_dimensions: int | None = None
    hybrid_config: dict | None = None


class CreateEndpointRequest(BaseModel):
    display_name: str
    description: str | None = None


class DeployIndexRequest(BaseModel):
    index_id: str
    deployed_index_id: str
    min_replica_count: int = 1
    max_replica_count: int = 1
    enable_hybrid_search: bool = True


class EmbedDataRequest(BaseModel):
    run_id: str
    bigquery_table: str
    where: str
    gcs_output_prefix: str
    use_bm25: bool = True


class BatchUpdateRequest(BaseModel):
    run_id: str
    update_type: str = Field(default="SCHEDULED")
    bigquery_table: str
    where: str
    gcs_output_prefix: str
    index_id: str | None = None
    mode: str = Field(default="UPSERT")
    delete_policy: str = Field(default="SOFT_DELETE")
    use_preembedded: bool = False


class SearchRequest(BaseModel):
    endpoint_id: str | None = None
    deployed_index_id: str | None = None
    query: str
    top_k: int = 10
    use_bm25: bool = False


class StreamingDatapoint(BaseModel):
    id: str
    embedding: list[float]
    sparse_embedding: dict | None = None
    restricts: list[dict] | None = None
    numeric_restricts: list[dict] | None = None
    embedding_metadata: dict | None = None
    crowding_tag: str | None = None


class StreamingUpdateRequest(BaseModel):
    index_id: str | None = None
    datapoints: list[StreamingDatapoint]


class StreamingDeleteRequest(BaseModel):
    index_id: str | None = None
    datapoint_ids: list[str]


@app.post("/v1/indexes")
def api_create_index(req: CreateIndexRequest):
    return create_index(config, req.model_dump())


@app.post("/v1/endpoints")
def api_create_endpoint(req: CreateEndpointRequest):
    return create_endpoint(config, req.model_dump())


@app.post("/v1/endpoints/{endpoint_id}/deploy")
def api_deploy_index(endpoint_id: str, req: DeployIndexRequest):
    payload = req.model_dump()
    payload["endpoint_id"] = endpoint_id
    return deploy_index(config, payload)


@app.post("/v1/embed_data")
def api_embed_data(req: EmbedDataRequest):
    return embed_data(config, req.model_dump())


@app.post("/v1/batch/updates")
def api_batch_update(req: BatchUpdateRequest):
    return batch_update(config, req.model_dump())


@app.post("/v1/search")
def api_search(req: SearchRequest):
    return search_index(config, req.model_dump())


@app.post("/v1/streaming/update")
def api_streaming_update(req: StreamingUpdateRequest):
    return streaming_update(config, req.model_dump())


@app.post("/v1/streaming/delete")
def api_streaming_delete(req: StreamingDeleteRequest):
    return streaming_delete(config, req.model_dump())
