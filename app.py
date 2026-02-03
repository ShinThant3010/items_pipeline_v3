import time

from fastapi import FastAPI, Request, HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from functions.utils.config import load_config
from functions.core.vertex_index import create_index, create_endpoint, deploy_index
from functions.core.search import search_index
from functions.core.index_updates import (
    embed_data,
    batch_update,
    streaming_update,
    streaming_delete,
)

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
    shard_size: str = Field(default="SHARD_SIZE_SMALL")
    distance_measure_type: str = Field(default="DOT_PRODUCT")
    feature_norm_type: str = Field(default="UNIT_L2_NORM")
    index_update_method: str = Field(default="STREAM_UPDATE")
    approximate_neighbors_count: int = Field(default=150)
    leaf_node_embedding_count: int = Field(default=1000)
    leaf_nodes_to_search_percent: int = Field(default=5)


class EmbedDataRequest(BaseModel):
    bigquery_table: str
    where: str
    gcs_output_prefix: str
    dimension: int | None = None


class StreamingDatapoint(BaseModel):
    id: str
    embedding: list[float]
    restricts: list[dict] | None = None
    numeric_restricts: list[dict] | None = None
    embedding_metadata: dict | None = None


class StreamingUpdateRequest(BaseModel):
    index_id: str | None = None
    datapoints_source: str = "api"
    datapoints: list[StreamingDatapoint] | None = None
    datapoints_gcs_prefix: str | None = None


class StreamingDeleteRequest(BaseModel):
    index_id: str | None = None
    datapoint_ids: list[str]


class BatchUpdateRequest(BaseModel):
    index_id: str | None = None
    contents_delta_uri: str
    is_complete_overwrite: bool = False


class CreateEndpointRequest(BaseModel):
    display_name: str
    description: str | None = None
    public_endpoint_enabled: bool = True


class DeployIndexRequest(BaseModel):
    endpoint_id: str
    index_id: str
    deployed_index_id: str
    machine_type: str = "e2-standard-2"
    min_replica_count: int = 1
    max_replica_count: int = 1


class SearchRequest(BaseModel):
    endpoint_id: str | None = None
    deployed_index_id: str | None = None
    query: str | list[float]
    query_type: str = "vector"
    top_k: int = 10
    metadata_gcs_prefix: str | None = None
    restricts: list[dict] | None = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/v1/index/create")
def api_create_index(req: CreateIndexRequest):
    return create_index(config, req.model_dump())


@app.post("/v1/embed_data")
def api_embed_data(req: EmbedDataRequest):
    return embed_data(config, req.model_dump())


@app.post("/v1/streaming/update")
def api_streaming_update(req: StreamingUpdateRequest):
    return streaming_update(config, req.model_dump())


@app.post("/v1/streaming/delete")
def api_streaming_delete(req: StreamingDeleteRequest):
    return streaming_delete(config, req.model_dump())


@app.post("/v1/batch/updates")
def api_batch_update(req: BatchUpdateRequest):
    return batch_update(config, req.model_dump())


@app.post("/v1/endpoint/create")
def api_create_endpoint(req: CreateEndpointRequest):
    return create_endpoint(config, req.model_dump())


@app.post("/v1/endpoint/deploy")
def api_deploy_index(req: DeployIndexRequest):
    return deploy_index(config, req.model_dump())


@app.post("/v1/search")
def api_search(req: SearchRequest):
    return search_index(config, req.model_dump())
