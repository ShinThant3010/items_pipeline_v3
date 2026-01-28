from __future__ import annotations

import re
from typing import Any

from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import matching_engine_index_config
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from functions.utils.config import AppConfig, update_resource_names

try:
    from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
        HybridQuery,
        SparseEmbedding,
    )
    _HYBRID_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on SDK version
    HybridQuery = None
    SparseEmbedding = None
    _HYBRID_AVAILABLE = False

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _distance_measure(distance: str) -> matching_engine_index_config.DistanceMeasureType:
    """Map string distance names to Vertex distance enums."""
    mapping = {
        "DOT_PRODUCT": matching_engine_index_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE,
        "COSINE": matching_engine_index_config.DistanceMeasureType.COSINE_DISTANCE,
        "L2_NORM": matching_engine_index_config.DistanceMeasureType.SQUARED_L2_DISTANCE,
    }
    return mapping.get(distance, matching_engine_index_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE)


def _feature_norm_type(value: str | None) -> matching_engine_index_config.FeatureNormType | None:
    """Map string feature norm types to Vertex enums."""
    if not value:
        return None
    mapping = {
        "UNIT_L2_NORM": matching_engine_index_config.FeatureNormType.UNIT_L2_NORM,
        "NONE": matching_engine_index_config.FeatureNormType.FEATURE_NORM_TYPE_UNSPECIFIED,
    }
    return mapping.get(value, matching_engine_index_config.FeatureNormType.FEATURE_NORM_TYPE_UNSPECIFIED)


def _tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text into alphanumeric terms."""
    return _TOKEN_RE.findall(text.lower())


def _build_sparse_query(config: AppConfig, text: str) -> SparseEmbedding | dict[str, list[float | int]]:
    """Build a sparse query vector for hybrid search."""
    tokens = _tokenize(text)
    if not tokens:
        if _HYBRID_AVAILABLE:
            return SparseEmbedding(dimensions=[], values=[])
        return {"dimensions": [], "values": []}

    term_freq: dict[str, int] = {}
    for term in tokens:
        term_freq[term] = term_freq.get(term, 0) + 1

    bucket_scores: dict[int, float] = {}
    for term, tf in term_freq.items():
        idx = hash(term) % max(config.embedding_sparse_dimensions, 1)
        bucket_scores[idx] = bucket_scores.get(idx, 0.0) + float(tf)

    dimensions = list(bucket_scores.keys())
    values = [bucket_scores[idx] for idx in dimensions]
    if _HYBRID_AVAILABLE:
        return SparseEmbedding(dimensions=dimensions, values=values)
    return {"dimensions": dimensions, "values": values}


def create_index(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a Vertex AI Vector Search index."""
    aiplatform.init(project=config.project_id, location=config.region)

    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=payload["display_name"],
        contents_delta_uri=payload.get("contents_delta_uri"),
        dimensions=payload["dimensions"],
        approximate_neighbors_count=payload.get("approximate_neighbors_count", 150),
        leaf_node_embedding_count=payload.get("leaf_node_embedding_count", 500),
        leaf_nodes_to_search_percent=payload.get("leaf_nodes_to_search_percent", 7),
        description=payload.get("description"),
        index_update_method=payload.get("index_update_method", "BATCH_UPDATE"),
        distance_measure_type=_distance_measure(payload.get("distance_measure_type", "DOT_PRODUCT")),
        shard_size=payload.get("shard_size"),
        feature_norm_type=_feature_norm_type(payload.get("feature_norm_type")),
    )

    return {
        "index_id": index.resource_name,
        "status": "CREATED",
        "request": payload,
    }


def create_endpoint(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a Vertex AI Vector Search endpoint."""
    aiplatform.init(project=config.project_id, location=config.region)

    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=payload["display_name"],
        description=payload.get("description"),
        public_endpoint_enabled=payload.get("public_endpoint_enabled", True),
    )

    return {
        "endpoint_id": endpoint.resource_name,
        "status": "CREATED",
        "request": payload,
    }


def deploy_index(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Deploy an index to a Vector Search endpoint."""
    endpoint_id = payload.get("endpoint_id") or config.endpoint_id
    index_id = payload.get("index_id") or config.index_id
    deployed_index_id = payload.get("deployed_index_id") or config.deployed_index_id

    if not endpoint_id or not index_id or not deployed_index_id:
        raise ValueError("endpoint_id, index_id, and deployed_index_id are required")

    aiplatform.init(project=config.project_id, location=config.region)

    endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id)
    index = aiplatform.MatchingEngineIndex(index_name=index_id)

    endpoint.deploy_index(
        index=index,
        deployed_index_id=deployed_index_id,
        machine_type=payload.get("machine_type", "e2-standard-2"),
        min_replica_count=payload.get("min_replica_count", 1),
        max_replica_count=payload.get("max_replica_count", 1),
    )

    update_resource_names(
        index_id=index_id,
        endpoint_id=endpoint_id,
        deployed_index_id=deployed_index_id,
    )

    return {
        "deployed_index_id": deployed_index_id,
        "endpoint_id": endpoint_id,
        "status": "DEPLOYED",
        "request": payload,
    }


def _extract_neighbor(neighbor: Any) -> dict[str, Any]:
    """Normalize a neighbor result into a response dict."""
    datapoint = getattr(neighbor, "datapoint", None)
    if datapoint is not None:
        neighbor_id = getattr(datapoint, "datapoint_id", None) or getattr(datapoint, "id", None)
        metadata = getattr(datapoint, "embedding_metadata", None) or getattr(datapoint, "metadata", None)
    else:
        neighbor_id = getattr(neighbor, "id", None)
        metadata = None

    score = getattr(neighbor, "distance", None)
    if score is None:
        score = getattr(neighbor, "score", None)

    return {
        "id": neighbor_id,
        "score": score,
        "metadata": metadata,
    }


def search_index(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Search the deployed index. Supports BM25 toggle via use_bm25."""
    endpoint_id = payload.get("endpoint_id") or config.endpoint_id
    deployed_index_id = payload.get("deployed_index_id") or config.deployed_index_id
    if not endpoint_id or not deployed_index_id:
        raise ValueError("endpoint_id and deployed_index_id are required")

    query_text = payload.get("query", "")
    top_k = int(payload.get("top_k", 10))
    use_bm25 = bool(payload.get("use_bm25", False))

    vertexai.init(project=config.project_id, location=config.region)
    model = TextEmbeddingModel.from_pretrained(config.embedding_model_name)
    embedding = model.get_embeddings(
        [TextEmbeddingInput(text=query_text, task_type="RETRIEVAL_QUERY")],
        output_dimensionality=config.embedding_output_dimensionality,
    )[0]

    aiplatform.init(project=config.project_id, location=config.region)
    endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id)

    if use_bm25:
        if not _HYBRID_AVAILABLE:
            raise NotImplementedError(
                "Hybrid search is not supported by the installed google-cloud-aiplatform SDK. "
                "Upgrade google-cloud-aiplatform to a version that includes HybridQuery/SparseEmbedding."
            )
        sparse = _build_sparse_query(config, query_text)
        query = HybridQuery(dense_embedding=embedding.values, sparse_embedding=sparse)
        neighbors = endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[query],
            num_neighbors=top_k,
            return_full_datapoint=True,
        )
    else:
        neighbors = endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[embedding.values],
            num_neighbors=top_k,
            return_full_datapoint=True,
        )

    results = []
    if neighbors:
        results = [_extract_neighbor(n) for n in neighbors[0]]

    return {
        "query": query_text,
        "use_bm25": use_bm25,
        "results": results,
    }
