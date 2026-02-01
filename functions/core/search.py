from __future__ import annotations

import re
from typing import Any

from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from functions.utils.config import AppConfig

try:
    from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import HybridQuery
    _HYBRID_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on SDK version
    HybridQuery = None
    _HYBRID_AVAILABLE = False

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text into alphanumeric terms."""
    return _TOKEN_RE.findall(text.lower())


def _build_sparse_query(config: AppConfig, text: str) -> tuple[list[int], list[float]]:
    """Build a sparse keyword query vector using hashed term buckets."""
    tokens = _tokenize(text)
    if not tokens:
        return [], []

    term_freq: dict[str, int] = {}
    for term in tokens:
        term_freq[term] = term_freq.get(term, 0) + 1

    bucket_scores: dict[int, float] = {}
    for term, tf in term_freq.items():
        idx = hash(term) % max(config.embedding_sparse_dimensions, 1)
        bucket_scores[idx] = bucket_scores.get(idx, 0.0) + float(tf)

    dimensions = list(bucket_scores.keys())
    values = [bucket_scores[idx] for idx in dimensions]
    return dimensions, values


def _extract_neighbor(neighbor: Any) -> dict[str, Any]:
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
    """Search the deployed index with dense or hybrid keyword queries."""
    endpoint_id = payload.get("endpoint_id") or config.endpoint_id
    deployed_index_id = payload.get("deployed_index_id") or config.deployed_index_id
    if not endpoint_id or not deployed_index_id:
        raise ValueError("endpoint_id and deployed_index_id are required")

    query_text = payload.get("query", "")
    top_k = int(payload.get("top_k", 10))
    use_bm25 = bool(payload.get("use_bm25", False))
    rrf_alpha = payload.get("rrf_alpha")

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
                "Upgrade google-cloud-aiplatform to a version that includes HybridQuery."
            )

        dimensions, values = _build_sparse_query(config, query_text)
        query = HybridQuery(
            dense_embedding=embedding.values,
            sparse_embedding_dimensions=dimensions,
            sparse_embedding_values=values,
            rrf_ranking_alpha=rrf_alpha,
        )
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

