from __future__ import annotations

from typing import Any

from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import matching_engine_index_config

from functions.utils.config import AppConfig, update_resource_names


def _distance_measure(distance: str) -> matching_engine_index_config.DistanceMeasureType:
    """Map string distance names to Vertex distance enums."""
    mapping = {
        "DOT_PRODUCT": matching_engine_index_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE,
        "COSINE": matching_engine_index_config.DistanceMeasureType.COSINE_DISTANCE,
        "L2_NORM": matching_engine_index_config.DistanceMeasureType.SQUARED_L2_DISTANCE,
    }
    return mapping.get(distance, matching_engine_index_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE)


def _feature_norm_type(value: str | None) -> matching_engine_index_config.FeatureNormType:
    """Map string feature norm types to Vertex enums."""
    mapping = {
        "UNIT_L2_NORM": matching_engine_index_config.FeatureNormType.UNIT_L2_NORM,
        "NONE": matching_engine_index_config.FeatureNormType.NONE,
    }
    return mapping.get(value, matching_engine_index_config.FeatureNormType.NONE)


def create_index(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a Vertex AI Vector Search index."""
    aiplatform.init(project=config.project_id, location=config.region)

    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=payload["display_name"],
        description=payload.get("description"),
        dimensions=payload["dimensions"],
        shard_size=payload.get("shard_size", config.index_defaults.get("shard_size", "SHARD_SIZE_SMALL")),
        distance_measure_type=_distance_measure(
            payload.get(
                "distance_measure_type",
                config.index_defaults.get("distance_measure_type", "DOT_PRODUCT"),
            )
        ),
        feature_norm_type=_feature_norm_type(
            payload.get(
                "feature_norm_type",
                config.index_defaults.get("feature_norm_type", "UNIT_L2_NORM"),
            )
        ),
        index_update_method=payload.get(
            "index_update_method",
            config.index_defaults.get("index_update_method", "STREAM_UPDATE"),
        ),
        approximate_neighbors_count=payload.get(
            "approximate_neighbors_count",
            config.index_defaults.get("approximate_neighbors_count", 150),
        ),
        leaf_node_embedding_count=payload.get(
            "leaf_node_embedding_count",
            config.index_defaults.get("leaf_node_embedding_count", 1000),
        ),
        leaf_nodes_to_search_percent=payload.get(
            "leaf_nodes_to_search_percent",
            config.index_defaults.get("leaf_nodes_to_search_percent", 5),
        ),
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
