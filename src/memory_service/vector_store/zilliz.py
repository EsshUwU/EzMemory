"""
Zilliz Cloud / Milvus vector store implementation.

This implementation uses pymilvus.MilvusClient and the "quick setup" collection
API with dynamic fields enabled. We store:
- Primary key: string `id`
- Vector field: `embedding`
- Dynamic JSON payload: `payload` (goes into the $meta field internally)
"""

import uuid
from typing import List, Dict, Any, Optional

from pymilvus import MilvusClient

from .base import VectorStore


class ZillizVectorStore(VectorStore):
    """Zilliz Cloud (Milvus) vector store implementation."""

    DISTANCE_MAP = {
        "cosine": "COSINE",
        "euclid": "L2",
        "euclidean": "L2",
        "dot": "IP",
        "ip": "IP",
    }

    def __init__(self, uri: str, token: str, **kwargs):
        """
        Initialize Zilliz / Milvus client.

        Args:
            uri: Zilliz Cloud URI, e.g. https://...zillizcloud.com:19530
            token: Zilliz token, e.g. "user:password" or API key
            **kwargs: Additional MilvusClient parameters
        """
        if not uri:
            raise ValueError("Zilliz/Milvus URI is required")
        if not token:
            raise ValueError("Zilliz/Milvus token is required")

        self.client = MilvusClient(uri=uri, token=token, **kwargs)

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine",
        **kwargs,
    ) -> None:
        """
        Create a new collection in Zilliz / Milvus.

        We use the quick-setup API with:
        - string primary key field `id`
        - vector field `embedding`
        - dynamic fields enabled (payload stored as `payload` key)
        """
        metric = self.DISTANCE_MAP.get(distance_metric.lower(), "COSINE")

        # Use quick-setup API; dynamic fields are enabled by default.
        # max_length controls VARCHAR length for string IDs.
        self.client.create_collection(
            collection_name=collection_name,
            dimension=vector_size,
            primary_field_name="id",
            id_type="string",
            vector_field_name="embedding",
            metric_type=metric,
            auto_id=False,
            max_length=512,
            **kwargs,
        )

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            return bool(self.client.has_collection(collection_name=collection_name))
        except Exception:
            return False

    def insert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert vectors into Zilliz / Milvus."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        rows = []
        for point_id, vector, payload in zip(ids, vectors, payloads):
            rows.append(
                {
                    "id": point_id,
                    "embedding": vector,
                    # Dynamic field; stored in $meta but accessible as 'payload'
                    "payload": payload,
                }
            )

        self.client.insert(
            collection_name=collection_name,
            data=rows,
        )

        return ids

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Zilliz / Milvus.

        Note:
            - Filters are not yet translated; they are ignored if provided.
            - Scores use the distance returned by Milvus for the configured metric.
        """
        # Default to COSINE metric; must match collection index metric.
        search_params = {"metric_type": "COSINE", "params": {}}

        results = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="embedding",
            limit=limit,
            output_fields=["payload"],
            search_params=search_params,
        )

        hits = results[0] if results else []

        formatted: List[Dict[str, Any]] = []
        for hit in hits:
            # Zilliz returns dicts like: {"id": ..., "distance": ..., "entity": {...}}
            hit_id = str(hit.get("id"))
            distance = hit.get("distance")
            entity = hit.get("entity") or {}
            payload = entity.get("payload", {})

            score = float(distance) if distance is not None else 0.0

            if score_threshold is not None and score < score_threshold:
                continue

            formatted.append(
                {
                    "id": hit_id,
                    "score": score,
                    "payload": payload,
                }
            )

        return formatted

    def delete(
        self,
        collection_name: str,
        ids: List[str],
    ) -> None:
        """Delete vectors from Zilliz / Milvus by IDs."""
        if not ids:
            return

        self.client.delete(
            collection_name=collection_name,
            ids=ids,
        )

    def get_all(
        self,
        collection_name: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all vectors from collection.

        Implemented via scalar query without filter and pagination.
        """
        kwargs: Dict[str, Any] = {}
        if limit is not None:
            kwargs["limit"] = limit
        if offset is not None:
            kwargs["offset"] = offset

        rows = self.client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["id", "payload"],
            **kwargs,
        )

        return [
            {
                "id": str(row.get("id")),
                "payload": row.get("payload") or {},
            }
            for row in rows
        ]

    def count(self, collection_name: str) -> int:
        """Count vectors in collection."""
        stats = self.client.get_collection_stats(collection_name=collection_name)
        return int(stats.get("row_count", 0) or 0)

    def delete_all(self, collection_name: str) -> None:
        """
        Delete all vectors from collection.

        Uses a filter expression that matches all string IDs.
        """
        # Match all non-empty string IDs
        self.client.delete(
            collection_name=collection_name,
            filter='id like "%"',
        )

