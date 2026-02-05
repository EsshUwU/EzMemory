import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from .base import VectorStore


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation."""
    
    DISTANCE_MAP = {
        "cosine": Distance.COSINE,
        "euclid": Distance.EUCLID,
        "dot": Distance.DOT,
        "manhattan": Distance.MANHATTAN,
    }
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
    ):
        """
        Initialize Qdrant client.
        
        Args:
            host: Host address (for local/self-hosted)
            port: Port number (for local/self-hosted)
            url: Full URL (for Qdrant Cloud)
            api_key: API key (for Qdrant Cloud)
            prefer_grpc: Use gRPC for better performance
        """
        if url:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
            )
        else:
            self.client = QdrantClient(
                host=host or "localhost",
                port=port or 6333,
                prefer_grpc=prefer_grpc,
            )
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine",
        **kwargs
    ) -> None:
        """Create a new collection in Qdrant."""
        distance = self.DISTANCE_MAP.get(distance_metric.lower(), Distance.COSINE)
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
            ),
            **kwargs
        )
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception:
            return False
    
    def insert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert vectors into Qdrant."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        points = [
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            for point_id, vector, payload in zip(ids, vectors, payloads)
        ]
        
        self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
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
        """Search for similar vectors in Qdrant."""
        query_filter = None
        if filters:
            query_filter = self._build_filter(filters)
        
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
        
        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload,
            }
            for point in results.points
        ]
    
    def delete(
        self,
        collection_name: str,
        ids: List[str],
    ) -> None:
        """Delete vectors from Qdrant."""
        self.client.delete(
            collection_name=collection_name,
            points_selector=ids,
            wait=True,
        )
    
    def get_all(
        self,
        collection_name: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get all vectors from collection."""
        scroll_result = self.client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        
        points, _ = scroll_result
        
        return [
            {
                "id": str(point.id),
                "payload": point.payload,
            }
            for point in points
        ]
    
    def count(self, collection_name: str) -> int:
        """Count vectors in collection."""
        collection_info = self.client.get_collection(collection_name)
        return collection_info.points_count
    
    def delete_all(self, collection_name: str) -> None:
        """Delete all vectors from Qdrant collection."""
        # Delete all points by using a filter that matches everything
        # In Qdrant, we can delete the entire collection and recreate it
        # Or use delete with points_selector set to all points
        from qdrant_client.models import FilterSelector
        
        # Get collection info to preserve settings
        collection_info = self.client.get_collection(collection_name)
        
        # Delete all points using a universal filter (matches all)
        # Alternatively, we could recreate the collection, but deleting is safer
        self.client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[]  # Empty filter matches all points
                )
            ),
            wait=True,
        )
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dict."""
        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )
        
        return Filter(must=conditions) if conditions else None
