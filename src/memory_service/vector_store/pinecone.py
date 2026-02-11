import uuid
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from .base import VectorStore


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation."""
    
    DISTANCE_MAP = {
        "cosine": "cosine",
        "euclid": "euclidean",
        "euclidean": "euclidean",
        "dot": "dotproduct",
        "dotproduct": "dotproduct",
    }
    
    def __init__(
        self,
        api_key: str,
        cloud: str = "aws",
        region: str = "us-east-1",
        **kwargs
    ):
        """
        Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region
            **kwargs: Additional Pinecone client parameters
        """
        self.client = Pinecone(api_key=api_key, **kwargs)
        self.cloud = cloud
        self.region = region
        self._indexes = {}  # Cache for index connections
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine",
        **kwargs
    ) -> None:
        """
        Create a new index in Pinecone.
        
        Note: In Pinecone, collections are called indexes.
        """
        metric = self.DISTANCE_MAP.get(distance_metric.lower(), "cosine")
        
        # Use ServerlessSpec by default for cost-effectiveness
        spec = kwargs.get("spec") or ServerlessSpec(
            cloud=self.cloud,
            region=self.region
        )
        
        self.client.create_index(
            name=collection_name,
            dimension=vector_size,
            metric=metric,
            spec=spec,
        )
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if an index exists."""
        try:
            indexes = self.client.list_indexes()
            return any(idx.name == collection_name for idx in indexes.indexes)
        except Exception:
            return False
    
    def _get_index(self, collection_name: str):
        """Get or create connection to index."""
        if collection_name not in self._indexes:
            self._indexes[collection_name] = self.client.Index(collection_name)
        return self._indexes[collection_name]
    
    def insert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert vectors into Pinecone index."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        index = self._get_index(collection_name)
        
        # Prepare vectors in Pinecone format
        # Note: Pinecone requires flat metadata (no nested dicts, no null values)
        vectors_to_upsert = [
            {
                "id": point_id,
                "values": vector,
                "metadata": self._flatten_metadata(payload),
            }
            for point_id, vector, payload in zip(ids, vectors, payloads)
        ]
        
        # Batch upsert (Pinecone handles batching internally)
        index.upsert(vectors=vectors_to_upsert)
        
        return ids
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone index."""
        index = self._get_index(collection_name)
        
        # Build filter in Pinecone format
        pinecone_filter = self._build_filter(filters) if filters else None
        
        # Query index
        results = index.query(
            vector=query_vector,
            top_k=limit,
            filter=pinecone_filter,
            include_metadata=True,
            include_values=False,
        )
        
        # Filter by score threshold if specified
        matches = results.matches
        if score_threshold is not None:
            matches = [m for m in matches if m.score >= score_threshold]
        
        return [
            {
                "id": match.id,
                "score": match.score,
                "payload": self._unflatten_metadata(match.metadata),
            }
            for match in matches
        ]
    
    def delete(
        self,
        collection_name: str,
        ids: List[str],
    ) -> None:
        """Delete vectors from Pinecone index."""
        index = self._get_index(collection_name)
        index.delete(ids=ids)
    
    def get_all(
        self,
        collection_name: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all vectors from index using list_paginated + fetch.

        Lists vector IDs with pagination, then fetches metadata in batches.
        """
        index = self._get_index(collection_name)

        # Collect vector IDs using list_paginated
        all_ids: List[str] = []
        pagination_token = None

        while True:
            list_kwargs: Dict[str, Any] = {"prefix": "", "limit": 100}
            if pagination_token:
                list_kwargs["pagination_token"] = pagination_token

            response = index.list_paginated(**list_kwargs)

            if response.vectors:
                all_ids.extend([v.id for v in response.vectors])

            if response.pagination and response.pagination.next:
                pagination_token = response.pagination.next
            else:
                break

        # Apply offset and limit
        if offset:
            all_ids = all_ids[offset:]
        if limit:
            all_ids = all_ids[:limit]

        if not all_ids:
            return []

        # Fetch vectors with metadata in batches of 100
        results: List[Dict[str, Any]] = []
        for i in range(0, len(all_ids), 100):
            batch = all_ids[i : i + 100]
            fetch_response = index.fetch(ids=batch)

            for vec_id in batch:
                if vec_id in fetch_response.vectors:
                    vec = fetch_response.vectors[vec_id]
                    results.append(
                        {
                            "id": vec.id,
                            "payload": self._unflatten_metadata(
                                vec.metadata or {}
                            ),
                        }
                    )

        return results
    
    def count(self, collection_name: str) -> int:
        """Count vectors in index."""
        index = self._get_index(collection_name)
        stats = index.describe_index_stats()
        return stats.total_vector_count
    
    def _flatten_metadata(self, payload: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten nested dictionaries and filter null values for Pinecone.
        
        Pinecone metadata requirements:
        - No nested dictionaries
        - No null values
        - Values must be: string, number, boolean, or list of strings
        
        Example:
            {"metadata": {"tags": ["foo"], "confidence": 1.0}}
            becomes
            {"metadata.tags": ["foo"], "metadata.confidence": 1.0}
        """
        items = {}
        
        for key, value in payload.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if value is None:
                # Skip null values
                continue
            elif isinstance(value, dict):
                # Flatten nested dictionaries
                if value:  # Only process non-empty dicts
                    items.update(self._flatten_metadata(value, new_key, sep=sep))
            elif isinstance(value, list):
                # Keep lists as-is (Pinecone supports list of strings)
                items[new_key] = value
            elif isinstance(value, (str, int, float, bool)):
                # Keep primitive types
                items[new_key] = value
            else:
                # Convert other types to string
                items[new_key] = str(value)
        
        return items
    
    def _unflatten_metadata(self, flat_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
        """
        Unflatten metadata back to nested structure.
        
        Example:
            {"metadata.tags": ["foo"], "metadata.confidence": 1.0}
            becomes
            {"metadata": {"tags": ["foo"], "confidence": 1.0}}
        """
        result = {}
        
        for key, value in flat_dict.items():
            parts = key.split(sep)
            current = result
            
            # Navigate/create nested structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value at the leaf
            current[parts[-1]] = value
        
        return result
    
    def delete_all(self, collection_name: str) -> None:
        """Delete all vectors from Pinecone index."""
        index = self._get_index(collection_name)
        # Delete all vectors in the index (default namespace)
        index.delete(delete_all=True)
    
    def _build_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Pinecone filter from dict.
        
        Pinecone uses metadata filtering with operators like:
        - {"category": "tutorial"} for exact match
        - {"price": {"$gte": 100}} for comparison
        - {"$and": [...]} for logical operators
        """
        # Simple pass-through for now
        # Pinecone's filter format is similar to MongoDB
        return filters
