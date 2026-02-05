"""
Memory retrieval - search and query logic.
"""

from typing import List, Optional, Dict, Any
from .schemas import Memory, SearchResult
from ..vector_store.base import VectorStore


class MemoryRetrieval:
    """Handles memory retrieval and search."""
    
    def __init__(self, vector_store: VectorStore, collection_name: str):
        """
        Initialize retrieval.
        
        Args:
            vector_store: Vector store instance
            collection_name: Name of collection to use
        """
        self.vector_store = vector_store
        self.collection_name = collection_name
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar memories.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        results = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            filters=filters,
        )
        
        search_results = []
        for result in results:
            memory = Memory.from_payload(result["id"], result["payload"])
            search_results.append(
                SearchResult(
                    memory=memory,
                    score=result["score"],
                )
            )
        
        return search_results
    
    def get_by_tags(
        self,
        tags: List[str],
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """
        Get memories by tags.
        
        Args:
            tags: List of tags to filter by
            limit: Maximum number of results
            
        Returns:
            List of memories
        
        Note:
            This implementation fetches all memories and filters in-memory,
            which is not scalable for large datasets. For production use,
            consider implementing Qdrant's payload filtering with tag indexing.
            See: https://qdrant.tech/documentation/concepts/filtering/
        """
        if not tags:
            return []
        
        # WARNING: This fetches ALL memories - not scalable!
        # For better performance, use Qdrant's scroll with filters
        all_memories = []
        results = self.vector_store.get_all(
            collection_name=self.collection_name,
            limit=limit or 1000,  # Cap at 1000 to prevent memory issues
        )
        
        for result in results:
            memory = Memory.from_payload(result["id"], result["payload"])
            # Check if any of the requested tags are in the memory's tags
            if any(tag in memory.metadata.tags for tag in tags):
                all_memories.append(memory)
                if limit and len(all_memories) >= limit:
                    break
        
        return all_memories
