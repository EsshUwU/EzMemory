"""
Memory storage - persistence layer.
"""

from typing import List, Optional
from .schemas import Memory
from ..vector_store.base import VectorStore
from ..utils.hashing import generate_content_hash


class MemoryStorage:
    """Handles memory persistence in vector store."""
    
    def __init__(self, vector_store: VectorStore, collection_name: str, check_duplicates: bool = False):
        """
        Initialize storage.
        
        Args:
            vector_store: Vector store instance
            collection_name: Name of collection to use
            check_duplicates: Whether to check for duplicate content before inserting
        """
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.check_duplicates = check_duplicates
    
    def exists(self, memory_id: str) -> bool:
        """
        Check if a memory exists by ID.
        
        Args:
            memory_id: Memory ID to check
            
        Returns:
            True if memory exists, False otherwise
        """
        try:
            # Try to retrieve the memory - if it doesn't exist, it will be in empty results
            results = self.vector_store.get_all(
                collection_name=self.collection_name,
                limit=1,
            )
            # This is a simplified check - in production, use Qdrant's retrieve by ID
            return any(r["id"] == memory_id for r in results)
        except Exception:
            return False
    
    def store(self, memory: Memory, skip_if_exists: Optional[bool] = None) -> str:
        """
        Store a single memory.
        
        Args:
            memory: Memory to store
            skip_if_exists: Whether to skip if memory already exists (uses self.check_duplicates if None)
            
        Returns:
            Memory ID
        
        Raises:
            ValueError: If memory has no embedding or content
        """
        if not memory.content or not memory.content.strip():
            raise ValueError("Memory content cannot be empty")
        
        if not memory.embedding:
            raise ValueError("Memory must have an embedding before storage")
        
        if memory.id is None:
            memory.id = generate_content_hash(memory.content)
        
        # Check for duplicates if enabled
        if (skip_if_exists if skip_if_exists is not None else self.check_duplicates):
            if self.exists(memory.id):
                return memory.id  # Return existing ID without inserting
        
        ids = self.vector_store.insert(
            collection_name=self.collection_name,
            vectors=[memory.embedding],
            payloads=[memory.to_payload()],
            ids=[memory.id],
        )
        
        return ids[0]
    
    def store_batch(self, memories: List[Memory]) -> List[str]:
        """
        Store multiple memories.
        
        Args:
            memories: List of memories to store
            
        Returns:
            List of memory IDs
        
        Raises:
            ValueError: If any memory is invalid
        """
        if not memories:
            return []
        
        # Validate all memories
        for i, memory in enumerate(memories):
            if not memory.content or not memory.content.strip():
                raise ValueError(f"Memory at index {i} has empty content")
            if not memory.embedding:
                raise ValueError(f"Memory at index {i} has no embedding")
        
        # Generate IDs for memories that don't have one
        for memory in memories:
            if memory.id is None:
                memory.id = generate_content_hash(memory.content)
        
        ids = self.vector_store.insert(
            collection_name=self.collection_name,
            vectors=[m.embedding for m in memories],
            payloads=[m.to_payload() for m in memories],
            ids=[m.id for m in memories],
        )
        
        return ids
    
    def delete(self, memory_ids: List[str]) -> None:
        """
        Delete memories by ID.
        
        Args:
            memory_ids: List of memory IDs to delete
        """
        self.vector_store.delete(
            collection_name=self.collection_name,
            ids=memory_ids,
        )
    
    def get_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Memory]:
        """
        Get all memories.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of memories
        """
        results = self.vector_store.get_all(
            collection_name=self.collection_name,
            limit=limit,
            offset=offset,
        )
        
        return [
            Memory.from_payload(result["id"], result["payload"])
            for result in results
        ]
    
    def count(self) -> int:
        """
        Count total memories.
        
        Returns:
            Number of memories
        """
        return self.vector_store.count(self.collection_name)
    
    def delete_all(self) -> None:
        """
        Delete all memories from the collection.
        """
        self.vector_store.delete_all(self.collection_name)
