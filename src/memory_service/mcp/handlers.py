from typing import Optional, List
from ..memory import Memory, MemoryEncoder, MemoryStorage, MemoryRetrieval
from ..config.logging import get_logger

logger = get_logger(__name__)


class MemoryHandlers:
    """Handlers for memory operations."""
    
    def __init__(
        self,
        encoder: MemoryEncoder,
        storage: MemoryStorage,
        retrieval: MemoryRetrieval,
        search_limit: int = 5,
    ):
        """
        Initialize handlers.
        
        Args:
            encoder: Memory encoder instance
            storage: Memory storage instance
            retrieval: Memory retrieval instance
            search_limit: Default search limit
        """
        self.encoder = encoder
        self.storage = storage
        self.retrieval = retrieval
        self.search_limit = search_limit
    
    async def add_memory(self, content: str) -> dict:
        """
        Add a new memory.
        
        Args:
            content: Memory content
            
        Returns:
            Result dictionary
        """
        try:
            # Create memory
            memory = Memory(content=content)
            
            # Encode
            memory = self.encoder.encode(memory)
            
            # Store
            memory_id = self.storage.store(memory)
            
            logger.info(f"Added memory: {memory_id}")
            
            return {
                "status": "success",
                "message": "Memory added successfully",
                "memory_id": memory_id,
            }
        except Exception as e:
            logger.error(f"Error in add_memory: {e}")
            raise
    
    async def search_memory(self, query: str) -> dict:
        """
        Search for memories.
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        try:
            # Encode query
            query_vector = self.encoder.encode_query(query)
            
            # Search
            results = self.retrieval.search(
                query_vector=query_vector,
                limit=self.search_limit,
            )
            
            logger.info(f"Found {len(results)} memories for query: {query[:50]}...")
            
            return {
                "status": "success",
                "query": query,
                "count": len(results),
                "results": [result.to_dict() for result in results],
            }
        except Exception as e:
            logger.error(f"Error in search_memory: {e}")
            raise
    
    async def list_memory(
        self,
        limit: Optional[int] = 10,
        offset: Optional[int] = 0,
    ) -> dict:
        """
        List all memories.
        
        Args:
            limit: Maximum number of results
            offset: Number to skip
            
        Returns:
            List of memories
        """
        try:
            # Get memories
            memories = self.storage.get_all(limit=limit, offset=offset)
            total = self.storage.count()
            
            logger.info(f"Listed {len(memories)} memories (total: {total})")
            
            return {
                "status": "success",
                "count": len(memories),
                "total": total,
                "memories": [
                    {
                        "id": m.id,
                        "content": m.content,
                        "created_at": m.created_at.isoformat(),
                        "metadata": m.metadata.model_dump(),
                    }
                    for m in memories
                ],
            }
        except Exception as e:
            logger.error(f"Error in list_memory: {e}")
            raise
    
    async def delete_all_memory(self) -> dict:
        """
        Delete all memories from the collection.
        
        Returns:
            Result dictionary
        """
        try:
            # Get count before deletion
            count = self.storage.count()
            
            # Delete all memories
            self.storage.delete_all()
            
            logger.info(f"Deleted all {count} memories from collection")
            
            return {
                "status": "success",
                "message": f"All memories deleted successfully",
                "deleted_count": count,
            }
        except Exception as e:
            logger.error(f"Error in delete_all_memory: {e}")
            raise
