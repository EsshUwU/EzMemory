"""
Memory encoder - converts text to embeddings.
"""

from typing import List, Union
from .schemas import Memory
from ..embeddings.base import EmbeddingProvider
from ..utils.text import normalize_text


class MemoryEncoder:
    """Encodes memories using embedding provider."""
    
    def __init__(self, embedding_provider: EmbeddingProvider):
        """
        Initialize encoder.
        
        Args:
            embedding_provider: Embedding provider to use
        """
        self.embedding_provider = embedding_provider
    
    def encode(self, memory: Memory) -> Memory:
        """
        Encode a single memory.
        
        Args:
            memory: Memory to encode
            
        Returns:
            Memory with embedding
        
        Raises:
            ValueError: If memory content is empty
        """
        if not memory.content or not memory.content.strip():
            raise ValueError("Memory content cannot be empty")
        
        normalized_content = normalize_text(memory.content)
        # Use input_type="document" for storage (Voyage AI specific)
        embedding = self.embedding_provider.embed(normalized_content, input_type="document")
        memory.embedding = embedding
        return memory
    
    def encode_batch(self, memories: List[Memory]) -> List[Memory]:
        """
        Encode multiple memories.
        
        Args:
            memories: List of memories to encode
            
        Returns:
            List of memories with embeddings
        """
        if not memories:
            return memories
        
        contents = [normalize_text(m.content) for m in memories]
        # Use input_type="document" for storage (Voyage AI specific)
        embeddings = self.embedding_provider.embed(contents, input_type="document")
        
        for memory, embedding in zip(memories, embeddings):
            memory.embedding = embedding
        
        return memories
    
    def encode_query(self, query: str) -> List[float]:
        """
        Encode a search query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        normalized_query = normalize_text(query)
        # Use input_type="query" for search (Voyage AI specific, ignored by others)
        return self.embedding_provider.embed(normalized_query, input_type="query")
