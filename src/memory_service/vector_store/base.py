"""
Base interface for vector store providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from uuid import UUID


class VectorStore(ABC):
    """Abstract base class for vector store providers."""
    
    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine",
        **kwargs
    ) -> None:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance_metric: Distance metric to use
            **kwargs: Additional provider-specific parameters
        """
        pass
    
    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists, False otherwise
        """
        pass
    
    @abstractmethod
    def insert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Insert vectors with payloads.
        
        Args:
            collection_name: Name of the collection
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries
            ids: Optional list of IDs (generated if not provided)
            
        Returns:
            List of inserted IDs
        """
        pass
    
    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Optional metadata filters
            
        Returns:
            List of results with id, score, and payload
        """
        pass
    
    @abstractmethod
    def delete(
        self,
        collection_name: str,
        ids: List[str],
    ) -> None:
        """
        Delete vectors by IDs.
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs to delete
        """
        pass
    
    @abstractmethod
    def get_all(
        self,
        collection_name: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all vectors from collection.
        
        Args:
            collection_name: Name of the collection
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of results with id and payload
        """
        pass
    
    @abstractmethod
    def count(self, collection_name: str) -> int:
        """
        Count vectors in collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of vectors
        """
        pass
    
    @abstractmethod
    def delete_all(self, collection_name: str) -> None:
        """
        Delete all vectors from collection.
        
        Args:
            collection_name: Name of the collection
        """
        pass
