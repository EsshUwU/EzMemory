from abc import ABC, abstractmethod
from typing import List, Union


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text input.
        
        Args:
            text: Single text string or list of text strings
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass
