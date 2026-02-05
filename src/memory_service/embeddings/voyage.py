import voyageai
from typing import List, Union
from .base import EmbeddingProvider


class VoyageEmbedding(EmbeddingProvider):
    """Voyage AI embedding provider implementation."""
    
    def __init__(self, api_key: str, model: str = "voyage-3"):
        """
        Initialize Voyage AI provider.
        
        Args:
            api_key: Voyage AI API key
            model: Model name to use
        """
        self.client = voyageai.Client(api_key=api_key)
        self._model = model
        self._dimension = self._get_model_dimension()
    
    def _get_model_dimension(self) -> int:
        """Get the dimension for the current model."""
        # Import from constants to avoid duplication
        from ..config.constants import DEFAULT_VECTOR_SIZE
        
        # Check if model exists in constants
        if "voyageai" in DEFAULT_VECTOR_SIZE and self._model in DEFAULT_VECTOR_SIZE["voyageai"]:
            return DEFAULT_VECTOR_SIZE["voyageai"][self._model]
        
        # Fallback to 1024 (most common for Voyage models)
        return 1024
    
    def embed(self, text: Union[str, List[str]], input_type: str = "document") -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using Voyage AI.
        
        Args:
            text: Single text or list of texts
            input_type: Type of input - "document" for storage, "query" for search
            
        Returns:
            Single embedding or list of embeddings
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        try:
            result = self.client.embed(
                texts=texts,
                model=self._model,
                input_type=input_type
            )
            
            embeddings = result.embeddings
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            raise RuntimeError(f"Voyage AI embedding failed: {str(e)}") from e
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model
