from openai import OpenAI
from typing import List, Union
from .base import EmbeddingProvider


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider implementation."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        self.client = OpenAI(api_key=api_key)
        self._model = model
        self._dimension = self._get_model_dimension()
    
    def _get_model_dimension(self) -> int:
        """Get the dimension for the current model."""
        # Import from constants to avoid duplication
        from ..config.constants import DEFAULT_VECTOR_SIZE
        
        # Check if model exists in constants
        if "openai" in DEFAULT_VECTOR_SIZE and self._model in DEFAULT_VECTOR_SIZE["openai"]:
            return DEFAULT_VECTOR_SIZE["openai"][self._model]
        
        # Fallback to 1536 (most common for OpenAI)
        return 1536
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using OpenAI.
        
        Args:
            text: Single text or list of texts
            **kwargs: Additional parameters (ignored for OpenAI)
            
        Returns:
            Single embedding or list of embeddings
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self._model
            )
            
            embeddings = [item.embedding for item in response.data]
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {str(e)}") from e
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model
