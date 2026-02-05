from openai import OpenAI
from typing import List, Union, Optional
from .base import EmbeddingProvider


class OpenRouterEmbedding(EmbeddingProvider):
    """OpenRouter embedding provider implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-embedding-001",
        http_referer: Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key
            model: Model name to use
            http_referer: Optional HTTP referer for rankings
            site_name: Optional site name for rankings
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self._model = model
        self._http_referer = http_referer
        self._site_name = site_name
        self._dimension = self._get_model_dimension()
    
    def _get_model_dimension(self) -> int:
        """Get the dimension for the current model."""
        # Import from constants to avoid duplication
        from ..config.constants import DEFAULT_VECTOR_SIZE
        
        # Check if model exists in constants
        if "openrouter" in DEFAULT_VECTOR_SIZE and self._model in DEFAULT_VECTOR_SIZE["openrouter"]:
            return DEFAULT_VECTOR_SIZE["openrouter"][self._model]
        
        # Fallback to 768 (common for many models)
        return 768
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using OpenRouter.
        
        Args:
            text: Single text or list of texts
            **kwargs: Additional parameters (ignored for OpenRouter)
            
        Returns:
            Single embedding or list of embeddings
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        try:
            extra_headers = {}
            if self._http_referer:
                extra_headers["HTTP-Referer"] = self._http_referer
            if self._site_name:
                extra_headers["X-Title"] = self._site_name
            
            create_params = {
                "model": self._model,
                "input": texts,
                "encoding_format": "float",
            }
            
            if extra_headers:
                create_params["extra_headers"] = extra_headers
            
            # Note: Don't add 'dimensions' parameter as it's not supported by all models
            # The model will return its native dimension size
            
            response = self.client.embeddings.create(**create_params)
            
            embeddings = [item.embedding for item in response.data]
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            raise RuntimeError(f"OpenRouter embedding failed: {str(e)}") from e
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model
