"""Gemini embedding provider using the Google AI Gemini API."""

from typing import List, Optional, Union
from google import genai
from google.genai import types

from .base import EmbeddingProvider


class GeminiEmbedding(EmbeddingProvider):
    """Gemini embedding provider implementation via Google AI API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        embedding_dimension: Optional[int] = None,
        task_type: Optional[str] = None,
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google AI (Gemini) API key
            model: Model name (default: gemini-embedding-001)
            embedding_dimension: Optional output dimension (768, 1536, or 3072)
            task_type: Optional task type (RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, etc.)
        """

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._embedding_dimension = embedding_dimension
        self._task_type = task_type
        self._types = types

    def embed(
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using Gemini API.

        Args:
            text: Single text or list of texts
            **kwargs: Additional parameters (ignored)

        Returns:
            Single embedding or list of embeddings
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        config_kwargs = {}
        if self._embedding_dimension is not None:
            config_kwargs["output_dimensionality"] = self._embedding_dimension
        if self._task_type is not None:
            config_kwargs["task_type"] = self._task_type

        try:
            if config_kwargs:
                config = self._types.EmbedContentConfig(**config_kwargs)
                result = self._client.models.embed_content(
                    model=self._model,
                    contents=texts,
                    config=config,
                )
            else:
                result = self._client.models.embed_content(
                    model=self._model,
                    contents=texts,
                )

            embeddings = [list(e.values) for e in result.embeddings]
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            raise RuntimeError(f"Gemini embedding failed: {str(e)}") from e

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dimension is not None:
            return self._embedding_dimension
        from ..config.constants import DEFAULT_VECTOR_SIZE

        if "gemini" in DEFAULT_VECTOR_SIZE and self._model in DEFAULT_VECTOR_SIZE["gemini"]:
            return DEFAULT_VECTOR_SIZE["gemini"][self._model]
        return 3072  # Default for gemini-embedding-001

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model
