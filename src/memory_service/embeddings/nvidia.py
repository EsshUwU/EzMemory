from openai import OpenAI
from typing import List, Union, Optional
from .base import EmbeddingProvider


class NvidiaEmbedding(EmbeddingProvider):
    """NVIDIA embedding provider implementation (NVIDIA NIM / integrate API)."""

    def __init__(
        self,
        api_key: str,
        model: str = "nvidia/nv-embedqa-e5-v5",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        embedding_dimension: Optional[int] = None,
    ):
        """
        Initialize NVIDIA provider.

        Args:
            api_key: NVIDIA API key
            model: Model name (e.g. nvidia/nv-embedqa-e5-v5)
            base_url: API base URL (default: NVIDIA integrate API)
            embedding_dimension: Override dimension (e.g. from config after auto-detect)
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._model = model
        self._base_url = base_url
        self._dimension = (
            embedding_dimension
            if embedding_dimension is not None
            else self._get_model_dimension()
        )

    def _get_model_dimension(self) -> int:
        """Fallback dimension when not provided (actual from auto-detect at setup)."""
        return 1024

    def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using NVIDIA API.

        Args:
            text: Single text or list of texts
            **kwargs: Additional parameters (e.g. input_type, truncate)

        Returns:
            Single embedding or list of embeddings
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        try:
            extra_body = kwargs.get("extra_body") or {}
            if "input_type" not in extra_body:
                extra_body["input_type"] = "passage" if not is_single else "query"
            if "truncate" not in extra_body:
                extra_body["truncate"] = "NONE"

            response = self.client.embeddings.create(
                input=texts,
                model=self._model,
                encoding_format="float",
                extra_body=extra_body,
            )

            embeddings = [item.embedding for item in response.data]
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            raise RuntimeError(f"NVIDIA embedding failed: {str(e)}") from e

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model
