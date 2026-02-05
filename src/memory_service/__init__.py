"""
EzMemory - AI Agent Memory System
"""

__version__ = "0.1.0"

from .config import Config, ConfigManager
from .embeddings import EmbeddingProvider, VoyageEmbedding, OpenAIEmbedding, OpenRouterEmbedding
from .vector_store import VectorStore, QdrantVectorStore
from .memory import Memory, MemoryEncoder, MemoryStorage, MemoryRetrieval, MemoryLifecycle
from .mcp import MemoryMCPServer, MemoryHandlers

__all__ = [
    "__version__",
    "Config",
    "ConfigManager",
    "EmbeddingProvider",
    "VoyageEmbedding",
    "OpenAIEmbedding",
    "OpenRouterEmbedding",
    "VectorStore",
    "QdrantVectorStore",
    "Memory",
    "MemoryEncoder",
    "MemoryStorage",
    "MemoryRetrieval",
    "MemoryLifecycle",
    "MemoryMCPServer",
    "MemoryHandlers",
]
