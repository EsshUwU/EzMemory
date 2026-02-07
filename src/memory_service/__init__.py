__version__ = "0.1.0"

from .config import Config, ConfigManager, config_manager
from .embeddings import (
    EmbeddingProvider,
    VoyageEmbedding,
    OpenAIEmbedding,
    OpenRouterEmbedding,
    NvidiaEmbedding,
)
from .vector_store import (
    VectorStore,
    QdrantVectorStore,
    PineconeVectorStore,
    ZillizVectorStore,
)
from .memory import (
    Memory,
    MemoryEncoder,
    MemoryStorage,
    MemoryRetrieval,
    MemoryLifecycle,
)
from .mcp import MemoryMCPServer, MemoryHandlers

__all__ = [
    "__version__",
    # Config
    "Config",
    "ConfigManager",
    "config_manager",
    # Embeddings
    "EmbeddingProvider",
    "VoyageEmbedding",
    "OpenAIEmbedding",
    "OpenRouterEmbedding",
    "NvidiaEmbedding",
    # Vector stores
    "VectorStore",
    "QdrantVectorStore",
    "PineconeVectorStore",
    "ZillizVectorStore",
    # Memory core
    "Memory",
    "MemoryEncoder",
    "MemoryStorage",
    "MemoryRetrieval",
    "MemoryLifecycle",
    # MCP server
    "MemoryMCPServer",
    "MemoryHandlers",
]
