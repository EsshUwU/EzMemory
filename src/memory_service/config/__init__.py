"""
Configuration module initialization.
"""

from .settings import (
    Config,
    ConfigManager,
    EmbeddingConfig,
    VectorStoreConfig,
    SearchConfig,
    MemoryConfig,
    MCPConfig,
    config_manager,
)
from .logging import setup_logging, get_logger
from .constants import (
    CONFIG_DIR,
    CONFIG_FILE,
    CONFIG_DOCS,
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_VECTOR_SIZE,
)

__all__ = [
    "Config",
    "ConfigManager",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "SearchConfig",
    "MemoryConfig",
    "MCPConfig",
    "config_manager",
    "setup_logging",
    "get_logger",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "CONFIG_DOCS",
    "DEFAULT_EMBEDDING_MODELS",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_VECTOR_SIZE",
]
