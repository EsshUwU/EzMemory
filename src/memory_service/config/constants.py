from pathlib import Path
from os.path import expanduser

# Home directory config
HOME_DIR = Path(expanduser("~"))
CONFIG_DIR = HOME_DIR / ".ezmemory"
CONFIG_FILE = CONFIG_DIR / "config.json"
CONFIG_DOCS = CONFIG_DIR / "config.md"

# Default embedding models
DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "voyageai": "voyage-4",  # Latest Voyage model
    "openrouter": "google/gemini-embedding-001",
}

# Vector database settings
DEFAULT_COLLECTION_NAME = "ezmemory_collection"

# Default vector sizes for common models
# NOTE: This is only a fallback for known models. You can manually specify
# embedding_dimension in your config for any model not listed here.
# Many models support multiple dimensions - check your provider's documentation.
DEFAULT_VECTOR_SIZE = {
    "openai": {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    },
    "voyageai": {
        "voyage-4-large": 1024,
        "voyage-4": 1024,
        "voyage-3.5": 1024,
        "voyage-3.5-lite": 1024,
        "voyage-3-large": 1024,
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-large-2": 1536,
        "voyage-large-2-instruct": 1024,
        "voyage-2": 1024,
        "voyage-01": 1024,
    },
    "openrouter": {
        "google/gemini-embedding-001": 3072,
        "openai/text-embedding-ada-002": 1536,
        "openai/text-embedding-3-large": 3072,
        "openai/text-embedding-3-small": 1536,
        "mistralai/mistral-embed-2312": 1024,
        "mistralai/codestral-embed-2505": 1536,
        "qwen/qwen3-embedding-8b": 4096,
        "qwen/qwen3-embedding-4b": 2560,
        "thenlper/gte-base": 768,
        "thenlper/gte-large": 1024,
        "intfloat/e5-large-v2": 1024,
        "intfloat/e5-base-v2": 768,
        "intfloat/multilingual-e5-large": 1024,
        "sentence-transformers/paraphrase-minilm-l6-v2": 384,
        "sentence-transformers/all-minilm-l12-v2": 384,
        "sentence-transformers/multi-qa-mpnet-base-dot-v1": 768,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/all-minilm-l6-v2": 384,
        "baai/bge-base-en-v1.5": 768,
        "baai/bge-large-en-v1.5": 1024,
        "baai/bge-m3": 1024,
    },
}

# Default dimension to use when model is not found in DEFAULT_VECTOR_SIZE
# This will trigger a warning to prompt users to specify embedding_dimension in config
DEFAULT_FALLBACK_DIMENSION = 1024

# Search settings
DEFAULT_SEARCH_LIMIT = 5
SUPPORTED_SEARCH_ALGORITHMS = ["hnsw", "exact"]
DEFAULT_DISTANCE_METRIC = "cosine"
SUPPORTED_DISTANCE_METRICS = ["cosine", "euclid", "dot", "manhattan"]

# Memory settings
DEFAULT_MEMORY_TTL_DAYS = 90  # 90 days default retention

# MCP Server settings
DEFAULT_MCP_HOST = "localhost"
DEFAULT_MCP_PORT = 8080
