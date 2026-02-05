import json
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field
from .constants import (
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_MEMORY_TTL_DAYS,
    DEFAULT_MCP_HOST,
    DEFAULT_MCP_PORT,
)


class OpenAIProviderConfig(BaseModel):
    """OpenAI provider configuration."""
    api_key: Optional[str] = Field(None, description="OpenAI API key")
    model: str = Field("text-embedding-3-small", description="Model name")
    embedding_dimension: Optional[int] = Field(None, description="Vector dimension")


class VoyageAIProviderConfig(BaseModel):
    """VoyageAI provider configuration."""
    api_key: Optional[str] = Field(None, description="VoyageAI API key")
    model: str = Field("voyage-4", description="Model name")
    embedding_dimension: Optional[int] = Field(None, description="Vector dimension")


class OpenRouterProviderConfig(BaseModel):
    """OpenRouter provider configuration."""
    api_key: Optional[str] = Field(None, description="OpenRouter API key")
    model: str = Field("google/gemini-embedding-001", description="Model name")
    base_url: str = Field("https://openrouter.ai/api/v1", description="API base URL")
    http_referer: Optional[str] = Field(None, description="HTTP referer for rankings")
    site_name: Optional[str] = Field(None, description="Site name for rankings")
    embedding_dimension: Optional[int] = Field(None, description="Vector dimension")


class EmbeddingConfig(BaseModel):
    """Embedding configuration with support for multiple providers."""
    active_provider: str = Field("openai", description="Currently active provider")
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)
    voyageai: VoyageAIProviderConfig = Field(default_factory=VoyageAIProviderConfig)
    openrouter: OpenRouterProviderConfig = Field(default_factory=OpenRouterProviderConfig)
    
    def get_active_config(self):
        """Get the configuration for the active provider."""
        if self.active_provider == "openai":
            return self.openai
        elif self.active_provider == "voyageai":
            return self.voyageai
        elif self.active_provider == "openrouter":
            return self.openrouter
        else:
            raise ValueError(f"Unknown provider: {self.active_provider}")


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    provider: str = Field(..., description="Vector store provider (qdrant, pinecone)")
    host: Optional[str] = Field(None, description="Host address (Qdrant)")
    port: Optional[int] = Field(None, description="Port number (Qdrant)")
    url: Optional[str] = Field(None, description="Full URL for cloud instances (Qdrant)")
    api_key: Optional[str] = Field(None, description="API key for cloud instances")
    collection_name: str = Field(DEFAULT_COLLECTION_NAME, description="Collection/Index name")
    distance_metric: str = Field(DEFAULT_DISTANCE_METRIC, description="Distance metric")
    prefer_grpc: bool = Field(False, description="Use gRPC for better performance (Qdrant)")
    cloud: Optional[str] = Field(None, description="Cloud provider for Pinecone (aws, gcp, azure)")
    region: Optional[str] = Field(None, description="Cloud region for Pinecone")


class SearchConfig(BaseModel):
    """Search configuration."""
    default_limit: int = Field(DEFAULT_SEARCH_LIMIT, description="Default number of results")
    score_threshold: Optional[float] = Field(None, description="Minimum score threshold")
    hnsw_ef: Optional[int] = Field(None, description="HNSW ef parameter for search")


class MemoryConfig(BaseModel):
    """Memory lifecycle configuration."""
    ttl_days: int = Field(DEFAULT_MEMORY_TTL_DAYS, description="Default memory retention in days")
    enable_decay: bool = Field(False, description="Enable memory decay over time")
    decay_factor: float = Field(0.95, description="Decay factor for memory scoring")


class MCPConfig(BaseModel):
    """MCP server configuration."""
    host: str = Field(DEFAULT_MCP_HOST, description="Server host")
    port: int = Field(DEFAULT_MCP_PORT, description="Server port")
    auto_start: bool = Field(False, description="Auto-start server on init")


class Config(BaseModel):
    """Main configuration."""
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    search: SearchConfig = Field(default_factory=SearchConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)


class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_path: Path = CONFIG_FILE):
        self.config_path = config_path
        self._config: Optional[Config] = None
    
    def ensure_config_dir(self) -> None:
        """Create config directory if it doesn't exist."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    def _migrate_old_config(self, data: dict) -> dict:
        """Migrate old config format to new nested format."""
        # Check if this is an old format config
        if "embedding" in data and "provider" in data["embedding"]:
            old_embedding = data["embedding"]
            provider = old_embedding.get("provider")
            
            # Create new nested structure
            new_embedding = {
                "active_provider": provider,
                "openai": {
                    "api_key": None,
                    "model": "text-embedding-3-small",
                    "embedding_dimension": None
                },
                "voyageai": {
                    "api_key": None,
                    "model": "voyage-4",
                    "embedding_dimension": None
                },
                "openrouter": {
                    "api_key": None,
                    "model": "google/gemini-embedding-001",
                    "base_url": "https://openrouter.ai/api/v1",
                    "http_referer": None,
                    "site_name": None,
                    "embedding_dimension": None
                }
            }
            
            # Populate the active provider's config
            if provider == "openai":
                new_embedding["openai"] = {
                    "api_key": old_embedding.get("api_key"),
                    "model": old_embedding.get("model", "text-embedding-3-small"),
                    "embedding_dimension": old_embedding.get("embedding_dimension")
                }
            elif provider == "voyageai":
                new_embedding["voyageai"] = {
                    "api_key": old_embedding.get("api_key"),
                    "model": old_embedding.get("model", "voyage-4"),
                    "embedding_dimension": old_embedding.get("embedding_dimension")
                }
            elif provider == "openrouter":
                new_embedding["openrouter"] = {
                    "api_key": old_embedding.get("api_key"),
                    "model": old_embedding.get("model", "google/gemini-embedding-001"),
                    "base_url": old_embedding.get("base_url", "https://openrouter.ai/api/v1"),
                    "http_referer": old_embedding.get("http_referer"),
                    "site_name": old_embedding.get("site_name"),
                    "embedding_dimension": old_embedding.get("embedding_dimension")
                }
            
            data["embedding"] = new_embedding
        
        return data
    
    def load(self) -> Config:
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {self.config_path}. "
                "Please run the setup first."
            )
        
        with open(self.config_path, "r") as f:
            data = json.load(f)
        
        # Migrate old format if needed
        data = self._migrate_old_config(data)
        
        self._config = Config(**data)
        
        # Save migrated config back to file
        if "provider" in str(data.get("embedding", {})):
            self.save(self._config)
        
        return self._config
    
    def save(self, config: Config) -> None:
        """Save configuration to file."""
        self.ensure_config_dir()
        
        with open(self.config_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
        
        self._config = config
    
    def get(self) -> Config:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load()
        return self._config
    
    def update(self, **kwargs) -> Config:
        """Update configuration fields."""
        config = self.get()
        config_dict = config.model_dump()
        
        # Deep update
        for key, value in kwargs.items():
            if "." in key:
                parts = key.split(".")
                current = config_dict
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
        
        updated_config = Config(**config_dict)
        self.save(updated_config)
        return updated_config


# Global config manager instance
config_manager = ConfigManager()
