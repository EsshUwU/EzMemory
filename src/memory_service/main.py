"""
Main CLI entry point for EzMemory.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import questionary
from rich.console import Console
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich import box

from .config.constants import (
    CONFIG_DIR,
    CONFIG_FILE,
    CONFIG_DOCS,
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_VECTOR_SIZE,
    DEFAULT_FALLBACK_DIMENSION,
    DEFAULT_COLLECTION_NAME,
)
from .config.settings import (
    Config,
    ConfigManager,
    EmbeddingConfig,
    KnownCollectionInfo,
    OpenAIProviderConfig,
    VoyageAIProviderConfig,
    OpenRouterProviderConfig,
    NvidiaProviderConfig,
    GeminiProviderConfig,
    VectorStoreConfig,
)
from .config.logging import setup_logging, get_logger
from .config.config_docs import CONFIG_DOCS_CONTENT
from .embeddings import VoyageEmbedding, OpenAIEmbedding, OpenRouterEmbedding, NvidiaEmbedding, GeminiEmbedding
from .vector_store import QdrantVectorStore, PineconeVectorStore, ZillizVectorStore
from .memory import Memory, MemoryEncoder, MemoryStorage, MemoryRetrieval
from .memory.viewer import format_memory_for_display
from .mcp import MemoryMCPServer, MemoryHandlers
from .utils import add_cursor_mcp, add_vscode_mcp
from .http_server import HTTPServer

console = Console()
logger = get_logger(__name__)

ACCENT_COLOR = "cyan"

# Stylized "EZMEMORY" banner with box-drawing characters
_EZMEMORY_BANNER_LINES = [
    "███████╗███████╗███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗",
    "██╔════╝╚══███╔╝████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝",
    "█████╗    ███╔╝ ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝",
    "██╔══╝   ███╔╝  ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝",
    "███████╗███████╗██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║",
    "╚══════╝╚══════╝╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝",
]


def _styled_banner_line(line: str) -> str:
    """Apply cyan color to the line."""
    if len(line) == 0:
        return ""
    
    return f"[bold cyan]{line}[/]"


class BackToMainMenu(Exception):
    """Raised to unwind submenus back to the main menu."""


class EzMemoryCLI:
    """CLI for EzMemory setup and management."""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.console = console
        # Short status line shown on the main menu after operations
        self.last_status: Optional[str] = None

    def _set_status(self, message: str) -> None:
        """Update the status line shown on the main menu."""
        self.last_status = message
        logger.info(message)

    def print_header(self):
        """Print welcome header (blocky EZMEMORY banner + tagline)."""
        banner = "\n".join(_styled_banner_line(ln) for ln in _EZMEMORY_BANNER_LINES)
        tagline = "[dim]AI Agent Memory System[/dim]"
        content = Align.center(f"{banner}\n\n{tagline}")
        panel = Panel(
            content,
            border_style=ACCENT_COLOR,
            box=box.ROUNDED,
            padding=(1, 6),
        )
        self.console.print(panel)

    def _render_page(self, title: str, subtitle: Optional[str] = None):
        """Render a page-style screen with a header."""
        self.console.clear()
        self.print_header()
        self.console.print(f"[bold]{title}[/bold]")
        if subtitle:
            self.console.print(f"[dim]{subtitle}[/dim]")
        self.console.print()

    async def setup_wizard(self) -> Config:
        """Run interactive setup wizard."""
        self._render_page("EzMemory Setup", "Let's configure your memory system.")

        # Vector Database Selection
        vector_db = await self._select_vector_database()
        vector_config = await self._configure_vector_store(vector_db)

        # Embedding Provider Selection (includes dimension detection)
        embedding_provider = await self._select_embedding_provider()
        embedding_config = await self._configure_embedding(
            embedding_provider, existing_config=None
        )

        # Test vector store connection
        self.console.print(
            "\n[bold yellow]Testing vector database connection...[/bold yellow]\n"
        )
        try:
            vector_store = self._create_vector_store(vector_config)
            self.console.print(
                f"✓ [green]Connected to {vector_db.title()} successfully[/green]"
            )
        except Exception as e:
            self.console.print(
                f"✗ [red]Failed to connect to {vector_db.title()}: {e}[/red]"
            )
            sys.exit(1)

        # Ask for collection name (optional)
        name_prefix = await self._ask_collection_name()

        # Initialize collection with detected dimension
        self.console.print("\n[bold yellow]Setting up collection...[/bold yellow]\n")
        try:
            # Use detected dimension from embedding_config
            vector_size = self._get_vector_size(embedding_config)

            # Generate collection name based on prefix + embedding model
            collection_name = self._get_collection_name(
                embedding_config, vector_config.provider, name_prefix
            )
            vector_config.collection_name = collection_name

            if not vector_store.collection_exists(collection_name):
                vector_store.create_collection(
                    collection_name=collection_name,
                    vector_size=vector_size,
                    distance_metric=vector_config.distance_metric,
                )
                self.console.print(
                    f"✓ [green]Created collection: {collection_name}[/green]"
                )
            else:
                self.console.print(
                    f"✓ [green]Collection already exists: {collection_name}[/green]"
                )
        except Exception as e:
            self.console.print(f"✗ [red]Failed to create collection: {e}[/red]")
            sys.exit(1)

        # Build known_collections entry
        active_provider_config = embedding_config.get_active_config()
        known_collections = {
            collection_name: KnownCollectionInfo(
                provider=embedding_config.active_provider,
                model=active_provider_config.model,
                dim=vector_size,
                http_referer=getattr(active_provider_config, "http_referer", None),
                site_name=getattr(active_provider_config, "site_name", None),
            )
        }

        # Create full config
        config = Config(
            embedding=embedding_config,
            vector_store=vector_config,
            known_collections=known_collections,
        )

        # Save config
        self.config_manager.save(config)

        # Save config docs
        self._save_config_docs()

        self.console.print(f"\n✓ [green]Configuration saved to {CONFIG_FILE}[/green]")
        self.console.print(
            f"[dim]Model and other settings can be edited in: {CONFIG_FILE}[/dim]"
        )
        self.console.print(f"[dim]Read configuration guide at: {CONFIG_DOCS}[/dim]\n")

        # Update status for main menu
        active = embedding_config.get_active_config()
        self._set_status(
            f"Setup complete · Vector DB: {vector_config.provider} · "
            f"Embedding: {embedding_config.active_provider}/{active.model} · "
            f"Collection: {collection_name}"
        )

        # Pause so user can read the results
        self.console.print()
        self.console.print("[dim]Press Enter to continue...[/dim]", end="")
        await questionary.text("", default="").ask_async()

        return config

    async def _select_vector_database(self) -> str:
        """Select vector database."""
        self._render_page("1. Select Vector Database")

        while True:
            choice = await questionary.select(
                "",
                choices=[
                    "Qdrant",
                    "Pinecone",
                    "Zilliz (Milvus Cloud)",
                    questionary.Separator(),
                    "Local Storage (coming soon)",
                    questionary.Separator(),
                    "Back",
                ],
            ).ask_async()

            if choice == "Back":
                raise BackToMainMenu()

            if choice == "Local Storage (coming soon)":
                self.console.print("[red]Local Storage is not yet supported.[/red]\n")
                continue

            if choice == "Qdrant":
                return "qdrant"
            if choice == "Pinecone":
                return "pinecone"
            if choice == "Zilliz (Milvus Cloud)":
                return "zilliz"

            # Defensive fallback
            return str(choice).lower()

    async def _configure_vector_store(self, provider: str) -> VectorStoreConfig:
        """Configure vector store."""
        if provider == "qdrant":
            self._render_page("Configure Qdrant")

            url = await questionary.text("Qdrant Cloud URL:").ask_async()

            api_key = await questionary.password(
                "API Key:",
            ).ask_async()

            return VectorStoreConfig(
                provider=provider,
                host=None,
                port=None,
                url=url,
                api_key=api_key,
                collection_name=DEFAULT_COLLECTION_NAME,
                distance_metric="cosine",
                prefer_grpc=False,
                cloud=None,
                region=None,
            )

        elif provider == "pinecone":
            self._render_page("Configure Pinecone")

            api_key = await questionary.password(
                "Pinecone API Key:",
            ).ask_async()

            cloud = await questionary.select(
                "Select cloud provider:",
                choices=["aws", "gcp", "azure"],
                default="aws",
            ).ask_async()

            # Set appropriate default region based on cloud provider
            default_regions = {
                "aws": "us-east-1",
                "gcp": "us-central1",
                "azure": "eastus",
            }

            # Show cloud-specific region examples
            region_examples = {
                "aws": "us-east-1, us-west-2, eu-west-1",
                "gcp": "us-central1, europe-west4, asia-southeast1",
                "azure": "eastus, westus, northeurope",
            }

            self.console.print(
                f"\n[dim]Example {cloud.upper()} regions: {region_examples[cloud]}[/dim]"
            )

            region = await questionary.text(
                "Region:",
                default=default_regions[cloud],
            ).ask_async()

            return VectorStoreConfig(
                provider=provider,
                host=None,
                port=None,
                url=None,
                api_key=api_key,
                collection_name=DEFAULT_COLLECTION_NAME,
                distance_metric="cosine",
                prefer_grpc=False,
                cloud=cloud,
                region=region,
            )

        elif provider == "zilliz":
            self._render_page("Configure Zilliz (Milvus Cloud)")

            uri = await questionary.text(
                "Zilliz Cloud URI (from console, e.g. https://...zillizcloud.com:19530):",
            ).ask_async()

            token = await questionary.password(
                "Zilliz Token (user:password or API key):",
            ).ask_async()

            return VectorStoreConfig(
                provider=provider,
                host=None,
                port=None,
                url=uri,
                api_key=token,
                collection_name=DEFAULT_COLLECTION_NAME,
                distance_metric="cosine",
                prefer_grpc=False,
                cloud=None,
                region=None,
            )

        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")

    async def _select_embedding_provider(self) -> str:
        """Select embedding provider."""
        self._render_page("2. Select Embedding Provider")

        while True:
            choice = await questionary.select(
                "",
                choices=[
                    "OpenAI",
                    "VoyageAI",
                    "OpenRouter",
                    "NVIDIA",
                    "Gemini",
                    questionary.Separator(),
                    "Back",
                ],
            ).ask_async()

            if choice == "Back":
                raise BackToMainMenu()

            if choice == "OpenAI":
                return "openai"
            if choice == "VoyageAI":
                return "voyageai"
            if choice == "OpenRouter":
                return "openrouter"
            if choice == "NVIDIA":
                return "nvidia"
            if choice == "Gemini":
                return "gemini"

            # Defensive fallback
            return str(choice).lower()

    async def _select_embedding_model(self, provider: str) -> str:
        """Select embedding model based on provider."""
        self._render_page(f"Select {provider.title()} Model")

        if provider.lower() == "voyageai":
            models = [
                "voyage-4-large",
                "voyage-4",
                "voyage-3.5",
                "voyage-3.5-lite",
                "voyage-3-large",
                "voyage-3",
                "voyage-3-lite",
                "voyage-large-2",
                "voyage-large-2-instruct",
                "voyage-2",
                "voyage-01",
            ]
        elif provider.lower() == "openrouter":
            models = [
                "google/gemini-embedding-001",
                "openai/text-embedding-ada-002",
                "openai/text-embedding-3-large",
                "openai/text-embedding-3-small",
                "mistralai/mistral-embed-2312",
                "mistralai/codestral-embed-2505",
                "qwen/qwen3-embedding-8b",
                "qwen/qwen3-embedding-4b",
                "thenlper/gte-base",
                "thenlper/gte-large",
                "intfloat/e5-large-v2",
                "intfloat/e5-base-v2",
                "intfloat/multilingual-e5-large",
                "sentence-transformers/paraphrase-minilm-l6-v2",
                "sentence-transformers/all-minilm-l12-v2",
                "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-minilm-l6-v2",
                "baai/bge-base-en-v1.5",
                "baai/bge-large-en-v1.5",
                "baai/bge-m3",
            ]
        elif provider.lower() == "nvidia":
            models = [
                "nvidia/llama-3_2-nemoretriever-300m-embed-v2",
                "nvidia/llama-3_2-nemoretriever-300m-embed-v1",
                "nvidia/nv-embed-v1",
                "nvidia/nv-embedqa-e5-v5",
                "nvidia/llama-3.2-nv-embedqa-1b-v2",
                "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                "nvidia/bge-m3",
            ]
        elif provider.lower() == "openai":
            models = [
                "text-embedding-3-large",
                "text-embedding-3-small",
                "text-embedding-ada-002",
            ]
        elif provider.lower() == "gemini":
            models = [
                "gemini-embedding-001",
            ]
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        choice = await questionary.select(
            "Choose a model:",
            choices=[*models, questionary.Separator(), "Back"],
        ).ask_async()

        if choice == "Back":
            raise BackToMainMenu()

        return choice

    async def _get_embedding_dimension_for_provider(
        self, provider: str, provider_config
    ) -> int:
        """
        Get embedding dimension for a model.

        Priority:
        1. Check DEFAULT_VECTOR_SIZE for known models
        2. Detect dimension by creating test embedding
        3. Ask user to manually enter dimension
        4. Fall back to DEFAULT_FALLBACK_DIMENSION if user skips
        """
        model = provider_config.model

        # Check if we have a cached dimension
        if provider in DEFAULT_VECTOR_SIZE and model in DEFAULT_VECTOR_SIZE[provider]:
            dimension = DEFAULT_VECTOR_SIZE[provider][model]
            self.console.print(
                f"\n✓ [green]Using cached dimension: {dimension}[/green]"
            )
            return dimension

        # Model not in cache, detect dimension
        self.console.print(
            f"\n[yellow]Model '{model}' not in cache. Detecting dimension...[/yellow]"
        )

        try:
            # Create embedding provider
            embedding_provider = self._create_embedding_provider_from_config(
                provider, provider_config
            )

            # Generate test embedding
            test_embedding = embedding_provider.embed("hello")
            dimension = len(test_embedding)

            self.console.print(f"✓ [green]Detected dimension: {dimension}[/green]")
            return dimension
        except Exception as e:
            self.console.print(f"✗ [red]Dimension not found: {e}[/red]\n")

            # Ask user to manually enter dimension
            manual_dim = await questionary.text(
                "Enter embedding dimension manually (or press Enter to use default 1024):",
                default="",
            ).ask_async()

            if manual_dim and manual_dim.strip():
                try:
                    dimension = int(manual_dim.strip())
                    self.console.print(
                        f"✓ [green]Using manually entered dimension: {dimension}[/green]"
                    )
                    return dimension
                except ValueError:
                    self.console.print(
                        f"[yellow]Invalid input. Using default dimension: {DEFAULT_FALLBACK_DIMENSION}[/yellow]"
                    )
                    return DEFAULT_FALLBACK_DIMENSION
            else:
                self.console.print(
                    f"[yellow]Using default dimension: {DEFAULT_FALLBACK_DIMENSION}[/yellow]"
                )
                return DEFAULT_FALLBACK_DIMENSION

    async def _configure_embedding(
        self, provider: str, existing_config: Optional[EmbeddingConfig] = None
    ) -> EmbeddingConfig:
        """Configure embedding provider."""
        self._render_page(f"Configure {provider.title()}")

        # Check if API key already exists for this provider
        existing_api_key = None
        if existing_config:
            if provider.lower() == "openai" and existing_config.openai.api_key:
                existing_api_key = existing_config.openai.api_key
            elif provider.lower() == "voyageai" and existing_config.voyageai.api_key:
                existing_api_key = existing_config.voyageai.api_key
            elif (
                provider.lower() == "openrouter" and existing_config.openrouter.api_key
            ):
                existing_api_key = existing_config.openrouter.api_key
            elif provider.lower() == "nvidia" and existing_config.nvidia.api_key:
                existing_api_key = existing_config.nvidia.api_key
            elif provider.lower() == "gemini" and existing_config.gemini.api_key:
                existing_api_key = existing_config.gemini.api_key

        # Get API key (skip if already exists)
        if existing_api_key:
            self.console.print(f"[green]✓ API key already configured[/green]")
            api_key = existing_api_key
        else:
            api_key = await questionary.password(
                f"{provider.title()} API Key:",
            ).ask_async()

        # Select model
        model = await self._select_embedding_model(provider)

        # Build provider-specific config
        if provider.lower() == "openrouter":
            http_referer = await questionary.text(
                "HTTP Referer (optional, for rankings):",
                default="",
            ).ask_async()

            site_name = await questionary.text(
                "Site Name (optional, for rankings):",
                default="",
            ).ask_async()

            provider_config = OpenRouterProviderConfig(
                api_key=api_key,
                model=model,
                base_url="https://openrouter.ai/api/v1",
                http_referer=http_referer if http_referer else None,
                site_name=site_name if site_name else None,
                embedding_dimension=None,  # Will be detected
            )
        elif provider.lower() == "nvidia":
            provider_config = NvidiaProviderConfig(
                api_key=api_key,
                model=model,
                base_url="https://integrate.api.nvidia.com/v1",
                embedding_dimension=None,  # Will be detected
            )
        elif provider.lower() == "gemini":
            provider_config = GeminiProviderConfig(
                api_key=api_key,
                model=model,
                embedding_dimension=None,  # Will be detected
            )
        elif provider.lower() == "voyageai":
            provider_config = VoyageAIProviderConfig(
                api_key=api_key,
                model=model,
                embedding_dimension=None,  # Will be detected
            )
        else:  # openai
            provider_config = OpenAIProviderConfig(
                api_key=api_key,
                model=model,
                embedding_dimension=None,  # Will be detected
            )

        # Get dimension (from cache or detect)
        actual_dimension = await self._get_embedding_dimension_for_provider(
            provider, provider_config
        )
        provider_config.embedding_dimension = actual_dimension

        self.console.print(f"\n[bold cyan]Using model:[/bold cyan] {provider}/{model}")
        self.console.print(
            f"[bold cyan]Embedding dimension:[/bold cyan] {actual_dimension}"
        )

        # Create or update full embedding config
        if existing_config is None:
            embedding_config = EmbeddingConfig(active_provider=provider.lower())
        else:
            embedding_config = existing_config
            embedding_config.active_provider = provider.lower()

        # Update the provider-specific config
        if provider.lower() == "openai":
            embedding_config.openai = OpenAIProviderConfig(
                **provider_config.model_dump()
            )
        elif provider.lower() == "voyageai":
            embedding_config.voyageai = VoyageAIProviderConfig(
                **provider_config.model_dump()
            )
        elif provider.lower() == "openrouter":
            embedding_config.openrouter = OpenRouterProviderConfig(
                **provider_config.model_dump()
            )
        elif provider.lower() == "nvidia":
            embedding_config.nvidia = NvidiaProviderConfig(
                **provider_config.model_dump()
            )
        elif provider.lower() == "gemini":
            embedding_config.gemini = GeminiProviderConfig(
                **provider_config.model_dump()
            )

        return embedding_config

    def _save_config_docs(self):
        """Save configuration documentation."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_DOCS, "w", encoding="utf-8") as f:
            f.write(CONFIG_DOCS_CONTENT)

    def _get_vector_size(self, embedding_config: EmbeddingConfig) -> int:
        """
        Get vector size for the active embedding model.

        Priority:
        1. embedding_dimension from provider config (if set)
        2. DEFAULT_VECTOR_SIZE lookup
        3. DEFAULT_FALLBACK_DIMENSION with warning
        """
        active_config = embedding_config.get_active_config()
        provider = embedding_config.active_provider
        model = active_config.model

        # Priority 1: Check if dimension is set in config
        if active_config.embedding_dimension:
            return active_config.embedding_dimension

        # Priority 2: Check if we have a default for this model
        if provider in DEFAULT_VECTOR_SIZE and model in DEFAULT_VECTOR_SIZE[provider]:
            return DEFAULT_VECTOR_SIZE[provider][model]

        # Priority 3: Fall back to default with warning
        self.console.print(
            f"\n[yellow]⚠ Unknown model '{model}' - using default dimension {DEFAULT_FALLBACK_DIMENSION}[/yellow]"
        )
        self.console.print(
            f"[yellow]  If this is incorrect, set 'embedding_dimension' in {CONFIG_FILE}[/yellow]\n"
        )
        return DEFAULT_FALLBACK_DIMENSION

    def _get_collection_name(
        self,
        embedding_config: EmbeddingConfig,
        vector_provider: Optional[str] = None,
        name_prefix: str = "default",
    ) -> str:
        """
        Generate collection name based on a user-chosen prefix, embedding provider,
        and model.

        Format: {name_prefix}_{provider}_{sanitized_model}

        For Pinecone, ensures name is:
        - 45 characters or less
        - Lowercase
        - Only contains letters, numbers, and hyphens

        For Zilliz/Milvus, ensures name:
        - Uses underscores instead of hyphens
        - Only contains letters, numbers, and underscores
        """
        provider = embedding_config.active_provider
        model = embedding_config.get_active_config().model

        # Special rules per vector provider
        if vector_provider == "zilliz":
            # Milvus/Zilliz allow only letters, numbers, and underscores
            sanitized_model = (
                model.replace("/", "_")
                .replace("-", "_")
                .replace(".", "_")
            )
            base_name = f"{name_prefix}_{provider}_{sanitized_model}".lower()
            return base_name

        # Default: use hyphens (works for Qdrant and others)
        sanitized_model = model.replace("/", "-").replace("_", "-").replace(".", "-")
        base_name = f"{name_prefix}-{provider}-{sanitized_model}".lower()

        # Check if this is for Pinecone and needs length restriction
        if vector_provider == "pinecone" and len(base_name) > 45:
            # Use a shorter version with hash to ensure uniqueness
            import hashlib

            model_hash = hashlib.md5(model.encode()).hexdigest()[:8]
            base_name = f"{name_prefix}-{provider}-{model_hash}".lower()

            # If still too long, truncate prefix and provider
            if len(base_name) > 45:
                base_name = f"{name_prefix[:5]}-{provider[:3]}-{model_hash}".lower()

        return base_name

    def _adapt_collection_name(
        self, collection_name: str, vector_provider: str
    ) -> str:
        """
        Transform an existing collection name to match the target vector DB
        provider's naming rules.

        - Zilliz: all separators become underscores
        - Qdrant: all separators become hyphens
        - Pinecone: all separators become hyphens + enforce 45-char limit
        """
        if vector_provider == "zilliz":
            # Zilliz only allows letters, numbers, and underscores
            adapted = collection_name.replace("-", "_").replace(".", "_")
        else:
            # Qdrant / Pinecone use hyphens
            adapted = collection_name.replace("_", "-").replace(".", "-")

        adapted = adapted.lower()

        # Pinecone: enforce 45-character limit
        if vector_provider == "pinecone" and len(adapted) > 45:
            import hashlib

            name_hash = hashlib.md5(collection_name.encode()).hexdigest()[:8]
            # Keep as much of the start as possible + hash for uniqueness
            adapted = f"{adapted[:36]}-{name_hash}"

        return adapted

    async def _ask_collection_name(self) -> str:
        """
        Ask user for a collection name prefix.

        Rules:
        - Press Enter for 'default'
        - Max 10 characters
        - Only alphabets allowed
        """
        self.console.print(
            "\n[bold cyan]Collection Name[/bold cyan] "
            "[dim](max 10 letters, alphabets only — press Enter for default)[/dim]"
        )

        while True:
            name = await questionary.text(
                "Collection name:",
                default="",
            ).ask_async()

            if not name or not name.strip():
                return "default"

            name = name.strip()

            if len(name) > 10:
                self.console.print(
                    "[red]Name must be 10 characters or less.[/red]"
                )
                continue

            if not name.isalpha():
                self.console.print(
                    "[red]Name must contain only alphabets (a-z).[/red]"
                )
                continue

            return name.lower()

    def _create_vector_store(self, config: VectorStoreConfig):
        """Create vector store instance."""
        if config.provider == "qdrant":
            return QdrantVectorStore(
                host=config.host,
                port=config.port,
                url=config.url,
                api_key=config.api_key,
                prefer_grpc=config.prefer_grpc,
            )
        elif config.provider == "pinecone":
            if not config.api_key:
                raise ValueError("Pinecone requires an API key")
            return PineconeVectorStore(
                api_key=config.api_key,
                cloud=config.cloud or "aws",
                region=config.region or "us-east-1",
            )
        elif config.provider == "zilliz":
            if not config.url or not config.api_key:
                raise ValueError("Zilliz requires both URI (url) and token (api_key)")
            return ZillizVectorStore(
                uri=config.url,
                token=config.api_key,
            )
        else:
            raise ValueError(f"Unsupported vector store: {config.provider}")

    def _create_embedding_provider_from_config(self, provider: str, provider_config):
        """Create embedding provider instance from provider-specific config."""
        if provider == "openai":
            return OpenAIEmbedding(
                api_key=provider_config.api_key, model=provider_config.model
            )
        elif provider == "voyageai":
            return VoyageEmbedding(
                api_key=provider_config.api_key, model=provider_config.model
            )
        elif provider == "openrouter":
            return OpenRouterEmbedding(
                api_key=provider_config.api_key,
                model=provider_config.model,
                http_referer=provider_config.http_referer,
                site_name=provider_config.site_name,
            )
        elif provider == "nvidia":
            return NvidiaEmbedding(
                api_key=provider_config.api_key,
                model=provider_config.model,
                base_url=provider_config.base_url,
                embedding_dimension=provider_config.embedding_dimension,
            )
        elif provider == "gemini":
            return GeminiEmbedding(
                api_key=provider_config.api_key,
                model=provider_config.model,
                embedding_dimension=provider_config.embedding_dimension,
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _create_embedding_provider(self, embedding_config: EmbeddingConfig):
        """Create embedding provider instance from full embedding config."""
        active_config = embedding_config.get_active_config()
        return self._create_embedding_provider_from_config(
            embedding_config.active_provider, active_config
        )

    async def start_mcp_server(self, config: Optional[Config] = None):
        """Start MCP server. Raises exceptions on errors instead of exiting."""
        if config is None:
            config = self.config_manager.load()

        # Validate API key exists for active provider
        active_config = config.embedding.get_active_config()
        if not active_config.api_key or active_config.api_key.strip() == "":
            raise ValueError(
                f"API key for {config.embedding.active_provider} is missing or empty. Please configure your embedding provider."
            )

        # Get vector size for the current embedding model
        vector_size = self._get_vector_size(config.embedding)

        # Use collection name from config (set during setup / edit embedding)
        collection_name = config.vector_store.collection_name

        # Save embedding dimension to provider config if it was auto-detected
        config_changed = False
        if not active_config.embedding_dimension:
            active_config.embedding_dimension = vector_size
            config_changed = True

        if config_changed:
            self.config_manager.save(config)

        # Create instances with better error messages
        try:
            vector_store = self._create_vector_store(config.vector_store)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to vector database: {str(e)}")

        try:
            embedding_provider = self._create_embedding_provider(config.embedding)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize embedding provider (check API key): {str(e)}"
            )

        # Ensure collection exists for this embedding model
        if not vector_store.collection_exists(collection_name):
            vector_store.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                distance_metric=config.vector_store.distance_metric,
            )
            self.console.print(
                f"\n✓ [green]Created new collection: {collection_name}[/green]"
            )
        else:
            self.console.print(
                f"\n✓ [green]Using existing collection: {collection_name}[/green]"
            )

        # Display embedding information
        self.console.print(
            f"[bold cyan]Embedding Model:[/bold cyan] {config.embedding.active_provider}/{active_config.model}"
        )
        self.console.print(
            f"[bold cyan]Vector Size:[/bold cyan] {vector_size} dimensions"
        )

        # Create memory components
        encoder = MemoryEncoder(embedding_provider)
        storage = MemoryStorage(vector_store, collection_name)
        retrieval = MemoryRetrieval(vector_store, collection_name)

        # Create handlers
        handlers = MemoryHandlers(
            encoder=encoder,
            storage=storage,
            retrieval=retrieval,
            search_limit=config.search.default_limit,
        )

        # Create and start server
        server = MemoryMCPServer(handlers)

        # Print server info
        url = f"http://{config.mcp.host}:{config.mcp.port}/mcp"

        panel = Panel(
            f"[bold green]MCP Server URL:[/bold green]\n\n[cyan]{url}[/cyan]\n\n"
            "[dim]Add this URL to your MCP client configuration.[/dim]",
            title="EzMemory MCP Server",
            border_style=ACCENT_COLOR,
            box=box.ROUNDED,
        )

        self.console.print("\n")
        self.console.print(panel)
        self.console.print("\n[bold]Available Tools[/bold]")
        tools_table = Table(
            box=box.MINIMAL, show_header=True, header_style=ACCENT_COLOR
        )
        tools_table.add_column("Tool", style="bold")
        tools_table.add_column("Description", style="dim")
        tools_table.add_row("add_memory", "Store new memories")
        tools_table.add_row("search_memory", "Search for relevant memories")
        tools_table.add_row("list_memory", "List all stored memories")
        tools_table.add_row("delete_memory", "Delete specific memories")
        self.console.print(tools_table)
        self.console.print("\n[dim]Press Ctrl+C to stop the server[/dim]\n")

        # Run server (await the async method)
        await server.run(host=config.mcp.host, port=config.mcp.port)

    async def start_http_server(self, config: Optional[Config] = None):
        """Start HTTP server. Raises exceptions on errors instead of exiting."""
        if config is None:
            config = self.config_manager.load()

        # Validate API key exists for active provider
        active_config = config.embedding.get_active_config()
        if not active_config.api_key or active_config.api_key.strip() == "":
            raise ValueError(
                f"API key for {config.embedding.active_provider} is missing or empty. Please configure your embedding provider."
            )

        # Get vector size for the current embedding model
        vector_size = self._get_vector_size(config.embedding)

        # Use collection name from config (set during setup / edit embedding)
        collection_name = config.vector_store.collection_name

        # Save embedding dimension to provider config if it was auto-detected
        config_changed = False
        if not active_config.embedding_dimension:
            active_config.embedding_dimension = vector_size
            config_changed = True

        if config_changed:
            self.config_manager.save(config)

        # Create instances with better error messages
        try:
            vector_store = self._create_vector_store(config.vector_store)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to vector database: {str(e)}")

        try:
            embedding_provider = self._create_embedding_provider(config.embedding)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize embedding provider (check API key): {str(e)}"
            )

        # Ensure collection exists for this embedding model
        if not vector_store.collection_exists(collection_name):
            vector_store.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                distance_metric=config.vector_store.distance_metric,
            )
            self.console.print(
                f"\n✓ [green]Created new collection: {collection_name}[/green]"
            )
        else:
            self.console.print(
                f"\n✓ [green]Using existing collection: {collection_name}[/green]"
            )

        # Display embedding information
        self.console.print(
            f"[bold cyan]Embedding Model:[/bold cyan] {config.embedding.active_provider}/{active_config.model}"
        )
        self.console.print(
            f"[bold cyan]Vector Size:[/bold cyan] {vector_size} dimensions"
        )

        # Create memory components
        encoder = MemoryEncoder(embedding_provider)
        storage = MemoryStorage(vector_store, collection_name)
        retrieval = MemoryRetrieval(vector_store, collection_name)

        # Create handlers
        handlers = MemoryHandlers(
            encoder=encoder,
            storage=storage,
            retrieval=retrieval,
            search_limit=config.search.default_limit,
        )

        # Create HTTP server
        http_server = HTTPServer(handlers)
        app = http_server.get_app()

        # Print server info
        url = f"http://localhost:6789"

        panel = Panel(
            f"[bold green]HTTP Server URL:[/bold green]\n\n[cyan]{url}[/cyan]\n\n"
            "[dim]API Docs available at http://localhost:6789/docs[/dim]",
            title="EzMemory HTTP Server",
            border_style=ACCENT_COLOR,
            box=box.ROUNDED,
        )

        self.console.print("\n")
        self.console.print(panel)
        self.console.print("\n[bold]Available Endpoints[/bold]")
        endpoints_table = Table(
            box=box.MINIMAL, show_header=True, header_style=ACCENT_COLOR
        )
        endpoints_table.add_column("Endpoint", style="bold")
        endpoints_table.add_column("Method", style="dim")
        endpoints_table.add_column("Description", style="dim")
        endpoints_table.add_row("/api/add_memory", "POST", "Store new memories")
        endpoints_table.add_row("/api/search_memory", "POST", "Search for relevant memories")
        endpoints_table.add_row("/api/list_memory", "POST", "List all stored memories")
        endpoints_table.add_row("/api/delete_all_memory", "POST", "Delete all memories")
        endpoints_table.add_row("/health", "GET", "Health check")
        self.console.print(endpoints_table)
        self.console.print("\n[dim]Press Ctrl+C to stop the server[/dim]\n")

        # Start server with uvicorn (without reload)
        import uvicorn
        config = uvicorn.Config(app, host="0.0.0.0", port=6789, reload=False)
        server = uvicorn.Server(config)
        await server.serve()

    async def show_main_menu(self) -> str:
        """Show main menu and return user choice."""
        # Clear screen and show header only (no title/subtitle yet)
        self.console.clear()
        self.print_header()

        # Show a compact configuration summary for better UX
        try:
            config = self.config_manager.load()
        except FileNotFoundError:
            config = None

        if config:
            active_embedding = config.embedding.get_active_config()
            
            # Modern table-based summary layout
            summary_table = Table(
                show_header=False,
                box=box.MINIMAL,
                padding=(0, 2),
                show_edge=False,
            )
            # Use flexible column widths so long values don't get truncated,
            # while keeping the same 4-column layout.
            summary_table.add_column(style="bold cyan", width=12, no_wrap=True)
            summary_table.add_column(style="", ratio=2)
            summary_table.add_column(style="bold cyan", width=12, no_wrap=True)
            summary_table.add_column(style="", ratio=3)
            
            summary_table.add_row(
                "Vector DB",
                config.vector_store.provider.title(),
                "Collection",
                config.vector_store.collection_name,
            )
            summary_table.add_row(
                "Embedding",
                f"{config.embedding.active_provider}/{active_embedding.model}",
                "Dimension",
                str(active_embedding.embedding_dimension or "auto"),
            )

            if self.last_status:
                summary_table.add_row(
                    "Status",
                    f"[dim]{self.last_status}[/dim]",
                    "",
                    "",
                )

            summary_panel = Panel(
                summary_table,
                title="[bold]Current Configuration[/bold]",
                border_style=ACCENT_COLOR,
                box=box.ROUNDED,
                padding=(1, 2),
            )
            self.console.print(summary_panel)

        # Compact menu section
        self.console.print()
        self.console.print("   [bold cyan]Main Menu[/bold cyan]")

        choice = await questionary.select(
            "",
            choices=[
                "Start FastMCP Server",
                "Start HTTP Server",
                "Edit Embedding Model",
                "Edit Vector Database",
                "Memory",
                "Add to Agent",
                "View Configuration",
                "Quit",
            ],
            style=questionary.Style([
                ('question', 'bold fg:cyan'),
                ('selected', 'fg:cyan bold'),
                ('pointer', 'fg:cyan'),
            ]),
        ).ask_async()

        return choice

    def _format_secret_status(self, value: Optional[str]) -> str:
        """Return a safe display value for secrets."""
        if value and value.strip():
            return "set"
        return "not set"

    async def add_to_agent(self):
        """Add EzMemory MCP to supported agents."""
        self._render_page("Add to Agent")

        choice = await questionary.select(
            "Select agent to configure:",
            choices=[
                "Cursor",
                "VS Code",
                "Claude Desktop (coming soon)",
                "Back",
            ],
        ).ask_async()

        if choice == "Cursor":
            try:
                added = add_cursor_mcp()
                if added:
                    self.console.print(
                        "\n[green]MCP server added to Cursor successfully ✓[/green]\n"
                    )
                else:
                    self.console.print(
                        "\n[green]MCP server already added to Cursor.[/green]\n"
                    )
            except Exception as e:
                self.console.print(
                    f"\n[red]Failed to update Cursor MCP config: {e}[/red]\n"
                )
        elif choice == "VS Code":
            try:
                added = add_vscode_mcp()
                if added:
                    self.console.print(
                        "\n[green]MCP server added to VS Code successfully ✓[/green]\n"
                    )
                else:
                    self.console.print(
                        "\n[green]MCP server already added to VS Code.[/green]\n"
                    )
            except Exception as e:
                self.console.print(
                    f"\n[red]Failed to update VS Code MCP config: {e}[/red]\n"
                )
        elif choice == "Claude Desktop (coming soon)":
            self.console.print(
                "\n[yellow]Support for this agent will be added soon.[/yellow]\n"
            )
        # "Back" simply returns to main menu

    async def view_configs(self):
        """Display current configuration settings."""
        self._render_page("Current Configuration")

        try:
            config = self.config_manager.load()
        except FileNotFoundError:
            self.console.print("[red]✗ Configuration file not found.[/red]")
            self.console.print(
                "[yellow]Run setup to create a configuration first.[/yellow]\n"
            )
            return

        active_embedding = config.embedding.get_active_config()

        table = Table(show_header=False, box=box.ROUNDED)
        table.add_column("Setting", style="bold")
        table.add_column("Value")

        table.add_row("Collection", config.vector_store.collection_name)
        table.add_row(
            "Embedding",
            f"{config.embedding.active_provider}/{active_embedding.model}",
        )
        table.add_row(
            "Embedding Dimension",
            str(active_embedding.embedding_dimension or "auto"),
        )
        table.add_row("Vector DB", config.vector_store.provider)
        if config.vector_store.provider == "qdrant":
            if config.vector_store.url:
                table.add_row("Qdrant URL", config.vector_store.url)
            else:
                table.add_row(
                    "Qdrant Host",
                    f"{config.vector_store.host}:{config.vector_store.port}",
                )
            table.add_row(
                "Prefer gRPC",
                "yes" if config.vector_store.prefer_grpc else "no",
            )
        elif config.vector_store.provider == "pinecone":
            table.add_row(
                "Pinecone Cloud",
                f"{config.vector_store.cloud}/{config.vector_store.region}",
            )
        elif config.vector_store.provider == "zilliz":
            if config.vector_store.url:
                table.add_row("Zilliz URI", config.vector_store.url)

        table.add_row("Distance Metric", config.vector_store.distance_metric)
        table.add_row("Search Limit", str(config.search.default_limit))
        table.add_row("Score Threshold", str(config.search.score_threshold or "none"))
        table.add_row("HNSW EF", str(config.search.hnsw_ef or "default"))
        table.add_row("Memory TTL (days)", str(config.memory.ttl_days))
        table.add_row(
            "Memory Decay", "enabled" if config.memory.enable_decay else "disabled"
        )
        table.add_row("Decay Factor", str(config.memory.decay_factor))
        table.add_row("MCP Host", config.mcp.host)
        table.add_row("MCP Port", str(config.mcp.port))
        table.add_row("MCP Auto Start", "yes" if config.mcp.auto_start else "no")

        if config.embedding.active_provider == "openai":
            table.add_row(
                "OpenAI API Key", self._format_secret_status(active_embedding.api_key)
            )
        elif config.embedding.active_provider == "voyageai":
            table.add_row(
                "VoyageAI API Key", self._format_secret_status(active_embedding.api_key)
            )
        elif config.embedding.active_provider == "openrouter":
            table.add_row(
                "OpenRouter API Key",
                self._format_secret_status(active_embedding.api_key),
            )
            table.add_row(
                "OpenRouter Base URL",
                getattr(active_embedding, "base_url", ""),
            )
            table.add_row(
                "HTTP Referer",
                getattr(active_embedding, "http_referer", None) or "none",
            )
            table.add_row(
                "Site Name",
                getattr(active_embedding, "site_name", None) or "none",
            )
        elif config.embedding.active_provider == "nvidia":
            table.add_row(
                "NVIDIA API Key",
                self._format_secret_status(active_embedding.api_key),
            )
            table.add_row(
                "NVIDIA Base URL",
                getattr(active_embedding, "base_url", ""),
            )
        elif config.embedding.active_provider == "gemini":
            table.add_row(
                "Gemini API Key",
                self._format_secret_status(active_embedding.api_key),
            )

        table.add_row(
            "Vector DB API Key", self._format_secret_status(config.vector_store.api_key)
        )

        panel = Panel(
            table,
            title="Configuration Summary",
            border_style=ACCENT_COLOR,
            box=box.ROUNDED,
        )
        self.console.print(panel)
        self.console.print(f"[dim]Config file: {CONFIG_FILE}[/dim]\n")
        await questionary.select(
            "Press back to return",
            choices=["Back"],
        ).ask_async()

    async def memory_menu(self):
        """Show Memory submenu: View Memory, Delete All Memory, Add Memory."""
        while True:
            self._render_page("Memory")

            choice = await questionary.select(
                "Select an option:",
                choices=[
                    "View Memory",
                    "Delete All Memory",
                    "Add Memory",
                    "Back",
                ],
            ).ask_async()

            if choice == "Back":
                break
            if choice == "View Memory":
                await self.view_memory()
            elif choice == "Delete All Memory":
                await self.delete_all_memory()
            elif choice == "Add Memory":
                await self.add_memory()

    async def view_memory(self):
        """Display all memories from the current collection."""
        self._render_page("View Memory")

        try:
            config = self.config_manager.load()
        except FileNotFoundError:
            self.console.print("[red]✗ Configuration file not found.[/red]")
            self.console.print(
                "[yellow]Run setup to create a configuration first.[/yellow]\n"
            )
            return

        # Validate API key exists for active provider
        active_config = config.embedding.get_active_config()
        if not active_config.api_key or active_config.api_key.strip() == "":
            self.console.print(
                f"[red]✗ API key for {config.embedding.active_provider} is missing or empty.[/red]"
            )
            self.console.print(
                "[yellow]Please configure your embedding provider first.[/yellow]\n"
            )
            return

        # Create vector store and storage instances
        try:
            vector_store = self._create_vector_store(config.vector_store)
        except Exception as e:
            self.console.print(f"[red]✗ Failed to connect to vector database: {e}[/red]\n")
            return

        storage = MemoryStorage(vector_store, config.vector_store.collection_name)

        # Get memory count
        try:
            total_count = storage.count()
        except Exception as e:
            self.console.print(f"[red]✗ Failed to count memories: {e}[/red]\n")
            return

        if total_count == 0:
            self.console.print("[yellow]No memories found in the collection.[/yellow]")
            self.console.print(f"[dim]Collection: {config.vector_store.collection_name}[/dim]\n")
            await questionary.select(
                "Press back to return",
                choices=["Back"],
            ).ask_async()
            return

        # Display collection info
        self.console.print(f"[bold cyan]Collection:[/bold cyan] {config.vector_store.collection_name}")
        self.console.print(f"[bold cyan]Total Memories:[/bold cyan] {total_count}\n")

        # Pagination settings
        page_size = 10
        offset = 0

        while True:
            # Get memories for current page
            try:
                memories = storage.get_all(limit=page_size, offset=offset)
            except Exception as e:
                self.console.print(f"[red]✗ Failed to retrieve memories: {e}[/red]\n")
                break

            if not memories:
                if offset == 0:
                    self.console.print("[yellow]No memories found.[/yellow]\n")
                else:
                    self.console.print("[yellow]No more memories to display.[/yellow]\n")
                break

            # Display memories in a table format
            table = Table(
                show_header=True,
                header_style="bold cyan",
                box=box.ROUNDED,
                show_lines=False,
            )
            table.add_column("#", style="dim", width=4, justify="right")
            table.add_column("Content", style="", ratio=3)
            table.add_column("Created", style="dim", width=18)
            table.add_column("Tags", style="dim", ratio=1)

            for idx, memory in enumerate(memories):
                # Truncate content for table display
                content_preview = memory.content
                if len(content_preview) > 80:
                    content_preview = content_preview[:77] + "..."

                # Format date/time
                date_str = memory.created_at.strftime("%Y-%m-%d %H:%M:%S")

                # Format tags
                tags_str = ", ".join(memory.metadata.tags[:2])  # Show first 2 tags
                if len(memory.metadata.tags) > 2:
                    tags_str += f" (+{len(memory.metadata.tags) - 2})"
                if not tags_str:
                    tags_str = "[dim]-[/dim]"

                table.add_row(
                    str(offset + idx + 1),
                    content_preview,
                    date_str,
                    tags_str,
                )

            self.console.print(table)
            self.console.print()

            # Show pagination info
            current_page = (offset // page_size) + 1
            total_pages = (total_count + page_size - 1) // page_size
            showing = f"Showing {offset + 1}-{offset + len(memories)} of {total_count}"
            self.console.print(f"[dim]{showing}[/dim]")

            # Navigation options
            choices = []
            if offset + page_size < total_count:
                choices.append("Next Page")
            if offset > 0:
                choices.append("Previous Page")
            choices.append("View Details")
            choices.append("Back")

            nav_choice = await questionary.select(
                "Select an option:",
                choices=choices,
            ).ask_async()

            if nav_choice == "Next Page":
                offset += page_size
                self.console.clear()
                self.print_header()
                self.console.print("[bold]View Memory[/bold]\n")
            elif nav_choice == "Previous Page":
                offset = max(0, offset - page_size)
                self.console.clear()
                self.print_header()
                self.console.print("[bold]View Memory[/bold]\n")
            elif nav_choice == "View Details":
                # Show detailed view of a specific memory
                memory_choices = [
                    f"Memory #{offset + i + 1}: {mem.content[:50]}..."
                    for i, mem in enumerate(memories)
                ]
                memory_choices.append("Back to List")

                detail_choice = await questionary.select(
                    "Select a memory to view details:",
                    choices=memory_choices,
                ).ask_async()

                if detail_choice != "Back to List":
                    # Extract memory index
                    mem_idx = memory_choices.index(detail_choice)
                    memory = memories[mem_idx]
                    current_page = (offset // page_size) + 1
                    total_pages = (total_count + page_size - 1) // page_size

                    # Display detailed memory (show page number)
                    self.console.print()
                    detail_panel = Panel(
                        format_memory_for_display(memory, offset + mem_idx),
                        title=f"[bold]Memory Details[/bold] — Page {current_page} of {total_pages}",
                        border_style=ACCENT_COLOR,
                        box=box.ROUNDED,
                        padding=(1, 2),
                    )
                    self.console.print(detail_panel)
                    self.console.print()

                    action = await questionary.select(
                        "Press back to return",
                        choices=["Back", "Delete"],
                    ).ask_async()

                    if action == "Delete":
                        try:
                            storage.delete([memory.id])
                            self.console.print("[green]✓ Memory deleted.[/green]\n")
                            total_count = storage.count()
                        except Exception as e:
                            self.console.print(f"[red]✗ Failed to delete memory: {e}[/red]\n")

                    # Refresh the page
                    self.console.clear()
                    self.print_header()
                    self.console.print("[bold]View Memory[/bold]\n")
                    self.console.print(f"[bold cyan]Collection:[/bold cyan] {config.vector_store.collection_name}")
                    self.console.print(f"[bold cyan]Total Memories:[/bold cyan] {total_count}\n")
            else:  # Back
                break

    async def delete_all_memory(self):
        """Delete all memories from the current collection after confirmation."""
        self._render_page("Delete All Memory")

        try:
            config = self.config_manager.load()
        except FileNotFoundError:
            self.console.print("[red]✗ Configuration file not found.[/red]")
            self.console.print(
                "[yellow]Run setup to create a configuration first.[/yellow]\n"
            )
            return

        active_config = config.embedding.get_active_config()
        if not active_config.api_key or active_config.api_key.strip() == "":
            self.console.print(
                f"[red]✗ API key for {config.embedding.active_provider} is missing or empty.[/red]"
            )
            self.console.print(
                "[yellow]Please configure your embedding provider first.[/yellow]\n"
            )
            return

        try:
            vector_store = self._create_vector_store(config.vector_store)
        except Exception as e:
            self.console.print(f"[red]✗ Failed to connect to vector database: {e}[/red]\n")
            return

        storage = MemoryStorage(vector_store, config.vector_store.collection_name)

        try:
            total_count = storage.count()
        except Exception as e:
            self.console.print(f"[red]✗ Failed to count memories: {e}[/red]\n")
            return

        if total_count == 0:
            self.console.print("[yellow]No memories in the collection. Nothing to delete.[/yellow]")
            self.console.print(f"[dim]Collection: {config.vector_store.collection_name}[/dim]\n")
            await questionary.select(
                "Press back to return",
                choices=["Back"],
            ).ask_async()
            return

        self.console.print(
            f"[bold cyan]Collection:[/bold cyan] {config.vector_store.collection_name}"
        )
        self.console.print(
            f"[bold red]This will permanently delete all {total_count} memories.[/bold red]\n"
        )

        confirmed = await questionary.confirm(
            "Are you sure you want to delete all memories?",
            default=False,
        ).ask_async()

        if not confirmed:
            self.console.print("[dim]Cancelled.[/dim]\n")
            return

        try:
            storage.delete_all()
            self.console.print(
                f"[green]✓ Deleted all {total_count} memories from the collection.[/green]\n"
            )
        except Exception as e:
            self.console.print(f"[red]✗ Failed to delete memories: {e}[/red]\n")
            return

        self._set_status(f"Deleted all memories · Collection: {config.vector_store.collection_name}")
        await questionary.select(
            "Press back to return",
            choices=["Back"],
        ).ask_async()

    async def add_memory(self):
        """Add a new memory to the current collection."""
        self._render_page("Add Memory")

        try:
            config = self.config_manager.load()
        except FileNotFoundError:
            self.console.print("[red]✗ Configuration file not found.[/red]")
            self.console.print(
                "[yellow]Run setup to create a configuration first.[/yellow]\n"
            )
            return

        active_config = config.embedding.get_active_config()
        if not active_config.api_key or active_config.api_key.strip() == "":
            self.console.print(
                f"[red]✗ API key for {config.embedding.active_provider} is missing or empty.[/red]"
            )
            self.console.print(
                "[yellow]Please configure your embedding provider first.[/yellow]\n"
            )
            return

        try:
            vector_store = self._create_vector_store(config.vector_store)
        except Exception as e:
            self.console.print(f"[red]✗ Failed to connect to vector database: {e}[/red]\n")
            return

        try:
            embedding_provider = self._create_embedding_provider(config.embedding)
        except Exception as e:
            self.console.print(
                f"[red]✗ Failed to initialize embedding provider: {e}[/red]\n"
            )
            return

        storage = MemoryStorage(vector_store, config.vector_store.collection_name)
        encoder = MemoryEncoder(embedding_provider)

        self.console.print(
            f"[bold cyan]Collection:[/bold cyan] {config.vector_store.collection_name}\n"
        )

        content = await questionary.text(
            "Enter memory content:",
            default="",
        ).ask_async()

        if content is None or not (content and content.strip()):
            self.console.print("[dim]No content entered. Cancelled.[/dim]\n")
            return

        try:
            memory = Memory(content=content.strip())
            memory = encoder.encode(memory)
            memory_id = storage.store(memory)
            self.console.print(
                f"[green]✓ Memory added successfully.[/green]"
            )
            self.console.print(f"[dim]ID: {memory_id}[/dim]\n")
        except ValueError as e:
            self.console.print(f"[red]✗ {e}[/red]\n")
        except Exception as e:
            self.console.print(f"[red]✗ Failed to add memory: {e}[/red]\n")
            return

        self._set_status("Memory added")
        await questionary.select(
            "Press back to return",
            choices=["Back"],
        ).ask_async()

    async def edit_embedding_provider(self):
        """Edit embedding provider configuration."""
        self._render_page("Edit Embedding Provider")

        try:
            # Load current config
            config = self.config_manager.load()

            # ----------------------------------------------------------
            # If known collections exist, offer to select an existing one
            # ----------------------------------------------------------
            if config.known_collections:
                initial_choice = await questionary.select(
                    "Choose an option:",
                    choices=[
                        "Configure new embedding",
                        "Select existing collection",
                        questionary.Separator(),
                        "Back",
                    ],
                ).ask_async()

                if initial_choice == "Back":
                    return

                if initial_choice == "Select existing collection":
                    # Build display choices from known collections
                    col_choices = []
                    for name, info in config.known_collections.items():
                        label = f"{name}  ({info.provider}/{info.model}, {info.dim}d)"
                        col_choices.append(
                            questionary.Choice(title=label, value=name)
                        )
                    col_choices.append(questionary.Separator())
                    col_choices.append(
                        questionary.Choice(title="Back", value="__back__")
                    )

                    selected = await questionary.select(
                        "Select a collection:",
                        choices=col_choices,
                    ).ask_async()

                    if selected == "__back__":
                        return

                    # Apply the selected collection's settings
                    info = config.known_collections[selected]
                    config.embedding.active_provider = info.provider

                    # Update the provider-specific config
                    active_cfg = config.embedding.get_active_config()
                    active_cfg.model = info.model
                    active_cfg.embedding_dimension = info.dim
                    if info.provider == "openrouter" and hasattr(
                        active_cfg, "http_referer"
                    ):
                        active_cfg.http_referer = info.http_referer
                        active_cfg.site_name = info.site_name

                    # Adapt the collection name for the current vector DB
                    current_vdb = config.vector_store.provider
                    adapted_name = self._adapt_collection_name(
                        selected, current_vdb
                    )

                    if adapted_name != selected:
                        self.console.print(
                            f"\n[yellow]Collection name adapted for "
                            f"{current_vdb}: {selected} → {adapted_name}[/yellow]"
                        )
                        # Also register the adapted name in known_collections
                        config.known_collections[adapted_name] = info

                    config.vector_store.collection_name = adapted_name

                    # Ensure API key exists for the provider
                    if (
                        not active_cfg.api_key
                        or not active_cfg.api_key.strip()
                    ):
                        api_key = await questionary.password(
                            f"{info.provider.title()} API Key:",
                        ).ask_async()
                        active_cfg.api_key = api_key

                    # Ensure collection exists in vector store
                    self.console.print(
                        "\n[bold yellow]Setting up collection...[/bold yellow]\n"
                    )
                    vector_store = self._create_vector_store(config.vector_store)
                    if not vector_store.collection_exists(adapted_name):
                        vector_store.create_collection(
                            collection_name=adapted_name,
                            vector_size=info.dim,
                            distance_metric=config.vector_store.distance_metric,
                        )
                        self.console.print(
                            f"✓ [green]Created collection: {adapted_name}[/green]"
                        )
                    else:
                        self.console.print(
                            f"✓ [green]Collection already exists: {adapted_name}[/green]"
                        )

                    self.config_manager.save(config)

                    self.console.print(
                        f"\n✓ [green]Switched to collection: {adapted_name}[/green]"
                    )
                    self._set_status(
                        f"Collection switched · {info.provider}/{info.model} · "
                        f"Collection: {adapted_name}"
                    )

                    # Pause so user can read the results
                    self.console.print()
                    self.console.print(
                        "[dim]Press Enter to continue...[/dim]", end=""
                    )
                    await questionary.text("", default="").ask_async()
                    return

            # ----------------------------------------------------------
            # Normal flow: select provider → model → collection name
            # ----------------------------------------------------------
            embedding_provider = await self._select_embedding_provider()
            embedding_config = await self._configure_embedding(
                embedding_provider, existing_config=config.embedding
            )

            # Update embedding config
            config.embedding = embedding_config

            # Use detected dimension
            vector_size = self._get_vector_size(embedding_config)

            # Ask for collection name (optional)
            name_prefix = await self._ask_collection_name()

            # Generate full collection name
            collection_name = self._get_collection_name(
                embedding_config, config.vector_store.provider, name_prefix
            )
            config.vector_store.collection_name = collection_name

            # Create collection if needed
            self.console.print(
                "\n[bold yellow]Setting up collection...[/bold yellow]\n"
            )
            vector_store = self._create_vector_store(config.vector_store)
            if not vector_store.collection_exists(collection_name):
                vector_store.create_collection(
                    collection_name=collection_name,
                    vector_size=vector_size,
                    distance_metric=config.vector_store.distance_metric,
                )
                self.console.print(
                    f"✓ [green]Created new collection: {collection_name}[/green]"
                )
            else:
                self.console.print(
                    f"✓ [green]Collection already exists: {collection_name}[/green]"
                )

            # Save to known_collections
            active = embedding_config.get_active_config()
            config.known_collections[collection_name] = KnownCollectionInfo(
                provider=embedding_config.active_provider,
                model=active.model,
                dim=vector_size,
                http_referer=getattr(active, "http_referer", None),
                site_name=getattr(active, "site_name", None),
            )

            # Save updated config
            self.config_manager.save(config)

            self.console.print(
                f"\n✓ [green]Embedding provider updated successfully[/green]"
            )
            self.console.print(
                f"[dim]Model and other settings can be edited in: {CONFIG_FILE}[/dim]\n"
            )

            self._set_status(
                f"Embedding updated · {embedding_config.active_provider}/{active.model} · "
                f"Collection: {collection_name}"
            )

            # Pause so user can read the results
            self.console.print()
            self.console.print("[dim]Press Enter to continue...[/dim]", end="")
            await questionary.text("", default="").ask_async()
        except BackToMainMenu:
            self.console.print("[yellow]Returning to main menu...[/yellow]\n")
            return
        except Exception as e:
            self.console.print(
                f"\n✗ [red]Failed to update embedding provider: {e}[/red]"
            )
            self.console.print("[yellow]Returning to main menu...[/yellow]\n")
            # Method returns, allowing menu loop to continue

    async def edit_vector_database(self):
        """Edit vector database configuration."""
        self._render_page("Edit Vector Database")

        try:
            # Select new vector database
            vector_db = await self._select_vector_database()
            vector_config = await self._configure_vector_store(vector_db)

            # Load current config
            config = self.config_manager.load()

            # Test the new vector store
            self.console.print(
                "\n[bold yellow]Testing vector database connection...[/bold yellow]\n"
            )
            vector_store = self._create_vector_store(vector_config)
            self.console.print(
                f"✓ [green]Connected to {vector_db.title()} successfully[/green]"
            )

            # Update vector store config (preserve collection name)
            old_collection = config.vector_store.collection_name
            config.vector_store = vector_config
            config.vector_store.collection_name = old_collection

            # Get vector size for current embedding model
            vector_size = self._get_vector_size(config.embedding)

            # Regenerate collection name for the new vector provider format
            collection_name = self._get_collection_name(
                config.embedding, config.vector_store.provider, "default"
            )
            config.vector_store.collection_name = collection_name

            # Create collection if needed
            if not vector_store.collection_exists(collection_name):
                vector_store.create_collection(
                    collection_name=collection_name,
                    vector_size=vector_size,
                    distance_metric=vector_config.distance_metric,
                )
                self.console.print(
                    f"✓ [green]Created collection: {collection_name}[/green]"
                )
            else:
                self.console.print(
                    f"✓ [green]Collection already exists: {collection_name}[/green]"
                )

            # Save updated config
            self.config_manager.save(config)

            self.console.print(
                f"\n✓ [green]Vector database updated successfully[/green]"
            )
            if vector_config.provider == "qdrant":
                if vector_config.url:
                    self.console.print(
                        f"[bold cyan]Using:[/bold cyan] Qdrant Cloud at {vector_config.url}"
                    )
                else:
                    self.console.print(
                        f"[bold cyan]Using:[/bold cyan] Qdrant Local at {vector_config.host}:{vector_config.port}"
                    )
            elif vector_config.provider == "pinecone":
                self.console.print(
                    f"[bold cyan]Using:[/bold cyan] Pinecone ({vector_config.cloud}/{vector_config.region})"
                )
            elif vector_config.provider == "zilliz":
                if vector_config.url:
                    self.console.print(
                        f"[bold cyan]Using:[/bold cyan] Zilliz Cloud at {vector_config.url}"
                    )
            self.console.print(
                f"[dim]Vector database settings can be edited in: {CONFIG_FILE}[/dim]\n"
            )

            self._set_status(
                f"Vector DB updated · {vector_config.provider} · Collection: {collection_name}"
            )

            # Pause so user can read the results
            self.console.print()
            self.console.print("[dim]Press Enter to continue...[/dim]", end="")
            await questionary.text("", default="").ask_async()
        except BackToMainMenu:
            self.console.print("[yellow]Returning to main menu...[/yellow]\n")
            return
        except Exception as e:
            self.console.print(f"\n✗ [red]Failed to update vector database: {e}[/red]")
            self.console.print("[yellow]Returning to main menu...[/yellow]\n")
            # Method returns, allowing menu loop to continue

    async def run(self):
        """Main CLI entry point."""
        setup_logging()

        # Check if config exists
        if not CONFIG_FILE.exists():
            try:
                config = await self.setup_wizard()
            except BackToMainMenu:
                self.console.print("\n[yellow]Setup cancelled.[/yellow]\n")
                return

            # Ask to start MCP server after initial setup
            start_server = await questionary.confirm(
                "Start MCP server?",
                default=True,
            ).ask_async()

            if start_server:
                await self.start_mcp_server(config)
            else:
                self.console.print(
                    "\n[yellow]Setup complete! Run again to start the MCP server.[/yellow]\n"
                )
        else:
            # Config exists, show menu in a loop
            while True:
                choice = await self.show_main_menu()

                if choice == "Start FastMCP Server":
                    self.console.print("\n[green]Starting MCP server...[/green]\n")
                    try:
                        config = self.config_manager.load()
                        await self.start_mcp_server(config)
                        break  # Exit loop after starting server successfully
                    except FileNotFoundError:
                        self.console.print(
                            "\n[red]✗ Configuration file not found.[/red]"
                        )
                        self.console.print(
                            "[yellow]Please run initial setup or check your configuration.[/yellow]\n"
                        )
                        # Loop continues, showing menu again
                    except ValueError as e:
                        self.console.print(f"\n[red]✗ Configuration Error: {e}[/red]")
                        self.console.print(
                            "[yellow]Please edit your configuration and set the required values properly.[/yellow]\n"
                        )
                        # Loop continues, showing menu again
                    except RuntimeError as e:
                        self.console.print(f"\n[red]✗ Server Start Failed: {e}[/red]")
                        self.console.print(
                            "[yellow]Please check your configuration and try again.[/yellow]\n"
                        )
                        # Loop continues, showing menu again
                    except Exception as e:
                        self.console.print(f"\n[red]✗ Unexpected Error: {e}[/red]")
                        self.console.print(
                            "[yellow]Please check your configuration and try again.[/yellow]\n"
                        )
                        # Loop continues, showing menu again
                elif choice == "Start HTTP Server":
                    self.console.print("\n[green]Starting HTTP server...[/green]\n")
                    try:
                        config = self.config_manager.load()
                        await self.start_http_server(config)
                        break  # Exit loop after starting server successfully
                    except FileNotFoundError:
                        self.console.print(
                            "\n[red]✗ Configuration file not found.[/red]"
                        )
                        self.console.print(
                            "[yellow]Please run initial setup or check your configuration.[/yellow]\n"
                        )
                        # Loop continues, showing menu again
                    except ValueError as e:
                        self.console.print(f"\n[red]✗ Configuration Error: {e}[/red]")
                        self.console.print(
                            "[yellow]Please edit your configuration and set the required values properly.[/yellow]\n"
                        )
                        # Loop continues, showing menu again
                    except RuntimeError as e:
                        self.console.print(f"\n[red]✗ Server Start Failed: {e}[/red]")
                        self.console.print(
                            "[yellow]Please check your configuration and try again.[/yellow]\n"
                        )
                        # Loop continues, showing menu again
                    except Exception as e:
                        self.console.print(f"\n[red]✗ Unexpected Error: {e}[/red]")
                        self.console.print(
                            "[yellow]Please check your configuration and try again.[/yellow]\n"
                        )
                        # Loop continues, showing menu again
                elif choice == "Add to Agent":
                    await self.add_to_agent()
                    # Loop continues, showing menu again
                elif choice == "View Configuration":
                    await self.view_configs()
                    # Loop continues, showing menu again
                elif choice == "Memory":
                    await self.memory_menu()
                    # Loop continues, showing menu again
                elif choice == "Edit Embedding Model":
                    await self.edit_embedding_provider()
                    # Loop continues, showing menu again
                elif choice == "Edit Vector Database":
                    await self.edit_vector_database()
                    # Loop continues, showing menu again
                elif choice == "Quit":
                    self.console.print("\n[yellow]Goodbye![/yellow]\n")
                    break  # Exit loop


def main():
    """Main entry point."""
    cli = EzMemoryCLI()

    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Shutting down gracefully...[/yellow]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
