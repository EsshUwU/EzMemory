"""
Basic SDK usage for EzMemory using the config file created by `python ezmemory.py`.

This example shows how to:
- Load EzMemory config
- Initialize the active embedding provider (OpenAI, VoyageAI, or OpenRouter)
- Initialize the configured vector store (Qdrant, Pinecone, or Zilliz)
- Add, search, list, and delete memories
"""

from __future__ import annotations

import sys

from ezmemory import (  # type: ignore[import]
    Memory,
    MemoryEncoder,
    MemoryRetrieval,
    MemoryStorage,
    OpenAIEmbedding,
    OpenRouterEmbedding,
    PineconeVectorStore,
    QdrantVectorStore,
    VoyageEmbedding,
    ZillizVectorStore,
    config_manager,
)


def _build_embedding_provider(config):
    provider = config.embedding.active_provider
    active = config.embedding.get_active_config()

    if provider == "openai":
        return OpenAIEmbedding(api_key=active.api_key, model=active.model)
    if provider == "voyageai":
        return VoyageEmbedding(api_key=active.api_key, model=active.model)
    if provider == "openrouter":
        return OpenRouterEmbedding(
            api_key=active.api_key,
            model=active.model,
            http_referer=getattr(active, "http_referer", None),
            site_name=getattr(active, "site_name", None),
        )
    raise SystemExit(f"Unsupported embedding provider in config: {provider}")


def _build_vector_store(config, embedding_dimension: int):
    provider = config.vector_store.provider.lower()

    if provider == "qdrant":
        vs = QdrantVectorStore(
            host=config.vector_store.host,
            port=config.vector_store.port,
            url=config.vector_store.url,
            api_key=config.vector_store.api_key,
            prefer_grpc=config.vector_store.prefer_grpc,
        )
    elif provider == "pinecone":
        if not config.vector_store.api_key:
            raise SystemExit("Pinecone API key missing in config.vector_store.api_key")
        vs = PineconeVectorStore(
            api_key=config.vector_store.api_key,
            cloud=config.vector_store.cloud or "aws",
            region=config.vector_store.region or "us-east-1",
        )
    elif provider == "zilliz":
        if not config.vector_store.url or not config.vector_store.api_key:
            raise SystemExit("Zilliz requires url and api_key in config.vector_store")
        vs = ZillizVectorStore(uri=config.vector_store.url, token=config.vector_store.api_key)
    else:
        raise SystemExit(f"Unsupported vector store provider in config: {provider}")

    # Create collection/index if needed
    if not vs.collection_exists(config.vector_store.collection_name):
        vs.create_collection(
            collection_name=config.vector_store.collection_name,
            vector_size=embedding_dimension,
            distance_metric=config.vector_store.distance_metric,
        )
    return vs


def main():
    """Run basic usage example against the configured EzMemory instance."""
    print("🧠 EzMemory - SDK Basic Usage\n")

    # Load configuration
    print("Loading configuration from ~/.ezmemory/config.json ...")
    try:
        config = config_manager.load()
    except FileNotFoundError as exc:
        print("✗ Configuration not found. Please run setup first:")
        print("  python -m memory_service.main")
        raise SystemExit(1) from exc
    print("✓ Configuration loaded\n")

    # Initialize embedding provider
    embedding_provider = _build_embedding_provider(config)
    dim = embedding_provider.get_dimension()
    print(
        f"✓ Embedding provider: {config.embedding.active_provider} / "
        f"{embedding_provider.model_name} (dim={dim})\n"
    )

    # Initialize vector store
    vector_store = _build_vector_store(config, embedding_dimension=dim)
    print(f"✓ Vector store: {config.vector_store.provider} · "
          f"collection={config.vector_store.collection_name}\n")

    # Initialize memory components
    encoder = MemoryEncoder(embedding_provider)
    storage = MemoryStorage(vector_store, config.vector_store.collection_name)
    retrieval = MemoryRetrieval(vector_store, config.vector_store.collection_name)

    print("=" * 60)
    print("ADDING MEMORIES")
    print("=" * 60 + "\n")

    # Add some example memories
    memories_to_add = [
        "User prefers dark mode for all applications.",
        "User's favorite programming language is Python.",
        "User works as a software engineer at a tech company.",
        "User enjoys hiking on weekends.",
        "User is learning about machine learning and AI.",
    ]

    added_ids = []
    for content in memories_to_add:
        memory = Memory(content=content)
        memory = encoder.encode(memory)
        memory_id = storage.store(memory)
        added_ids.append(memory_id)
        print(f"✓ Added: {content}")
        print(f"  ID: {memory_id[:16]}...\n")

    print(f"Total memories added: {len(added_ids)}\n")

    print("=" * 60)
    print("SEARCHING MEMORIES")
    print("=" * 60 + "\n")

    # Search for memories
    queries = [
        "What are the user's preferences?",
        "Tell me about the user's hobbies",
        "What programming language does the user like?",
    ]

    for query in queries:
        print(f"Query: {query}")
        query_vector = encoder.encode_query(query)
        results = retrieval.search(query_vector, limit=3)

        print(f"Found {len(results)} relevant memories:\n")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.score:.4f}")
            print(f"     Content: {result.memory.content}")
            print(f"     ID: {result.memory.id[:16]}...\n")

        print("-" * 60 + "\n")

    print("=" * 60)
    print("LISTING ALL MEMORIES")
    print("=" * 60 + "\n")

    # List all memories
    all_memories = storage.get_all(limit=10)
    total_count = storage.count()

    print(f"Total memories in storage: {total_count}\n")
    for i, memory in enumerate(all_memories, 1):
        print(f"{i}. {memory.content}")
        print(f"   ID: {memory.id[:16]}...\n")

    print("=" * 60)
    print("DELETING A MEMORY")
    print("=" * 60 + "\n")

    # Delete one memory
    if added_ids:
        memory_to_delete = added_ids[0]
        print(f"Deleting memory: {memory_to_delete[:16]}...")
        storage.delete([memory_to_delete])
        print("✓ Memory deleted\n")

        new_count = storage.count()
        print(f"Remaining memories: {new_count}\n")

    print("=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60 + "\n")
    print("✓ All operations completed successfully!")
    print("\nNext steps:")
    print("  - Start the MCP server: ezmemory (CLI entry point)")
    print("  - Integrate with your AI agent via MCP or SDK")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
