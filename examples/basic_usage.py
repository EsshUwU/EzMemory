"""
Basic usage example for EzMemory.

This example shows how to use EzMemory programmatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory_service.config import config_manager
from src.memory_service.embeddings import OpenAIEmbedding, VoyageEmbedding, OpenRouterEmbedding
from src.memory_service.vector_store import QdrantVectorStore
from src.memory_service.memory import Memory, MemoryEncoder, MemoryStorage, MemoryRetrieval


def main():
    """Run basic usage example."""
    print("🧠 EzMemory - Basic Usage Example\n")
    
    # Load configuration
    print("Loading configuration...")
    try:
        config = config_manager.load()
        print(f"✓ Configuration loaded from {config_manager.config_path}\n")
    except FileNotFoundError:
        print("✗ Configuration not found. Please run setup first:")
        print("  python ezmemory.py\n")
        sys.exit(1)
    
    # Initialize embedding provider
    print(f"Initializing {config.embedding.provider} embedding provider...")
    if config.embedding.provider == "openai":
        embedding_provider = OpenAIEmbedding(
            api_key=config.embedding.api_key,
            model=config.embedding.model
        )
    elif config.embedding.provider == "voyageai":
        embedding_provider = VoyageEmbedding(
            api_key=config.embedding.api_key,
            model=config.embedding.model
        )
    elif config.embedding.provider == "openrouter":
        embedding_provider = OpenRouterEmbedding(
            api_key=config.embedding.api_key,
            model=config.embedding.model,
            http_referer=config.embedding.http_referer,
            site_name=config.embedding.site_name,
        )
    else:
        print(f"✗ Unsupported embedding provider: {config.embedding.provider}")
        sys.exit(1)
    
    print(f"✓ Using model: {config.embedding.model}\n")
    
    # Initialize vector store
    print("Initializing Qdrant vector store...")
    vector_store = QdrantVectorStore(
        host=config.vector_store.host,
        port=config.vector_store.port,
        url=config.vector_store.url,
        api_key=config.vector_store.api_key,
        prefer_grpc=config.vector_store.prefer_grpc,
    )
    print(f"✓ Connected to Qdrant\n")
    
    # Initialize memory components
    encoder = MemoryEncoder(embedding_provider)
    storage = MemoryStorage(vector_store, config.vector_store.collection_name)
    retrieval = MemoryRetrieval(vector_store, config.vector_store.collection_name)
    
    print("=" * 60)
    print("ADDING MEMORIES")
    print("=" * 60 + "\n")
    
    # Add some example memories
    memories_to_add = [
        "User prefers dark mode for all applications",
        "User's favorite programming language is Python",
        "User works as a software engineer at a tech company",
        "User enjoys hiking on weekends",
        "User is learning about machine learning and AI",
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
        print(f"   Created: {memory.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
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
    print("  - Start the MCP server: python ezmemory.py")
    print("  - Integrate with your AI agent")
    print("  - Explore configuration: ~/.ezmemory/config.json")


if __name__ == "__main__":
    main()
